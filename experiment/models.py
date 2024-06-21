import os
from evaluation_metrics import F1Score, ExactMatch, METEOR, BERTScore, SemanticAnswerSimilarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import lightning as L
from torch.nn import functional as F
import torch
import csv
import os
import numpy as np
import pandas as pd
from transformers import set_seed
from torchmetrics import MetricCollection
from subprocess import Popen, PIPE
from pathlib import Path
import sys
import json
from tqdm import tqdm
tqdm.pandas()

class ChoiModel(L.LightningModule):
    def __init__(self, model, tokenizer, fold, learning_rate=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.fold = fold
        self.tokenizer = tokenizer
        scalar_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score(),
                "qa_sas": SemanticAnswerSimilarity(),
                "qa_em": ExactMatch(),
                #"qa_meteor": METEOR(),
                "qa_bert_score": BERTScore()
            }
        )
        
        list_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score('list'),
                "qa_sas": SemanticAnswerSimilarity('list'),
                "qa_em": ExactMatch('list'),
                #"qa_meteor": METEOR('list'),
                "qa_bert_score": BERTScore('list')
            }
        )
        self.val_qa_metrics = scalar_qa_metrics.clone(prefix='val_')
        self.test_avg_qa_metrics = scalar_qa_metrics.clone(prefix='test_avg_')
        self.test_qa_metrics = list_qa_metrics.clone(prefix='test_')
        self.test_results = pd.DataFrame(columns=['predictions', 'labels'])

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_input_ids=decoder_input_ids)
        
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training

    def log_results(self, metrics, metrics_type, eval_method='test'):
        # Move metrics to the CPU and convert to a format that can be written to a CSV
        if not os.path.isfile(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv'):
            metrics_columns = [list(metrics.keys())]
            with open(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_columns)

        metrics_data=[]   
        for i in range(len(list(metrics.values())[0])):
            row = [tensor[i].item() for tensor in metrics.values()]
            metrics_data.append(row)

        # Write CSV data to the logger
        with open(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)
    
    def log_qa_predictions(self, qa_preds):
        # Log predictions
        csv_data = qa_preds.values.tolist()
        
        # If file does not exist, write column names as the first row
        if not os.path.isfile(self.logger.experiment.log_dir + '/test_qa_predictions.csv'):
            column_names = list(qa_preds.columns)
            with open(self.logger.experiment.log_dir + '/test_qa_predictions.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(column_names)

        # Write CSV data to the logger
        with open(self.logger.experiment.log_dir + '/test_qa_predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)        
        self.log("val_loss", outputs["loss"], prog_bar=True)
        generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )

        labels = batch["labels"]
        
        predictions = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.val_qa_metrics(decoded_preds, decoded_labels)
        self.log_dict(self.val_qa_metrics, prog_bar=True)

        
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)        
        self.log("test_loss", outputs["loss"], prog_bar=True)
        generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]
        
        predictions = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.test_avg_qa_metrics(decoded_preds, decoded_labels)
        self.log_dict(self.test_avg_qa_metrics, prog_bar=True)
        
        #TODO: add preds and labels to dataframe -> calc metrics and log when test epoch ends
        # Create a temporary DataFrame with the new data
        new_data = pd.DataFrame({"predictions": decoded_preds, "labels": decoded_labels})        
        # Append the new data to the original DataFrame
        self.test_results = pd.concat([self.test_results, new_data], ignore_index=True)


    def on_test_epoch_end(self):
        test_split = self.fold["test"].rename(columns={'global_turn_id': 'qid'})
        qa_metrics_results = self.test_qa_metrics(self.test_results['predictions'].astype(str).tolist(), self.test_results['labels'].astype(str).tolist())

        self.test_results = pd.concat([self.test_results, test_split], axis=1)
        
        if not self.trainer.sanity_checking:
            self.log_results(qa_metrics_results, "qa_metrics")
            self.log_qa_predictions(self.test_results)
        
        self.test_qa_metrics.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    
# code adapted from https://github.com/PhilippChr/CONVINSE/blob/main/convinse/heterogeneous_answering/fid_module/fid_module.py    
class FiD():
    def __init__(self, config, fold):
        self.config = config
        self.fold = fold
        
        self.path_to_fid_python_env = self.config['fid_env']
        self.path_to_fid = self.config['fid_path'] 
        
        self.train_path = fold['train_path']
        self.test_path = fold['test_path']
        
        #cuda_device = "1"
        #self.env = {"CUDA_VISIBLE_DEVICES": cuda_device}
        
        with open(self.test_path, 'r') as json_file:
            self.test_data = json.load(json_file)            

        
        list_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score('list'),
                "qa_sas": SemanticAnswerSimilarity('list'),
                "qa_em": ExactMatch('list'),
                #"qa_meteor": METEOR('list'),
                "qa_bert_score": BERTScore('list')
            }
        ).to("cuda")
        
        self.test_qa_metrics = list_qa_metrics.clone(prefix='test_')
    
    def set_logger(self, logger):
        self.logger = logger
        
    def set_checkpoint_path(self, path):
        self.checkpoint_path = path
    
    def train(self):        
        COMMAND = [self.path_to_fid_python_env, f"{self.path_to_fid}/train_reader.py"]
        COMMAND += ["--name", self.checkpoint_path]
        COMMAND += ["--train_data", self.train_path]
        #COMMAND += ["--use_checkpoint"]
        #COMMAND += ["--model_path", "../FiD/pretrained_models/nq_reader_base/"]
        #COMMAND += ["--model_path", "wyu1/FiD-NQ"]
        COMMAND += ["--eval_data", self.test_path]
        COMMAND += ["--model_size", "base"]
        COMMAND += ["--lr", str(self.config["fid_lr"])]
        COMMAND += ["--optim", str(self.config["fid_optim"])]
        COMMAND += ["--scheduler", str(self.config["fid_scheduler"])]
        COMMAND += ["--weight_decay", str(self.config["fid_weight_decay"])]
        COMMAND += ["--text_maxlength", str(self.config["fid_text_maxlength"])]
        COMMAND += ["--answer_maxlength", str(self.config["fid_answer_maxlength"])]
        COMMAND += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        COMMAND += ["--n_context", str(self.config["fid_max_evidences"])]
        COMMAND += ["--total_step", str(self.config["fid_total_step"])]
        COMMAND += ["--warmup_step", str(self.config["fid_warmup_step"])]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr, #env=self.env
                        )
        process.communicate()
        
        
    def test(self):
        COMMAND = [self.path_to_fid_python_env, f"{self.path_to_fid}/test_reader.py"]
        COMMAND += ["--name", self.checkpoint_path]
        COMMAND += ["--model_path", f"checkpoint/{self.checkpoint_path}/checkpoint/step-15000/"]
        COMMAND += ["--checkpoint_dir", f"checkpoint/"]
        COMMAND += ["--eval_data", self.test_path]
        COMMAND += ["--n_context", str(self.config["fid_max_evidences"])]
        COMMAND += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        COMMAND += ["--write_results"]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr, #env=self.env
                        )
        process.communicate()
    
    def log_results(self, metrics, metrics_type, eval_method='test'):
        # Move metrics to the CPU and convert to a format that can be written to a CSV
        if not os.path.isfile(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv'):
            metrics_columns = [list(metrics.keys())]
            os.makedirs(self.logger.log_dir, exist_ok=True)
            with open(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_columns)

        metrics_data=[]   
        for i in range(len(list(metrics.values())[0])):
            row = [tensor[i].item() for tensor in metrics.values()]
            metrics_data.append(row)

        # Write CSV data to the logger
        with open(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)
    
    def log_qa_predictions(self, qa_preds):
        # Log predictions
        csv_data = qa_preds.values.tolist()
        
        # If file does not exist, write column names as the first row
        if not os.path.isfile(self.logger.log_dir + '/test_qa_predictions.csv'):
            column_names = list(qa_preds.columns)
            with open(self.logger.log_dir + '/test_qa_predictions.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(column_names)

        # Write CSV data to the logger
        with open(self.logger.log_dir + '/test_qa_predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    
    def evaluate_results(self):
        generated_answers = self._parse_result()
        target_answers = []
        generated_ans = []
        question_ids = []
        recipe_ids = []
        questions = []

        for entry in self.test_data:
            question_id = entry['id']
            question = entry['question']
            recipe_id = entry['recipe_id']
            target_answer = entry['target']
            generated_answer = generated_answers.get(question_id, "")

            target_answers.append(target_answer)
            generated_ans.append(generated_answer)
            question_ids.append(question_id)
            recipe_ids.append(recipe_id)
            questions.append(question)

        evaluation_results_df = pd.DataFrame({
            'question_id': question_ids,
            'recipe_id': recipe_ids,
            'question': questions,
            'target_answer': target_answers,
            'generated_answer': generated_ans
        })

        qa_metrics_results = self.test_qa_metrics(evaluation_results_df['generated_answer'].tolist(), evaluation_results_df['target_answer'].tolist())
        self.log_results(qa_metrics_results, "qa_metrics")
        self.log_qa_predictions(evaluation_results_df)

    def _parse_result(self):
        """
        Parse the output generated by FiD, and add predicted
        (and generated) answers to the data.
        """
        generated_answers = {}
        with open(f"checkpoint/{self.checkpoint_path}/final_output.txt", "r") as fp:
            line = fp.readline()
            while line:
                try:
                    question_id, answer = line.split(None, 1)
                except:
                    question_id = line.strip()
                    answer = ""
                question_id = question_id.strip()
                answer = answer.strip()
                generated_answers[question_id] = answer
                line = fp.readline()
        return generated_answers

    
class MonoT5UQA(L.LightningModule):
    def __init__(self, model, tokenizer, fold, batch_size, learning_rate=1e-5):
        super().__init__()        

        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        
        prediction_tokens = ['▁false', '▁true']
        self.token_false_id = self.tokenizer.get_vocab()[prediction_tokens[0]]
        self.token_true_id  = self.tokenizer.get_vocab()[prediction_tokens[1]]
        self.FIRST_RANK = 0
        
        qa_checkpoint = "allenai/unifiedqa-v2-t5-3b-1363200" # you can specify the model size here
        self.qa_tokenizer = T5Tokenizer.from_pretrained(qa_checkpoint)
        self.qa_model = T5ForConditionalGeneration.from_pretrained(qa_checkpoint).to("cuda") 

        self.fold = fold
        self.batch_size = batch_size
    
        self.top_k = 10
        self.REL = tokenizer.encode('true')[0]
        self.NREL = tokenizer.encode('false')[0]
        self.FIRST_RANK = 0

        scalar_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score(),
                "qa_sas": SemanticAnswerSimilarity(),
                "qa_em": ExactMatch(),
                #"qa_meteor": METEOR(),
                "qa_bert_score": BERTScore()
            }
        )
        
        list_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score('list'),
                "qa_sas": SemanticAnswerSimilarity('list'),
                "qa_em": ExactMatch('list'),
                #"qa_meteor": METEOR('list'),
                "qa_bert_score": BERTScore('list')
            }
        )
        
        self.val_qa_metrics = scalar_qa_metrics.clone(prefix='val_')
        self.test_qa_metrics = list_qa_metrics.clone(prefix='test_')

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_input_ids=decoder_input_ids)
        
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training

    def log_results(self, metrics, metrics_type, eval_method='test'):
        # Move metrics to the CPU and convert to a format that can be written to a CSV
        if not os.path.isfile(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv'):
            metrics_columns = [list(metrics.keys())]
            with open(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_columns)

        metrics_data=[]   
        for i in range(len(list(metrics.values())[0])):
            row = [tensor[i].item() for tensor in metrics.values()]
            metrics_data.append(row)

        # Write CSV data to the logger
        with open(self.logger.experiment.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)
    
    def log_qa_predictions(self, qa_preds):
        # Log predictions
        csv_data = qa_preds.values.tolist()
        
        # If file does not exist, write column names as the first row
        if not os.path.isfile(self.logger.experiment.log_dir + '/test_qa_predictions.csv'):
            column_names = list(qa_preds.columns)
            with open(self.logger.experiment.log_dir + '/test_qa_predictions.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(column_names)

        # Write CSV data to the logger
        with open(self.logger.experiment.log_dir + '/test_qa_predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    
    def qa(self, row):
        q, d, rank = row['query'], row['text'], row['rank']
        if rank >= self.top_k:
            return ' '
        enc = self.tokenizer.encode_plus(f'{q} \\n {d}', return_tensors='pt',  padding='longest').to("cuda")

        input_ids  = enc['input_ids'].to("cuda")
        set_seed(42)
        outputs = self.qa_model.generate(
            input_ids=input_ids).to("cuda")
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return res
    
    
    def get_qa_results(self, run):
        scores = []
        queries, texts = run['query'], run['text']
        max_input_length = 512
        it = range(0, len(queries), self.batch_size)
        #it = pt.tqdm(it, desc='monoQA', unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            input_ids = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d} Relevant: ' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest').to("cuda").input_ids
  
            with torch.no_grad():
                set_seed(42)
                outputs = self.model.generate(
                    input_ids,
                    return_dict_in_generate=True, 
                    output_scores=True 
                )
            result = outputs.scores[0][:, [self.token_false_id, self.token_true_id]]
            result = torch.nn.functional.softmax(result, dim=1)
            scores += result[:, 1].cpu().detach().tolist()

        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        #run = add_ranks(run)
        run["rank"] = run.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + self.FIRST_RANK
        run = run.sort_values(by=['qid', 'rank'], ascending=True)
        run = run[run['rank'] == 0]
        run['answer'] = run.apply(self.qa, axis=1)
        #run['gold_answer'] = run.apply(self.generate_target_answer, axis=1)
        return run        
    
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)        
        self.log("val_loss", outputs["loss"], prog_bar=True)
        """outputs = self.model.generate(
                batch["input_ids"],
                return_dict_in_generate=True, 
                output_scores=True
            )

        scores = outputs.scores[0][:, [self.token_false_id, self.token_true_id]]
        scores = torch.nn.functional.softmax(scores, dim=1)
        probabilities = scores[:, 1].tolist()

        for passage, probability in zip(passages, probabilities):
            passage['score'] = probability

        ranked_passages = sorted(passages, key=lambda x: x['score'], reverse=True)
        
        
        self.val_qa_metrics(decoded_preds, decoded_labels)
        self.log_dict(self.val_qa_metrics, prog_bar=True)"""

        
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)        
        self.log("test_loss", outputs["loss"], prog_bar=True)
        """generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]
        
        predictions = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.test_avg_qa_metrics(decoded_preds, decoded_labels)
        self.log_dict(self.test_avg_qa_metrics, prog_bar=True)
        
        #TODO: add preds and labels to dataframe -> calc metrics and log when test epoch ends
        # Create a temporary DataFrame with the new data
        new_data = pd.DataFrame({"predictions": decoded_preds, "labels": decoded_labels})        
        # Append the new data to the original DataFrame
        self.test_results = pd.concat([self.test_results, new_data], ignore_index=True)"""


    def on_validation_epoch_end(self):
        dev_split = self.fold["dev"].rename(columns={'global_turn_id': 'qid'})
        qa_preds = self.get_qa_results(dev_split)
        self.val_qa_metrics(qa_preds['answer'].tolist(), qa_preds['gold_answer'].tolist())
        self.log_dict(self.val_qa_metrics, prog_bar=True)
        
    def on_test_epoch_end(self):
        test_split = self.fold["test"].rename(columns={'global_turn_id': 'qid'})
        qa_preds = self.get_qa_results(test_split)
        
        qa_metrics_results = self.test_qa_metrics(qa_preds['answer'].tolist(), qa_preds['gold_answer'].tolist())
        if not self.trainer.sanity_checking:
            self.log_results(qa_metrics_results, "qa_metrics")
            self.log_qa_predictions(qa_preds)
        
        self.test_qa_metrics.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    

class MonoQookA(L.LightningModule):
    def __init__(self, model, tokenizer, fold, batch_size, learning_rate=1e-5):
        super().__init__()        

        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

        self.fold = fold
        self.batch_size = batch_size
    
        self.top_k = 10
        self.REL = tokenizer.encode('true')[0]
        self.NREL = tokenizer.encode('false')[0]
        self.FIRST_RANK = 0

        scalar_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score(),
                "qa_sas": SemanticAnswerSimilarity(),
                "qa_em": ExactMatch(),
                #"qa_meteor": METEOR(),
                "qa_bert_score": BERTScore()
            }
        )
        
        list_qa_metrics = MetricCollection(
            {
                "qa_answer_f1": F1Score('list'),
                "qa_sas": SemanticAnswerSimilarity('list'),
                "qa_em": ExactMatch('list'),
                #"qa_meteor": METEOR('list'),
                "qa_bert_score": BERTScore('list')
            }
        )
        
        self.val_qa_metrics = scalar_qa_metrics.clone(prefix='val_')
        self.test_qa_metrics = list_qa_metrics.clone(prefix='test_')


    @property
    def get_model(self):
        return self._model
    
    @property
    def get_tokenizer(self):
        return self._tokenizer
    
    def __split_tasks(self, text):
        if text.startswith('true ') or text.startswith('false '):
            return text.split()[0], ' '.join(text.split()[1:])
        else:
            return 'false', text
    
    def qa(self, row):
        q, d, rank = row['query'], row['text'], row['rank']
        if rank >= self.top_k:
            return ' '
        enc = self.tokenizer.encode_plus(f'Question Answering: {q} <extra_id_0> {d}', return_tensors='pt',  padding='longest')

        input_ids  = enc['input_ids'].to("cuda")
        set_seed(42)
        beam_outputs = self.model.generate(
            input_ids=input_ids,# attention_mask=attention_masks,
            do_sample=True,
            max_length=128,
        #             top_k=120,
            top_k=60,
            top_p=0.98,
            early_stopping=True,
            num_beams=4,
            num_return_sequences=1
        ).to("cuda")
        res = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)[0]
        return ' '.join(res.split()[1:])
    
    def generate_target_answer(self, sample):
        if pd.isna(sample['actual_answer']):
            return f"{sample['expected_answer'].lower()}."
        else:
            return f"{sample['actual_answer'].lower()}."
    
    def get_qa_results(self, run):
        scores = []
        queries, texts = run['query'], run['text']
        max_input_length = 512
        it = range(0, len(queries), self.batch_size)
        #it = pt.tqdm(it, desc='monoQA', unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Question Answering: {q} <extra_id_0> {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')

            input_ids  = enc['input_ids'].to("cuda")
            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                set_seed(42)
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        #run = add_ranks(run)
        run["rank"] = run.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + self.FIRST_RANK
        run['answer'] = run.apply(self.qa, axis=1)
        run = run.sort_values(by=['qid', 'rank'], ascending=True)
        run = run[run['rank'] == 0]
        #run['gold_answer'] = run.apply(self.generate_target_answer, axis=1)
        return run
    
    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_input_ids=decoder_input_ids)
        
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training
       
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)        
        self.log("val_loss", outputs["loss"], prog_bar=True)
        generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )

        labels = batch["labels"]
        
        predictions = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [self.__split_tasks(pred) for pred in decoded_preds]
        decoded_labels = [self.__split_tasks(label) for label in decoded_labels]

        rel_preds = [m[0] for m in decoded_preds]
        rel_labels = [m[0] for m in decoded_labels]

        ans_preds = [m[1] for m in decoded_preds]
        ans_labels = [m[1] for m in decoded_labels]

        
    def log_results(self, metrics, metrics_type, eval_method='test'):
        # Move metrics to the CPU and convert to a format that can be written to a CSV
        if not os.path.isfile(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv'):
            metrics_columns = [list(metrics.keys())]
            with open(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_columns)

        metrics_data=[]   
        for i in range(len(list(metrics.values())[0])):
            row = [tensor[i].item() for tensor in metrics.values()]
            metrics_data.append(row)

        # Write CSV data to the logger
        with open(self.logger.log_dir + f'/{eval_method}_{metrics_type}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_data)
    
    def log_qa_predictions(self, qa_preds):
        # Log predictions
        csv_data = qa_preds.values.tolist()
        
        # If file does not exist, write column names as the first row
        if not os.path.isfile(self.logger.log_dir + '/test_qa_predictions.csv'):
            column_names = list(qa_preds.columns)
            with open(self.logger.log_dir + '/test_qa_predictions.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(column_names)

        # Write CSV data to the logger
        with open(self.logger.log_dir + '/test_qa_predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    
    def on_validation_epoch_end(self):
        dev_split = self.fold["dev"].rename(columns={'global_turn_id': 'qid'})
        qa_preds = self.get_qa_results(dev_split)
        self.val_qa_metrics(qa_preds['answer'].tolist(), qa_preds['gold_answer'].tolist())
        self.log_dict(self.val_qa_metrics, prog_bar=True)
        
    def on_test_epoch_end(self):
        test_split = self.fold["test"].rename(columns={'global_turn_id': 'qid'})
        qa_preds = self.get_qa_results(test_split)
        
        qa_metrics_results = self.test_qa_metrics(qa_preds['answer'].tolist(), qa_preds['gold_answer'].tolist())
        if not self.trainer.sanity_checking:
            self.log_results(qa_metrics_results, "qa_metrics")
            self.log_qa_predictions(qa_preds)
        
        self.test_qa_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)        
        self.log("test_loss", outputs["loss"], prog_bar=True)
        generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]
        
        predictions = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [self.__split_tasks(pred) for pred in decoded_preds]
        decoded_labels = [self.__split_tasks(label) for label in decoded_labels]

        rel_preds = [m[0] for m in decoded_preds]
        rel_labels = [m[0] for m in decoded_labels]

        ans_preds = [m[1] for m in decoded_preds]
        ans_labels = [m[1] for m in decoded_labels]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
