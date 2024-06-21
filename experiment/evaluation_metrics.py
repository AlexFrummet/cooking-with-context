from torchmetrics import Metric
import string, re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import evaluate
import torch
import numpy as np

class SemanticAnswerSimilarity(Metric):
    def __init__(self, mode='scalar'):
        super().__init__()
        self.cross_encoder_used, self.sas_model = self.__init_sas()
        self.mode = mode
        if mode == 'scalar':    
            self.add_state("sas_values", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
            self.add_state("num_examples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.add_state("sas_values", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
    
    def __init_sas(self, sas_model_name_or_path: str = "cross-encoder/stsb-roberta-large"):
        config = AutoConfig.from_pretrained(sas_model_name_or_path)
        cross_encoder_used = False
        if config.architectures is not None:
            cross_encoder_used = any([arch.endswith('ForSequenceClassification') for arch in config.architectures])
        if cross_encoder_used:
            model = CrossEncoder(sas_model_name_or_path)
        else:
            model = SentenceTransformer(sas_model_name_or_path)
        return cross_encoder_used, model
    
    # this function is taken and adapted from https://github.com/webis-de/SCAI-QReCC/blob/main/notebooks/sas.ipynb
    def __semantic_answer_similarity(self, predictions: List[List[str]],
                                   gold_labels: List[List[str]],
                                   ) -> Tuple[List[float],List[float]]:
        """
        Computes Transformer-based similarity of predicted answer to gold labels to derive a more meaningful metric than EM or F1.
        :param predictions: Predicted answers as list of multiple preds per question
        :param gold_labels: Labels as list of multiple possible answers per question
        :param sas_model_name_or_path: SentenceTransformers semantic textual similarity model, should be path or string
                                         pointing to downloadable models.
        """
        assert len(predictions) == len(gold_labels)        

        # Compute similarities
        top_1_sas = []
        top_k_sas = []
        scores = []

        # Based on Modelstring we can load either Bi-Encoders or Cross Encoders.
        # Similarity computation changes for both approaches
        if self.cross_encoder_used:
            for preds, labels in zip (predictions,gold_labels):
                tmp_scores = []
                if isinstance(labels,list):
                    for l in labels:
                        tmp_scores.append(self.sas_model.predict((preds, l), show_progress_bar=False))
                    scores.append(np.max(tmp_scores))
                else:
                    scores.append(self.sas_model.predict((preds, labels), show_progress_bar=False))
        else:
            # For Bi-encoders we can flatten predictions and labels into one list
            lengths: List[Tuple[int,int]] = []
            all_texts: List[str] = []
            for p, l in zip(predictions, gold_labels):                                  # type: ignore
                # TODO potentially exclude (near) exact matches from computations
                all_texts.extend(p)
                all_texts.extend(l)
                lengths.append((len(p), len(l)))
            # then compute embeddings
            embeddings = self.sas_model.encode(all_texts)

            # then select which embeddings will be used for similarity computations
            current_position = 0
            for i, (len_p, len_l) in enumerate(lengths):
                pred_embeddings = embeddings[current_position:current_position + len_p, :]
                current_position += len_p
                label_embeddings = embeddings[current_position:current_position + len_l, :]
                current_position += len_l
                scores = cosine_similarity(pred_embeddings, label_embeddings)
        
        if self.mode == 'scalar':
            return np.mean(scores)
        else:
            return scores
    
    def update(self, preds, targets):
        if self.mode == 'scalar':
            self.sas_values += self.__semantic_answer_similarity(preds, targets)
            self.num_examples += 1
        else:
            self.sas_values = torch.cat((self.sas_values, torch.Tensor(self.__semantic_answer_similarity(preds, targets)).to("cuda")))


    def compute(self):
        if self.mode == 'scalar':
            return self.sas_values / self.num_examples
        else:
            return self.sas_values.view(-1)
    
    
class BERTScore(Metric):
    def __init__(self, mode='scalar'):
        super().__init__()
        self.bertscore = evaluate.load('bertscore')
        self.mode = mode
        if mode == 'scalar':        
            self.add_state("bertscore_values", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
            self.add_state("num_examples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.add_state("bertscore_values", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds, targets):
        if self.mode == 'scalar':
            results = np.mean(self.bertscore.compute(predictions=preds, references=targets, model_type="distilbert-base-uncased")['f1'])
            self.bertscore_values += results
            self.num_examples += 1
        else:
            for pred, target in zip(preds, targets):
                result = self.bertscore.compute(predictions=[pred], references=[target], model_type="distilbert-base-uncased")['f1']
                self.bertscore_values = torch.cat((self.bertscore_values, torch.Tensor(result).to("cuda")))


    def compute(self):
        if self.mode == 'scalar':
            return self.bertscore_values / self.num_examples
        else:
            return self.bertscore_values.view(-1)
    

class METEOR(Metric):
    def __init__(self, mode='scalar'):
        super().__init__()
        self.meteor = evaluate.load('meteor')
        self.mode = mode
        if mode == 'scalar':
            self.add_state("meteor_values", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")  
            self.add_state("num_examples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.add_state("meteor_values", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds, targets):
        if self.mode == 'scalar':
            results = self.meteor.compute(predictions=preds, references=targets)['meteor']
            self.meteor_values += results
            self.num_examples += 1
        else:
            for pred,target in zip(preds, targets):
                result = self.meteor.compute(predictions=[pred], references=[target])['meteor']
                self.meteor_values = torch.cat((self.meteor_values, torch.Tensor([result]).to("cuda")))

    def compute(self):
        if self.mode == 'scalar':
            return self.meteor_values / self.num_examples 
        else:
            return self.meteor_values.view(-1)


class ExactMatch(Metric):
    def __init__(self, mode='scalar'):
        super().__init__()
        self.mode = mode
        if mode == 'scalar':
            self.add_state("em_values", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
            self.add_state("num_examples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")         
        else:
            self.add_state("em_values", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
   
    # taken from https://github.com/prdwb/orconvqa-release/blob/master/scorer.py
    def __normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    # taken from https://github.com/prdwb/orconvqa-release/blob/master/scorer.py
    def __exact_match_score(self, prediction, ground_truth):
        return (self.__normalize_answer(prediction) == self.__normalize_answer(ground_truth))

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            if self.mode == 'scalar':
                self.em_values += self.__exact_match_score(pred, target)
                self.num_examples += 1
            else:
                self.em_values = torch.cat((self.em_values, torch.Tensor([self.__exact_match_score(pred, target)]).to("cuda")))

    def compute(self):
        if self.mode == 'scalar':
            return self.em_values / self.num_examples            
        else:
            return self.em_values.view(-1)


class F1Score(Metric):
    def __init__(self, mode='scalar'):
        super().__init__()
        self.mode = mode
        if mode == 'scalar':
            self.add_state("f1_values", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
            self.add_state("num_examples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.add_state("f1_values", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
                   
     # taken from https://github.com/prdwb/orconvqa-release/blob/master/scorer.py
    def __normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    # taken from https://github.com/prdwb/orconvqa-release/blob/master/scorer.py
    def __f1_score(self, prediction, ground_truth):
        prediction_tokens = self.__normalize_answer(prediction).split()
        ground_truth_tokens = self.__normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    
    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            if self.mode == 'scalar':
                self.f1_values += self.__f1_score(pred, target)
                self.num_examples += 1
            else:
                self.f1_values = torch.cat((self.f1_values, torch.Tensor([self.__f1_score(pred, target)]).to("cuda")),dim=0)

    def compute(self):
        if self.mode == 'scalar':
            return self.f1_values / self.num_examples
        else:
            return self.f1_values.view(-1)