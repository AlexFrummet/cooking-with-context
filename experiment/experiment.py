from models import ChoiModel, MonoQookA, FiD, MonoT5UQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration, BartTokenizer
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from preprocess import Preprocessor
import torch

class Experiment:   

    def __init__(self,
                 dataset,
                 document_representation,
                 base_model,
                 query_representation_method,
                 context_window,
                 data_path,
                 need_type):
        seed_everything(42, workers=True)
        self.dataset = dataset
        self.need_type = need_type
        self.document_representation = document_representation
        self.query_representation_method= query_representation_method
        self.context_window = context_window
        self.model_type, self.parameters =  base_model['model_type'], base_model['parameters']

        self.tokenizer, self.model = self.__init_base_model(self.parameters['checkpoint'])
        self.preprocessor = Preprocessor(f"{data_path}/{need_type}/{dataset}",
                                    query_representation_method,
                                    document_representation,
                                    context_window,
                                    self.model_type,
                                    parameters = self.parameters)
        if self.model_type != "fid":
            self.preprocessor.prepare_training_dataset(self.model, self.tokenizer)
        
        self.folds = self.preprocessor.get_folds()
        

    def __init_base_model(self, checkpoint):
        if "bart" in checkpoint.lower():
            tokenizer = BartTokenizer.from_pretrained(checkpoint)
            model = BartForConditionalGeneration.from_pretrained(checkpoint)
        elif checkpoint == "fid":
            tokenizer, model = None, None
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        return tokenizer, model
        
    def __init_training_model(self, model, tokenizer, fold):
        if self.model_type == "fid":
            return FiD(self.parameters, fold)
        elif self.model_type == "monot5uqa":
            return MonoT5UQA(model, tokenizer, fold, self.parameters['batch_size'], float(self.parameters['learning_rate']))
        elif self.document_representation == "document":
            return ChoiModel(model, tokenizer, fold, float(self.parameters['learning_rate']))
        elif self.document_representation in ["step", "node"] and self.model_type.startswith("monoqa"):
            return MonoQookA(model, tokenizer, fold, self.parameters['batch_size'], float(self.parameters['learning_rate']))
        else:
            return None
                
    def __init_trainer(self, logger):
        callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor=self.parameters['optimizing_metric'])]
        return L.Trainer(
            max_epochs=self.parameters['num_epochs'],
            callbacks=callbacks,
            accelerator="gpu",
            devices=[0],
            logger=logger,
            log_every_n_steps=10,
        )

    def __init_logger(self, fold):
        log_dir = f"logs/{self.need_type}/{self.dataset}/{self.document_representation}/{self.query_representation_method}/{self.context_window}/{self.model_type}/"
        return CSVLogger(save_dir=log_dir, name=f"{fold}")

    def run_experiment(self):
        for fold, fold_data in self.folds.items():
            tokenizer, model = self.__init_base_model(self.parameters['checkpoint'])
            self.training_model = self.__init_training_model(model, tokenizer, fold_data)
            logger = self.__init_logger(fold)

            if self.model_type == "fid":
                self.training_model.set_logger(logger)
                self.training_model.set_checkpoint_path(f"{self.need_type}/{self.dataset}/{self.document_representation}/{self.query_representation_method}/{self.context_window}/{self.model_type}/{fold}")

                self.training_model.train()
                self.training_model.test()
                self.training_model.evaluate_results()
            else:
                trainer = self.__init_trainer(logger)
                self.train_and_test_fold(trainer, fold_data)

    def run_inference(self):
        for fold, fold_data in self.folds.items():
            tokenizer, model = self.__init_base_model(self.parameters['checkpoint'])
            self.training_model = self.__init_training_model(model, tokenizer, fold_data)
            trainer = self.__init_trainer()

            if self.model_type == "fid":
                self.training_model.test()
                self.training_model.evaluate_results()
            else:
                self.train_and_test_fold(trainer, fold_data)

    def train_and_test_fold(self, trainer, fold_data):
        trainer.fit(
            model=self.training_model,
            train_dataloaders=fold_data['train_dataloader'],
            val_dataloaders=fold_data['dev_dataloader']
        )
        self.test_fold(trainer, fold_data)

    def test_fold(self, trainer, fold_data):
        trainer.test(dataloaders=fold_data['test_dataloader'], ckpt_path='last')

            
