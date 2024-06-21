import pandas as pd
import logging
logging.basicConfig(format='[%(levelname)s] (%(filename)s) %(message)s', level=logging.INFO)
import numpy as np
import ast
import os
import csv
import random
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import json
import warnings
warnings.filterwarnings("ignore")

class Preprocessor:
   
    def __init__(self, data_path, query_representation_method, document_representation, context_window, model_type, parameters):
        self.parameters = parameters
        self.folds = self.__create_5_fold_split(query_representation_method, document_representation, context_window, data_path, model_type)

    def prepare_training_dataset(self, model, tokenizer):
        self.tokenizer=tokenizer
        self.model = model
        for fold in self.folds:
            logging.info(f"Prepare {fold}")
            curr_tokenized_dataset = self.__tokenize_dataset(self.folds[fold])
            data_collator, train_dataloader, dev_dataloader, test_dataloader = self.__setup_data_loaders(curr_tokenized_dataset)
            self.folds[fold]['data_collator'] = data_collator
            self.folds[fold]['train_dataloader'] = train_dataloader
            self.folds[fold]['dev_dataloader'] = dev_dataloader
            self.folds[fold]['test_dataloader'] = test_dataloader

    
    def get_folds(self):
        return self.folds
    
        
    def __create_5_fold_split(self, query_representation_method, document_representation, context_window, data_path, model_type):
        logging.info("Import folds for cross validation ...")
        num_splits = 5
        folds = {}        
        
        if model_type == "fid":
            for i in range(num_splits):
                fold_directory = os.path.join(data_path, "experiment_datasets", document_representation, model_type, f"fold_{i+1}", context_window, query_representation_method)
                # Load train.json and test.json
                with open(os.path.join(fold_directory, "train.json"), "r") as train_file:
                    train_data = json.load(train_file)
                with open(os.path.join(fold_directory, "test.json"), "r") as test_file:
                    test_data = json.load(test_file)

                folds[f'fold_{i+1}'] = {
                    'train': pd.DataFrame(train_data),
                    'dev': pd.DataFrame(test_data),  # Dev and Test data are shared in your code
                    'test': pd.DataFrame(test_data),
                    'train_path': str(os.path.join(fold_directory, "train.json")),
                    'dev_path': str(os.path.join(fold_directory, "test.json")),
                    'test_path': str(os.path.join(fold_directory, "test.json"))
                }
        else:
            for i in range(num_splits):
                folds[f'fold_{i+1}'] = {}
                folds[f'fold_{i+1}']['train'] = pd.read_csv(f"{data_path}/experiment_datasets/{document_representation}/{model_type}/fold_{i+1}/{context_window}/{query_representation_method}/train.csv",
                                                           delimiter=";")
                # Having shared test/dev set here!!
                folds[f'fold_{i+1}']['dev'] =pd.read_csv(f"{data_path}/experiment_datasets/{document_representation}/{model_type}/fold_{i+1}/{context_window}/{query_representation_method}/test.csv",
                                                           delimiter=";")
                folds[f'fold_{i+1}']['test'] =pd.read_csv(f"{data_path}/experiment_datasets/{document_representation}/{model_type}/fold_{i+1}/{context_window}/{query_representation_method}/test.csv",
                                                           delimiter=";")
        return folds
    
    
    def __tokenize_fn(self, sample):
        model_inputs = self.tokenizer(sample['input'], text_target=sample['target_answer'],
                                 truncation=True)
        return model_inputs
    
    def __tokenize_dataset(self, fold):
        logging.info(f"Create dataset fold for fine-tuning ...")
        dataset = DatasetDict({
            "train": Dataset.from_pandas(fold['train']),
            "dev": Dataset.from_pandas(fold['dev']),
            "test": Dataset.from_pandas(fold['test'])
        })
        logging.info(f"Tokenize model inputs ...")
        dataset = dataset.map(self.__tokenize_fn, 
                      remove_columns=dataset["train"].column_names)
        dataset.set_format("torch")
        return dataset
        
    
    def __setup_data_loaders(self, dataset):
        logging.info(f"Set up data loaders ...")
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.parameters['batch_size']
        )
        
        dev_dataloader = DataLoader(
            dataset["dev"],
            collate_fn=data_collator,
            batch_size=self.parameters['batch_size'],
            shuffle=False
        )
        
        test_dataloader = DataLoader(
            dataset["test"],
            collate_fn=data_collator,
            batch_size=self.parameters['batch_size'],
            shuffle=False
        )
        return data_collator, train_dataloader, dev_dataloader, test_dataloader
      


   