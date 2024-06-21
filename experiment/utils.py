import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path

def get_config(path):
    with open(path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the argument for the context condition
    parser.add_argument('-base',
                        '--base_model',
                        type=str,
                        choices=['t5', 'bart', 'monoqa', 'fid', 'monot5uqa'],
                        required=False,
                        help='Specify the base model to use for training')
    parser.add_argument('-doc_rep',
                        '--document_representation',
                        type=str,
                        choices=['document', 'step', 'node'],
                        required=True,
                        help='Specify the context condition')
    parser.add_argument('-data',
                        '--dataset',
                        type=str,
                        choices=['qooka', 'cook_dial', 'wizard_of_tasks'],
                        required=True,
                        help='Specify the dataset to use for the experiment')
    parser.add_argument('-query_rep',
                        '--query_representation',
                        type=str,
                        choices=['prepend_context', 'canard_rewritten', 'quretec_expansion', 'structured_rep'],
                        required=False,
                        help='Specify the query representation method')
    parser.add_argument('-context',
                        '--context_window',
                        type=str,
                        choices=['n_0', 'n_2', 'n_3', 'n_all'],
                        required=False,
                        help='Specify the number of previous turns to involve')
    parser.add_argument('-need',
                        '--need_type',
                        type=str,
                        choices=['fact_needs','competence_needs','all_needs'],
                        required=True,
                        help='Specify the need type you want to train your model on')
    parser.add_argument('-step_rep',
                        '--step_representation',
                        default='original',
                        type=str,
                        choices=['original','merged'],
                        required=False,
                        help='Specify the version of the step representation')
    return parser.parse_args()

def get_base_models(arg_base_model, config):
    if arg_base_model:
        base_model = [model for model in config['base_models'] if model['model_type'] == arg_base_model]
        return base_model
    else:
        return config['base_models'] 
    
def get_query_representation(arg_query_representation, config):
    query_representation = [arg_query_representation] if arg_query_representation else config['query_representation']
    return query_representation

def get_context_window(arg_context_window, config):
    context_window = [arg_context_window] if arg_context_window else config['context_window']
    return context_window

#def get_need_type(arg_need_type, config):
