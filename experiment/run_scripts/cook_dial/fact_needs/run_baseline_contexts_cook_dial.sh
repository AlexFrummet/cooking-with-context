#!/bin/bash

# CANARD Contexts
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation canard_rewritten --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation canard_rewritten --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation canard_rewritten --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/document/canard_rewritten/ -type d -name "checkpoints" -exec rm -r {} +

# QuReTeC Contexts
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation quretec_expansion --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation quretec_expansion --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation quretec_expansion --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/document/quretec_expansion/ -type d -name "checkpoints" -exec rm -r {} +

# Prepend Contexts
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation prepend_context --context_window n_0 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation prepend_context --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation prepend_context --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model t5 --document_representation document --query_representation prepend_context --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/document/prepend_context/ -type d -name "checkpoints" -exec rm -r {} +

