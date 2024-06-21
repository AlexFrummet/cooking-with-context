#!/bin/bash

# CANARD Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation canard_rewritten --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation canard_rewritten --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation canard_rewritten --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/step/canard_rewritten/ -type d -name "checkpoints" -exec rm -r {} +

# QuReTeC Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation quretec_expansion --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation quretec_expansion --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation quretec_expansion --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/step/quretec_expansion/ -type d -name "checkpoints" -exec rm -r {} +

# Prepend Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation prepend_context --context_window n_0 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation prepend_context --context_window n_2 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation prepend_context --context_window n_3 --need_type fact_needs
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation step --query_representation prepend_context --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/step/prepend_context/ -type d -name "checkpoints" -exec rm -r {} +

