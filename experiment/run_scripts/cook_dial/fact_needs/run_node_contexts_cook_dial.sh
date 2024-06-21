#!/bin/bash

# CANARD Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_2 --need_type fact_needs
find logs/fact_needs/cook_dial/node/canard_rewritten/n_2/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_3 --need_type fact_needs
find logs/fact_needs/cook_dial/node/canard_rewritten/n_3/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/node/canard_rewritten/n_all/monoqa/ -type d -name "checkpoints" -exec rm -r {} +

# QuReTeC Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_2 --need_type fact_needs
find logs/fact_needs/cook_dial/node/quretec_expansion/n_2/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_3 --need_type fact_needs
find logs/fact_needs/cook_dial/node/quretec_expansion/n_3/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/node/quretec_expansion/n_all/monoqa/ -type d -name "checkpoints" -exec rm -r {} +

# Prepend Contexts
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_0 --need_type fact_needs
find logs/fact_needs/cook_dial/node/prepend_context/n_0/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_2 --need_type fact_needs
find logs/fact_needs/cook_dial/node/prepend_context/n_2/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_3 --need_type fact_needs
find logs/fact_needs/cook_dial/node/prepend_context/n_3/monoqa/ -type d -name "checkpoints" -exec rm -r {} +
python run_experiment.py --dataset cook_dial --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_all --need_type fact_needs
find logs/fact_needs/cook_dial/node/prepend_context/n_all/monoqa/ -type d -name "checkpoints" -exec rm -r {} +

