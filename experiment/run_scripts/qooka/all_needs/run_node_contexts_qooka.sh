#!/bin/bash

# CANARD Contexts
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_2 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_3 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation canard_rewritten --context_window n_all --need_type all_needs
find logs/all_needs/qooka/node/canard_rewritten/ -type d -name "checkpoints" -exec rm -r {} +

# QuReTeC Contexts
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_2 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_3 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation quretec_expansion --context_window n_all --need_type all_needs
find logs/all_needs/qooka/node/quretec_expansion/ -type d -name "checkpoints" -exec rm -r {} +

# Prepend Contexts
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_0 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_2 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_3 --need_type all_needs
python run_experiment.py --dataset qooka --base_model monoqa --document_representation node --query_representation prepend_context --context_window n_all --need_type all_needs
find logs/all_needs/qooka/node/prepend_context/ -type d -name "checkpoints" -exec rm -r {} +

