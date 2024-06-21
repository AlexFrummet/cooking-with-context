#!/bin/bash

## Run all_needs node experiments
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_0 --need_type all_needs
rm -r checkpoint/all_needs/cook_dial/node/prepend_context/n_0/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_2 --need_type all_needs
rm -r checkpoint/all_needs/cook_dial/node/prepend_context/n_2/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_3 --need_type all_needs
rm -r checkpoint/all_needs/cook_dial/node/prepend_context/n_3/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_all --need_type all_needs
rm -r checkpoint/all_needs/cook_dial/node/prepend_context/n_all/

## Run competence_needs node experiments
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_0 --need_type competence_needs
rm -r checkpoint/competence_needs/cook_dial/node/prepend_context/n_0/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_2 --need_type competence_needs
rm -r checkpoint/competence_needs/cook_dial/node/prepend_context/n_2/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_3 --need_type competence_needs
rm -r checkpoint/competence_needs/cook_dial/node/prepend_context/n_3/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_all --need_type competence_needs
rm -r checkpoint/competence_needs/cook_dial/node/prepend_context/n_all/

## Run fact_needs node experiments
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_0 --need_type fact_needs
rm -r checkpoint/fact_needs/cook_dial/node/prepend_context/n_0/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_2 --need_type fact_needs
rm -r checkpoint/fact_needs/cook_dial/node/prepend_context/n_2/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_3 --need_type fact_needs
rm -r checkpoint/fact_needs/cook_dial/node/prepend_context/n_3/
python run_experiment.py --dataset cook_dial --base_model fid --document_representation node --query_representation prepend_context --context_window n_all --need_type fact_needs
rm -r checkpoint/fact_needs/cook_dial/node/prepend_context/n_all/
