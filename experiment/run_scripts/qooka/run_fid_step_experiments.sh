#!/bin/bash


## Run all_needs step experiments
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_0 --need_type all_needs
rm -r checkpoint/all_needs/qooka/step/prepend_context/n_0/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_2 --need_type all_needs
rm -r checkpoint/all_needs/qooka/step/prepend_context/n_2/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_3 --need_type all_needs
rm -r checkpoint/all_needs/qooka/step/prepend_context/n_3/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_all --need_type all_needs
rm -r checkpoint/all_needs/qooka/step/prepend_context/n_all/

## Run competence_needs step experiments
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_0 --need_type competence_needs
rm -r checkpoint/competence_needs/qooka/step/prepend_context/n_0/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_2 --need_type competence_needs
rm -r checkpoint/competence_needs/qooka/step/prepend_context/n_2/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_3 --need_type competence_needs
rm -r checkpoint/competence_needs/qooka/step/prepend_context/n_3/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_all --need_type competence_needs
rm -r checkpoint/competence_needs/qooka/step/prepend_context/n_all/

## Run fact_needs step experiments
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_0 --need_type fact_needs
rm -r checkpoint/fact_needs/qooka/step/prepend_context/n_0/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_2 --need_type fact_needs
rm -r checkpoint/fact_needs/qooka/step/prepend_context/n_2/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_3 --need_type fact_needs
rm -r checkpoint/fact_needs/qooka/step/prepend_context/n_3/
python run_experiment.py --dataset qooka --base_model fid --document_representation step --query_representation prepend_context --context_window n_all --need_type fact_needs
rm -r checkpoint/fact_needs/qooka/step/prepend_context/n_all/
