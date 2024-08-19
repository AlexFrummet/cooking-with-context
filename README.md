# Cooking with Context: Leveraging Context for Procedural Question Answering
This repository contains code and data for replicating the experiments described in the paper "Cooking with Context: Leveraging Context for Procedural Question Answering".

## Where do I find the parser for document representations?
Find the parser code in the `task_tree_parser` folder.

## Where is the code to replicate the IR experiments?
The code to replicate experiments is located in the `experiment` folder. To run an experiment:

1. Set up a virtual environment with Python 3.9 and activate it.
2. Install required packages using `pip install -r requirements.txt`.
3. Execute experiments with `python run_experiment.py`

Specify options using:
```shell
python run_experiment.py \
        --base_model ['t5', 'bart', 'monoqa', 'fid', 'monot5uqa'] \
        --document_representation ['document', 'step', 'node'] \
        --dataset ['qooka', 'cook_dial'] \
        --query_representation ['prepend_context', 'canard_rewritten', 'quretec_expansion'] \
        --context_window ['n_0', 'n_2', 'n_3', 'n_all'] \
        --need_type ['fact_needs','competence_needs','all_needs']        
```

## Where are the statistical analyses?
Statistical analyses can be found in the `analyses` folder.

## Citation
t.b.d.
