from experiment import Experiment
import utils
import yaml
import logging
logging.basicConfig(format='[%(levelname)s] (%(filename)s) %(message)s', level=logging.INFO)

if __name__ == "__main__":
    args = utils.parse_arguments()
    # Access the parsed argument value or fallback to YAML file specs
    document_representation = args.document_representation
    
    if document_representation == "document":
        config = utils.get_config("config/baseline.yml")
    elif document_representation == "step":
        config = utils.get_config("config/step_condition.yml")
    elif document_representation == "node":
        config = utils.get_config("config/node_condition.yml")

    base_models = utils.get_base_models(args.base_model, config)
    query_representations = utils.get_query_representation(args.query_representation, config)
    context_windows = utils.get_context_window(args.context_window, config)
    need_type = args.need_type
    dataset = args.dataset
    logging.info(f"Base Model: {base_models}")
    logging.info(f"Query Representation: {query_representations}")
    logging.info(f"Context Window: {context_windows}")
    logging.info(f"Data Type: ")
    for base_model in base_models:
        for query_representation_method in query_representations:
            for context_window in context_windows:
                current_experiment = Experiment(dataset,
                                                document_representation,
                                                base_model,
                                                query_representation_method,
                                                context_window,
                                                config['data_path'],
                                                need_type)
                current_experiment.run_experiment()
        