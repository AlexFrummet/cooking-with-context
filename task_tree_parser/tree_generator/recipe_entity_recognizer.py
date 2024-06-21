from pprint import pprint

import pandas as pd
import spacy
import warnings
import random

warnings.filterwarnings('ignore')


class RecipeEntityRecognizer:
    INGREDIENT_LIST = "data/entity_lists/ingredients.txt"
    UNIT_LIST = "data/entity_lists/units.txt"
    EQUIPMENT_LIST = "data/entity_lists/equipment_list.txt"
    COMPETENCE_NEEDS = ['preparation', 'cooking_technique']

    def __init__(self, recipe, recipe_ingredients, condition=None, info_need=None):
        self.recipe = recipe
        self.all_ingredients = self.create_keyword_list(RecipeEntityRecognizer.INGREDIENT_LIST)
        self.all_units = self.create_keyword_list(RecipeEntityRecognizer.UNIT_LIST)
        self.all_equipments = self.create_keyword_list(RecipeEntityRecognizer.EQUIPMENT_LIST)
        self.nlp = self.init_nlp_module()
        self.condition = condition
        self.info_needs = info_need
        self.recipe_ingredients = recipe_ingredients
        self.response_tree = self.__create_response_tree()

    def create_keyword_list(self, keyword_list):
        keywords = []
        with open(keyword_list, "r", encoding="utf-8") as keyword_file:
            for line in keyword_file:
                keywords.append(line.lower().strip())
        return keywords

    def init_nlp_module(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.remove_pipe("ner")
        english_ruler = nlp.add_pipe("entity_ruler", config={"validate": True})

        ingredient_patterns = [
            {"label": "ingredient", "pattern":
                [
                    {"POS": "ADJ", "OP": "?"},
                    {"LOWER": {"REGEX": f"{str(ingredient)}"}, "POS": {"IN": ["NOUN", "PROPN", "X"]}},
                ]
             } for ingredient in list(nlp.pipe(self.all_ingredients))
        ]

        equipment_patterns = [
            {"label": "equipment", "pattern":
                [
                    {"POS": "ADJ", "OP": "*"},
                    {"POS": "NOUN", "OP": "*"},
                    {"IS_DIGIT": True, "OP": "*"},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"POS": "NOUN", "OP": "*"},
                    {"POS": "PROPN", "OP": "*"},
                    {"LOWER": f"{str(equipment)}"}
                ]
             } for equipment in list(nlp.pipe(self.all_equipments))
        ]

        time_patterns = [
            {"label": "time", "pattern":
                [
                    {"TEXT": {"REGEX": "for|about"}, "OP": "?"},
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": {"REGEX": "until|-"}, "OP": "?"},
                    {"IS_DIGIT": True, "OP": "*"},
                    {"LOWER": {"REGEX": "min[a-z]*|hour(s)?|day(s)?"}, "OP": "+"}
                ]
             },
        ]

        amount_patterns = [
            {"label": "amount", "pattern":
                [
                    {"IS_DIGIT": True, "OP": "+"},
                    {"TEXT": f"{str(unit)}"}
                ]
             } for unit in list(nlp.pipe(self.all_units))
        ]

        cooking_technique_patterns = [
            {"label": "cooking_technique", "pattern":
                [
                    {"POS": "VERB", "IS_STOP": False}
                ]
             }
        ]

        temperature_patterns = [
            {"label": "temperature", "pattern":
                [
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": {"REGEX": "until|-|,|/|to"}, "OP": "?"},
                    {"IS_DIGIT": True, "OP": "+"},
                    {"LOWER": {"REGEX": "degree"}},
                ]
             },
            {"label": "temperature", "pattern":
                [
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": {"REGEX": "until|-|,|/|to"}, "OP": "?"},
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": "°"},
                    {"LOWER": "c", "OP": "?"}
                ]
             },
            {"label": "temperature", "pattern":
                [
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": {"REGEX": "until|-|,|/|to"}, "OP": "?"},
                    {"IS_DIGIT": True, "OP": "*"},
                    {"TEXT": "°"},
                    {"LOWER": "f", "OP": "?"}
                ]
             },
            {"label": "temperature", "pattern":
                [
                    {"LOWER": {"REGEX": "low|medium|high"}, "OP": "*"},
                    {"TEXT": "-", "OP": "?"},
                    {"LOWER": {"REGEX": "low|medium|high"}, "OP": "*"},
                    {"POS": "VERB", "OP": "?"},
                    {"LOWER": "heat", "POS": "NOUN"},
                    {"LOWER": "to", "OP": "?"},
                    {"LOWER": {"REGEX": "low|medium|high"}, "OP": "*"},
                    {"TEXT": "-", "OP": "?"},
                    {"LOWER": {"REGEX": "low|medium|high"}, "OP": "*"},
                ]
             },
        ]
        nlp.add_pipe("merge_entities")

        patterns = ingredient_patterns + time_patterns + amount_patterns + equipment_patterns + temperature_patterns + cooking_technique_patterns
        english_ruler.add_patterns(patterns)
        return nlp

    def __get_need_type(self, entity_type):

        if entity_type in RecipeEntityRecognizer.COMPETENCE_NEEDS:
            return "competence"
        elif self.condition == "red" and entity_type not in RecipeEntityRecognizer.COMPETENCE_NEEDS:
            return "fact"
        elif self.condition == "blue" and entity_type not in RecipeEntityRecognizer.COMPETENCE_NEEDS:
            return "knowledge"
        elif self.condition is None:
            return random.choice(["fact", "knowledge"])

    def __clean_need_type_distribution(self, nodes):
        num_fact_nodes = sum([1 for node in nodes if node['need_type'] == "fact"])
        num_knowledge_nodes = sum([1 for node in nodes if node['need_type'] == "knowledge"])
        cleaned_nodes = nodes.copy()
        for node in cleaned_nodes:
            if node['need_type'] == 'fact' and num_knowledge_nodes == 0 and num_fact_nodes > 1:
                node['need_type'] = 'knowledge'
                break
            elif node['need_type'] == 'fact' and num_knowledge_nodes == 0 and num_fact_nodes == 1:
                new_node = node.copy()
                new_node['need_type'] = 'knowledge'
                cleaned_nodes.append(new_node)
                break
            if node['need_type'] == 'knowledge' and num_fact_nodes == 0 and num_knowledge_nodes > 1:
                node['need_type'] = 'fact'
                break
            elif node['need_type'] == 'knowledge' and num_fact_nodes == 0 and num_knowledge_nodes == 1:
                new_node = node.copy()
                new_node['need_type'] = 'fact'
                cleaned_nodes.append(new_node)
                break

        return cleaned_nodes

    def __is_entity_in_ingredient_list(self, official_ingredient, detected_ingredient):
        comma_free_ingredient = str.replace(official_ingredient, ",", "")
        split_ingredient = comma_free_ingredient.split(" ")
        for word in split_ingredient:
            if word.startswith(detected_ingredient) or word.endswith(detected_ingredient):
                return True
        return False

    def __get_item_amount(self, entity_type, entity_text):
        entity_text_doc = self.nlp(entity_text)
        normalized_entity_text = entity_text_doc[0].lemma_
        if entity_type == "ingredient":
            for ingredient in self.recipe_ingredients:
                if self.__is_entity_in_ingredient_list(ingredient['ingredient'].lower(), entity_text.lower()):
                    return ingredient
                if self.__is_entity_in_ingredient_list(ingredient['ingredient'].lower(),
                                                       normalized_entity_text.lower()):
                    return ingredient
        return {"amount": None, "unit": None, "ingredient": None}

    def __create_node(self, entity, entity_type, sentence):
        return {
            "entity_type": entity_type,
            "amount": self.__get_item_amount(entity_type, str(entity.text))["amount"],
            "unit": self.__get_item_amount(entity_type, str(entity.text))["unit"],
            "official_ingredient": self.__get_item_amount(entity_type, str(entity.text))["ingredient"],
            "need_type": self.__get_need_type(entity_type),
            "entity_text": str(entity.text),
            "entity_start": int(entity.start_char),
            "entity_end": int(entity.end_char),
            "sentence": sentence
        }

    def __get_nodes(self, step):
        nodes = []
        if self.condition == "bold":
            for sent in step.sents:
                nodes.append(self.__create_node(sent, "preparation", str(sent.text)))
        else:
            for entity in step.ents:
                # why this??
                if self.info_needs is None or str(entity.label_) in self.info_needs:
                    curr_node = self.__create_node(entity, str(entity.label_), str(entity.sent))
                    if self.condition == "red" and curr_node["need_type"] == "fact":
                        nodes.append(curr_node)
                    elif self.condition == "green" and curr_node["need_type"] == "competence":
                        nodes.append(curr_node)
                    elif self.condition == "blue" and curr_node["need_type"] == "knowledge":
                        nodes.append(curr_node)
                    elif self.condition is None:
                        nodes.append(curr_node)
        if self.condition is None:
            nodes = self.__clean_need_type_distribution(nodes)
        return nodes

    def __format_ingredient_list(self, ingredient_list, sep_token):
        formatted_strings = []

        for entry in ingredient_list:
            amount = entry["amount"]
            unit = entry["unit"]
            ingredient = entry["ingredient"]

            formatted_strings.append(f"{amount} {unit} {ingredient}")

        result = sep_token.join(formatted_strings)
        return result

    def __create_response_tree(self):
        recipe_response_tree = []
        ingredient_list = self.__format_ingredient_list(self.recipe["ingredients"], ". ")
        ingredient_list_processed = self.nlp(ingredient_list)
        ingredient_list_nodes = self.__get_nodes(ingredient_list_processed)
        ingredient_list_response_tree_bag = {
            "step_no": 0,
            "step_text": self.__format_ingredient_list(self.recipe["ingredients"], ", "),
            "bag_of_nodes": ingredient_list_nodes,
            "step_image": None
        }
        recipe_response_tree.append(ingredient_list_response_tree_bag)
        for step in self.recipe["steps"]:
            curr_step = self.nlp(step['step_text'])
            curr_nodes = self.__get_nodes(curr_step)
            curr_response_tree_bag = {
                "step_no": step['step_no'],
                "step_text": step['step_text'],
                "bag_of_nodes": curr_nodes,
                "step_image": step['step_image'] if 'step_image' in step else None
            }
            recipe_response_tree.append(curr_response_tree_bag)
        return recipe_response_tree

    def __get_distributed_random_node_selection(self, bag_of_nodes, num_step_nodes):
        # group ingredients -- group all others
        # select one ingredient, select two of others
        # if selection < 3 -> select ingredient
        node_selection = []
        ingredient_nodes = []
        not_ingredient_nodes = []
        for node in bag_of_nodes:
            if node['need_type'] == "ingredient":
                ingredient_nodes.append(node)
            else:
                not_ingredient_nodes.append(node)
        try:
            node_selection.append(random.sample(ingredient_nodes, 1))
            node_selection.append(random.sample(not_ingredient_nodes, 2))
        except:
            node_selection.append(random.sample(bag_of_nodes, num_step_nodes))

        return node_selection

    def __get_random_nodes(self, bag_of_nodes, num_step_nodes):
        random_bag_nodes = []

        if self.condition is None:
            if sum([1 for node in bag_of_nodes if node['need_type'] == 'fact']) >= 1:
                random_bag_nodes.append(random.choice([node for node in bag_of_nodes if node['need_type'] == 'fact']))
            if sum([1 for node in bag_of_nodes if node['need_type'] == 'knowledge']) >= 1:
                random_bag_nodes.append(
                    random.choice([node for node in bag_of_nodes if node['need_type'] == 'knowledge']))
            if sum([1 for node in bag_of_nodes if node['entity_type'] == 'preparation']) >= 1:
                random_bag_nodes.append(
                    random.choice([node for node in bag_of_nodes if node['entity_type'] == 'preparation']))
            if sum([1 for node in bag_of_nodes if node['entity_type'] == 'cooking_technique']) >= 1:
                random_bag_nodes.append(
                    random.choice([node for node in bag_of_nodes if node['entity_type'] == 'cooking_technique']))
        else:
            # for each step one ingredient and two others
            # random_bag_nodes = random.sample(bag_of_nodes, num_step_nodes)
            random_bag_nodes = self.__get_distributed_random_node_selection(bag_of_nodes, num_step_nodes)

        return random_bag_nodes

    def get_random_response_tree(self, num_step_nodes=100):
        random_selection = []
        for step in self.response_tree:
            curr_step_bag_of_nodes = step['bag_of_nodes']
            if len(curr_step_bag_of_nodes) >= num_step_nodes:
                random_nodes = self.__get_random_nodes(curr_step_bag_of_nodes, num_step_nodes)
            else:
                random_nodes = curr_step_bag_of_nodes
            new_response_tree_step = {
                "step_no": step['step_no'],
                "step_text": step['step_text'],
                "bag_of_nodes": random_nodes,
                "step_image": step['step_image'] if 'step_image' in step else None
            }
            random_selection.append(new_response_tree_step)
        return random_selection


if __name__ == "__main__":
    recipes = pd.read_json("../data/whole_foods_recipe_documents_2023-06-07.jsonl", lines=True)
    pprint(recipes.iloc[0]['ingredients'])
    recipe_response_tree = RecipeEntityRecognizer(recipes.iloc[0], recipes.iloc[0]['ingredients'], 'red',
                                                  ['amount', 'time', 'temperature', 'equipment', 'ingredient'])
    pprint(recipe_response_tree.get_random_response_tree())
