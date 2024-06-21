import json
from datetime import datetime
import os
from typing import List

from scraper.scraper_utils import scrape_recipe
from tree_generator.recipe_entity_recognizer import RecipeEntityRecognizer
import pandas as pd


def scrape_recipes():
    with open('data/recipes.txt') as recipe_file:
        while recipe := recipe_file.readline().rstrip():
            scrape_recipe(recipe)


def generate_response_trees(path,
                            conditions: List,
                            info_needs: List = None):
    recipes = pd.read_json(path, lines=True)
    info_needs_label = "_".join(info_needs)
    if not os.path.exists(f'data/recipe_tree_docs/qooka_recipe_tree_docs_{datetime.today().strftime("%Y-%m-%d")}'):
        os.makedirs(f'data/recipe_tree_docs/qooka_recipe_tree_docs_{datetime.today().strftime("%Y-%m-%d")}')

    for condition in conditions:
        with open(
                f'data/recipe_tree_docs/qooka_recipe_tree_docs_{datetime.today().strftime("%Y-%m-%d")}/{condition}_{info_needs_label}_response_trees_{datetime.today().strftime("%Y-%m-%d")}.jsonl',
                'a',
                encoding='utf-8') as response_tree_file:
            for index, recipe in recipes.iterrows():
                recipe_entity_recognizer = RecipeEntityRecognizer(recipe, recipe['ingredients'], condition, info_needs)
                current_response_tree = recipe_entity_recognizer.get_random_response_tree()
                json_line = json.dumps({
                    "recipe_id": index,
                    "doc_id": recipe['doc_id'],
                    "recipe_title": recipe['title'],
                    "response_tree": [step for step in current_response_tree],
                    "ingredients": recipe['ingredients'],
                    #"duration": recipe['duration'],
                    "duration_minutes_prep": recipe['duration_minutes_prep'],
                    "duration_minutes_cooking": recipe['duration_minutes_cooking'],
                    "duration_minutes_total": recipe['duration_minutes_total'],
                    "makes": recipe["makes"],
                    "serves": recipe["serves"],
                    "required_materials": recipe["required_materials"],
                    "primary_image": recipe['primary_image']
                }, ensure_ascii=False)
                response_tree_file.write(str(json_line) + "\n")


if __name__ == '__main__':
    # scrape_recipes()
    generate_response_trees("data/recipe_documents_2022-06-03.jsonl",
                            ['red'],
                            ['amount', 'time', 'temperature', 'equipment', 'ingredient'])
    # print("TEST")