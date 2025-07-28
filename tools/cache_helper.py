import json
import os
import re
import pickle

from PIL import Image

from tools.constants import CACHE_DIR


def name_to_filename(name, type="json"):
    # Remove characters other than alphanumeric and replace spaces with _
    filename = re.sub(r'[^a-zA-Z0-9]', '_', name.lower()).replace(" ", "_")
    if type == "json":
        return f"{filename}.json"
    elif type == "text":
        return f"{filename}.txt"
    elif type == "png":
        return f"{filename}.png"
    elif type == "pkl":
        return f"{filename}.pkl"


def get_cache_content(parent_folder, name, type="json"):
    filename = name_to_filename(name, type)
    file_path = f"{CACHE_DIR}/{parent_folder}/{filename}"

    if not os.path.exists(file_path):
        return None

    if type == "png":
        return Image.open(file_path)
    elif type == "pkl":
        return pickle.load(open(file_path, "rb"))

    with open(file_path, "r") as f:
        if type == "json":
            return json.load(f)
        elif type == "text":
            return f.read()

    return None


def save_cache_content(parent_folder, name, content, type="json"):
    filename = name_to_filename(name, type)
    file_path = f"{CACHE_DIR}/{parent_folder}/{filename}"

    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if type == "png":
        return content.save(file_path)
    elif type == "pkl":
        return pickle.dump(content, open(file_path, "wb"))

    with open(file_path, "w") as f:
        if type == "json":
            json.dump(content, f)
        elif type == "text":
            f.write(content)
