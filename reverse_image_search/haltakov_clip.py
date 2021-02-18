# %% Import
import clip
import torch
import streamlit as st
import os
from sys import platform
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader

# Fix Openmp bug when computing vector for uploaed image on MacOs
if platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Jinja Template settings
root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(root, "templates")
env = Environment(loader=FileSystemLoader(templates_dir))

st.set_page_config(layout="wide")


# %% Load Data
photo_ids = pd.read_csv("data/photo_ids.csv")
photo_features = np.load("../data/features32.npy")

# %% Encode Query
def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # Retrieve the feature vector from the GPU and convert it to a numpy array
    return text_encoded.cpu().numpy()


# %% KNN
def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
    knn_distances = (photo_features @ text_features.T).squeeze(1)
    knn_indices = (-knn_distances).argsort()
    result = []
    for knn_index in knn_indices[:results_count]:
        result.append(photo_ids.iloc[knn_index]["photo_id"])
    return result


# %% Render Imagegrid from Template
def render_from_template(template_filename, best_photo_ids, write_to_disk):
    template = env.get_template(template_filename)
    output_file = os.path.join(root, "html", template_filename)
    html_image_grid = template.render(ids=best_photo_ids)
    if write_to_disk:
        with open(output_file, "w") as fh:
            fh.write(html_image_grid)
    return html_image_grid


# %% Display
def display_image_grid(best_photo_ids):
    html_image_grid = render_from_template(
        template_filename="index.html",
        best_photo_ids=best_photo_ids,
        write_to_disk=False,
    )
    st.components.v1.html(html_image_grid, height=5000, scrolling=True)


# %% Search
def search_unsplash(search_query, photo_features, photo_ids, results_count=3):
    text_features = encode_search_query(search_query)
    best_photo_ids = find_best_matches(
        text_features, photo_features, photo_ids, results_count
    )

    display_image_grid(best_photo_ids)
    return best_photo_ids


# %% GUI
#!%%time
st.write("# Search 2 Million Unsplash Images using natural language")
search_query = st.text_input("Describe the desired Image", value="yawning cat on couch")
search_unsplash(search_query, photo_features, photo_ids, 20)
