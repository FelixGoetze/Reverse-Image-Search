# %% Import
import clip
import torch
import streamlit as st
import os
from sys import platform
import pandas as pd
import numpy as np
from IPython.display import Image

# Fix Openmp bug when computing vector for uploaed image on MacOs
if platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

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


# %% Display
def display_photo(photo_id):
    imageurl = "https://source.unsplash.com/" + photo_id
    # Image(imageurl)
    st.image(imageurl, use_column_width=True)
    st.write("[View on Unsplash](https://unsplash.com/photos/" + photo_id + ")")


# %% Search
def search_unsplash(search_query, photo_features, photo_ids, results_count=3):
    text_features = encode_search_query(search_query)
    best_photo_ids = find_best_matches(
        text_features, photo_features, photo_ids, results_count
    )
    for photo_id in best_photo_ids:
        display_photo(photo_id)
    return best_photo_ids


# %% GUI
st.write("# Search 2 Million Unsplash Images using natural language")
search_query = st.text_input("Describe the desired Image", value="yawning cat on couch")
search_unsplash(search_query, photo_features, photo_ids, 3)
# %%
