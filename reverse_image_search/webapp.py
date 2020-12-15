# %% Import
import streamlit as st
from PIL import Image
import numpy as np
import json
from faiss.contrib.exhaustive_search import knn

# %load_ext autotime # measure time for each cell

st.set_page_config(page_title="Reverse Image Search", initial_sidebar_state="collapsed")

# %% Read data from disk
"# Reverse image search demo"
with open("data/pictures.json", "r") as filehandle:
    imagelist = json.load(filehandle)
vectors = np.load("data/vectors.npy")
# %% Select Image to Show
image = st.selectbox("Select an image", imagelist)
uploaded_file = st.file_uploader("Or Upload an image", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
"## Your Image :rocket:"
st.image(image, use_column_width=True)
image_index = imagelist.index(image)

# %% Show similar images using knn
if st.button("Find similar Images!"):
    knn_distances, knn_indices = knn(np.array([vectors[image_index, :]]), vectors, 5)
    "## Similar Images we found :fire:"
    for knn_index in knn_indices[0, 1:]:
        st.image(imagelist[knn_index], use_column_width=True)
# %%
