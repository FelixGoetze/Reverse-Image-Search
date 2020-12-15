# %% Import
import streamlit as st
from PIL import Image
import numpy as np
import json
from faiss.contrib.exhaustive_search import knn
import glob
from pygit2 import Repository


# %load_ext autotime # measure time for each cell

st.set_page_config(page_title="Reverse Image Search", initial_sidebar_state="collapsed")

# %% Read data from disk
st.write("# Reverse image search demo on [" + Repository(".").head.shorthand + "]")
with open("data/pictures.json", "r") as filehandle:
    imagelist = json.load(filehandle)
vectors = np.load("data/vectors.npy")
# %% Select Image to Show
image_id = st.selectbox("Select an image", imagelist)
image = "https://source.unsplash.com/" + image_id
uploaded_file = st.file_uploader("Or Upload an image", type=["jpg", "jpeg"])
caption = "[View on Unsplash](https://unsplash.com/photos/" + image_id + ")"
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    caption = "Your Image"
"## Your Image :rocket:"
st.image(image, use_column_width=True)
st.write(caption)
image_index = imagelist.index(image_id)

# %% Show similar images using knn
if st.button("Find similar Images!"):
    knn_distances, knn_indices = knn(np.array([vectors[image_index, :]]), vectors, 5)
    "## Similar Images we found :fire:"
    for knn_index in knn_indices[0, 1:]:
        st.image(
            "https://source.unsplash.com/" + imagelist[knn_index], use_column_width=True
        )
        st.write(
            "[View on Unsplash](https://unsplash.com/photos/"
            + imagelist[knn_index]
            + ")"
        )
# %%
