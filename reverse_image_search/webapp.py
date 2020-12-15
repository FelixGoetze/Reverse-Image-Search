# %% Import
import streamlit as st
from PIL import Image
import numpy as np
import json
from faiss.contrib.exhaustive_search import knn
from pygit2 import Repository
from img2vec_pytorch import Img2Vec
import os


# %load_ext autotime # measure time for each cell
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
st.set_page_config(page_title="Reverse Image Search", initial_sidebar_state="collapsed")

# %% Read data from disk
st.write("# Reverse image search demo on [" + Repository(".").head.shorthand + "]")
st.write("View source on [Github](https://github.com/FelixGoetze/Reverse-Image-Search)")
with open("data/pictures.json", "r") as filehandle:
    imagelist = json.load(filehandle)
vectors = np.load("data/vectors.npy")
# %% Select Image to Show
image_id = st.selectbox("Select an image", imagelist)
image = "https://source.unsplash.com/" + image_id
uploaded_file = st.file_uploader("Or Upload an image", type=["jpg", "jpeg"])
caption = "[View on Unsplash](https://unsplash.com/photos/" + image_id + ")"
image_index = imagelist.index(image_id)
image_vector = np.array([vectors[image_index, :]])
print(image_vector.shape)
if uploaded_file is not None:
    img2vec = Img2Vec(cuda=False)
    image = Image.open(uploaded_file)
    caption = "Your Image"
    image_vector = np.array([img2vec.get_vec(image)])
    print(image_vector.shape)
    st.write(image_vector)
"## Your Image :rocket:"
st.image(image, use_column_width=True)
st.write(caption)


# %% Show similar images using knn
if st.button("Find similar Images!"):
    knn_distances, knn_indices = knn(image_vector, vectors, 5)
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
