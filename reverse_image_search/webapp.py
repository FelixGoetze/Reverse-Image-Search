# %% Import
import streamlit as st
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
import json
from faiss.contrib.exhaustive_search import knn
import glob

# %load_ext autotime # measure time for each cell
input_path = "images/UnsplashHouseCollections"

st.set_page_config(page_title="Reverse Image Search", initial_sidebar_state="collapsed")

# %% Rebuild vectors and imagelist and write to disk
def rebuild_vectors(input_path):
    img2vec = Img2Vec(cuda=False)
    imagelist = glob.glob(input_path + "/" + "*.jpg")
    vectors = np.ones((len(imagelist), 512))
    for index, filename in enumerate(imagelist):
        img = Image.open(filename)
        vector = img2vec.get_vec(img)
        # vectors[index, :] = np.ones((1, 512)) # for dry run
        vectors[index, :] = vector
    with open("pictures.json", "w") as filehandle:
        json.dump(imagelist, filehandle)
    vectors = vectors.astype("float32")
    np.save("vectors.npy", vectors)
    return


if st.sidebar.button("Rebuild Index from folder"):
    rebuild_vectors(input_path)
# %% Read data from disk
"# Reverse image search demo"
with open("pictures.json", "r") as filehandle:
    imagelist = json.load(filehandle)
vectors = np.load("vectors.npy")
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
    st.balloons()
    knn_distances, knn_indices = knn(np.array([vectors[image_index, :]]), vectors, 5)
    "## Similar Images we found :fire:"
    for knn_index in knn_indices[0, 1:]:
        st.image(imagelist[knn_index], use_column_width=True)
# %%
