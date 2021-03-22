# %% Import
import clip
from numpy.lib.twodim_base import vander
import torch
import streamlit as st
import os
from sys import platform
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import SessionState


# Fix Openmp bug when computing vector for uploaed image on MacOs
if platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Jinja Template settings
root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(root, "templates")
env = Environment(loader=FileSystemLoader(templates_dir))

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# %% Load Data
@st.cache()
def load_data():
    photo_ids = pd.read_csv("data/photo_ids.csv")
    photo_features = np.load("../data/features32.npy")
    return photo_ids, photo_features


# %%
photo_ids, photo_features = load_data()

# %% Encode Query
def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # Retrieve the feature vector from the GPU and convert it to a numpy array
    return text_encoded.cpu().numpy()


def encode_image_query(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        image_encoded = model.encode_image(image)
        image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
    # Retrieve the feature vector from the GPU and convert it to a numpy array
    return image_encoded.cpu().numpy()


# %% KNN
def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
    knn_distances = (photo_features @ text_features.T).squeeze(1)

    idx = np.argpartition(knn_distances, -results_count)[-results_count:]
    knn_indices = idx[np.argsort(knn_distances[idx])][::-1]

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
        write_to_disk=True,
    )
    # TODO make height adapt to screen size and amoount of images
    st.components.v1.html(html_image_grid, height=10000, scrolling=True)


# %% Search
def search_unsplash(search_query, photo_features, photo_ids, results_count=3):
    text_features = encode_search_query(search_query)
    best_photo_ids = find_best_matches(
        text_features, photo_features, photo_ids, results_count
    )

    display_image_grid(best_photo_ids)
    return best_photo_ids


# %% GUI natural language search
st.write("# Search 2 million Unsplash images")


default_inputs = {
    "text_input": "<write a query>",
    "file_uploader": None,
    "selectbox": "<select example>",
    "slider": 0,
}
# Get state of the inputs from their defaults
state = SessionState.get(inputs=default_inputs)

inputs = {}
inputs["text_input"] = st.text_input(
    "Use natural language", default_inputs["text_input"]
)
inputs["file_uploader"] = st.file_uploader(
    "Upload an image to find similar images", type=["jpg", "jpeg"]
)
expander = st.beta_expander("Examples")
with expander:
    inputs["selectbox"] = st.selectbox(
        "Choose from a search query",
        (
            default_inputs["selectbox"],
            "Two dogs playing in the snow",
            "yawning cat",
            "technical debt",
        ),
    )
    inputs["slider"] = st.slider(
        "Or pick an image", 0, photo_features.shape[0], default_inputs["slider"]
    )


# find changed input, update state
for k, v in inputs.items():
    if v != state.inputs[k]:
        state.inputs[k] = v
        break


# do search depending on input type
if k == "text_input":
    st.write("## Results")
    st.write(f"Images matching *'{v}'*")
    search_unsplash(v, photo_features, photo_ids, 20)
elif k == "file_uploader":
    image = Image.open(v)
    image_features = encode_image_query(image)
    st.write("## Results")
    st.write("Images similar to:")
    st.image(image)
    best_photo_ids = find_best_matches(image_features, photo_features, photo_ids, 20)
    display_image_grid(best_photo_ids)
elif k == "slider":
    image_vector = np.array([photo_features[v, :]])
    best_photo_ids = find_best_matches(image_vector, photo_features, photo_ids, 20)
    # TODO Skip one image
    st.write("## Results")
    st.write("Images similar to:")
    st.image(
        "https://source.unsplash.com/" + photo_ids.iloc[v]["photo_id"],
        use_column_width=True,
    )
    display_image_grid(best_photo_ids)
elif k == "selectbox":
    st.write("## Results")
    st.write(f"Images matching *'{v}'*")
    search_unsplash(v, photo_features, photo_ids, 20)
