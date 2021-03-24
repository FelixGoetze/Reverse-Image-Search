# %% Import
import clip
import torch
import streamlit as st
import os
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image, ExifTags
import SessionState

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# %%
@st.cache(allow_output_mutation=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    return device, model, transform


# %%
device, model, transform = load_model()
# %%
# Jinja Template settings
root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(root, "templates")
env = Environment(loader=FileSystemLoader(templates_dir))


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
def find_best_matches(
    text_features, photo_features, photo_ids, results_count=3, skip_first=False
):
    knn_distances = (photo_features @ text_features.T).squeeze(1)

    if skip_first:
        idx = np.argpartition(knn_distances, -results_count - 1)[-results_count - 1 :]
        knn_indices = idx[np.argsort(knn_distances[idx])][-2::-1]
    else:
        idx = np.argpartition(knn_distances, -results_count)[-results_count:]
        knn_indices = idx[np.argsort(knn_distances[idx])][::-1]

    result = []
    for knn_index in knn_indices:
        result.append(photo_ids.iloc[knn_index]["photo_id"])
    return result


# %% Render Imagegrid from Template
def render_from_template(template_filename, best_photo_ids, write_to_disk):
    template = env.get_template(template_filename)
    html_image_grid = template.render(ids=best_photo_ids)
    if write_to_disk:
        output_file = os.path.join(root, "html", template_filename)
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


# %%
def open_and_rotate(v):
    # https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image/
    try:
        image = Image.open(v)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

        return image

    except (AttributeError, KeyError, IndexError):
        image = Image.open(v)
        return image


# %% GUI natural language search
st.write("# Search 2 million Unsplash images")


default_inputs = {
    "text_input": "e.g. 'modern villa with swimming pool'",
    "file_uploader": None,
    "selectbox": "Select an example",
    "slider": 0,
}
# Get state of the inputs from their defaults
state = SessionState.get(inputs=default_inputs)

inputs = {}
inputs["text_input"] = st.text_input("Describe the image", default_inputs["text_input"])
inputs["file_uploader"] = st.file_uploader(
    "Or upload an image to find similar images", type=["jpg", "jpeg"]
)
expander = st.beta_expander("Examples")
with expander:
    inputs["selectbox"] = st.selectbox(
        "Choose from a search query",
        (
            default_inputs["selectbox"],
            "antique bicycle",
            "oldtimer mercedes",
            "red couch",
            "rustic kitchen",
            "garden with fountain",
        ),
    )
    inputs["slider"] = st.slider(
        "Or pick an image", 0, photo_features.shape[0], default_inputs["slider"]
    )

key = "text_input"
value = "modern villa with swimming pool"


# find changed input, update state
for k, v in inputs.items():
    if v != state.inputs[k]:
        key = k
        value = v
        state.inputs[k] = v
        break


# do search depending on input type
if key == "text_input":
    st.write("## Results")
    st.write(f"Images matching *'{value}'*")
    search_unsplash(value, photo_features, photo_ids, 10)
elif key == "file_uploader":
    image = open_and_rotate(value)
    image_features = encode_image_query(image)
    st.write("## Results")
    st.write("Images similar to:")
    st.image(image)
    best_photo_ids = find_best_matches(image_features, photo_features, photo_ids, 10)
    display_image_grid(best_photo_ids)
elif key == "slider":
    image_vector = np.array([photo_features[value, :]])
    best_photo_ids = find_best_matches(
        image_vector, photo_features, photo_ids, 10, True
    )
    st.write("## Results")
    st.write("Images similar to:")
    st.image(
        "https://source.unsplash.com/" + photo_ids.iloc[value]["photo_id"],
        use_column_width=True,
    )
    display_image_grid(best_photo_ids)
elif key == "selectbox":
    st.write("## Results")
    st.write(f"Images matching *'{value}'*")
    search_unsplash(value, photo_features, photo_ids, 10)
