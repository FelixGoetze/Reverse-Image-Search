# %% Import
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

# %% Load data
photo_ids = pd.read_csv("data/photo_ids.csv")

# %% create image ids
images = photo_ids["photo_id"].iloc[0:20].to_list()

# %% load jinja templates
from jinja2 import Environment, FileSystemLoader
import os

root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(root, "templates")
env = Environment(loader=FileSystemLoader(templates_dir))

# %% Render from template
def render_from_template(template_filename="index.html", write_to_disk=True):
    template = env.get_template(template_filename)
    output_file = os.path.join(root, "html", template_filename)
    html_image_grid = template.render(ids=images)
    if write_to_disk:
        with open(output_file, "w") as fh:
            fh.write(html_image_grid)
    return html_image_grid


# %%
html_image_grid = render_from_template(
    template_filename="index.html", write_to_disk=True
)

# %% Show image grid
st.components.v1.html(html_image_grid, height=2400, scrolling=True)

# %%
