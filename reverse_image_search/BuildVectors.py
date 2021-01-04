# %%
from img2vec_pytorch import Img2Vec
from PIL import Image
import glob
import numpy as np
import os
import pickle
from tqdm import tqdm
import time

input_path = "images/UnsplashAPIdataset/"
img2vec = Img2Vec(cuda=False)

# %%
def compute_vector(image_key):
    img = Image.open(input_path + image_key + ".jpg")
    vector = img2vec.get_vec(img)
    return vector


def save_vectors():
    with open("data/images2.pickle", "wb") as filehandle:
        pickle.dump(imagedict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
    vectors = vectors.astype("float32")
    np.save("data/vectors2.npy", vectors)


# %% Rebuild vectors and imagelist and write to disk
def build_vectors(input_path="images/UnsplashAPIdataset/"):
    # %%
    # load all jpgs from as imagedict_jpgs
    imagedict = {}
    imagedict_jpgs = {
        os.path.basename(x).split(".")[0]: None for x in glob.glob(input_path + "*.jpg")
    }
    vectors = np.ones((len(imagedict_jpgs), 512))
    try:
        for index, key in enumerate(tqdm(imagedict_jpgs)):
            vectors[index, :] = compute_vector(key)
            imagedict[key] = None
    except KeyboardInterrupt:
        save_vectors
    save_vectors
    # %%
    return


# %%
build_vectors(input_path)
