# %%
from img2vec_pytorch import Img2Vec
from PIL import Image
import glob
import numpy as np
import os
import pickle
from tqdm import tqdm

input_path = "images/UnsplashAPIdataset/"
img2vec = Img2Vec(cuda=False)
rebuild_vectors = False

# %%
def compute_vector(image_key):
    img = Image.open(input_path + image_key + ".jpg")
    vector = img2vec.get_vec(img)
    return vector


def save_vectors(vectors, imagedict):
    with open("data/images2.pickle", "wb") as filehandle:
        pickle.dump(imagedict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
    vectors = vectors.astype("float32")
    np.save("data/vectors2.npy", vectors)


# %% Rebuild vectors and imagelist and write to disk
def build_vectors(input_path="images/UnsplashAPIdataset/", rebuild_vectors=False):
    # %%
    # load images dict from folder
    imagedict_jpgs = {
        os.path.basename(x).split(".")[0]: None for x in glob.glob(input_path + "*.jpg")
    }
    # preallocate array
    vectors = np.empty((0, 512), float)
    imagedict = {}
    imageids = set()
    if rebuild_vectors:
        imageids = set(imagedict_jpgs.keys())
    if not rebuild_vectors:
        vectors = np.load("data/vectors2.npy")
        with open("data/images2.pickle", "rb") as filehandle:
            imagedict = pickle.load(filehandle)
        imageids = set(imagedict_jpgs.keys()) - set(imagedict.keys())

    for key in tqdm(imageids):
        try:
            vectors = np.append(
                vectors, [compute_vector(key)], axis=0
            )  # try list, then convert.
            imagedict[key] = None
        except:
            print("error:" + key)
            save_vectors(vectors, imagedict)
    save_vectors(vectors, imagedict)
    # save_vectors(vectors, imagedict)
    # %%
    return


# %%
build_vectors(input_path, rebuild_vectors=False)
