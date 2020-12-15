from img2vec_pytorch import Img2Vec
from PIL import Image
import glob
import json
import numpy as np

input_path = "images/APIdataset"

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
    with open("data/pictures.json", "w") as filehandle:
        json.dump(imagelist, filehandle)
    vectors = vectors.astype("float32")
    np.save("data/vectors.npy", vectors)
    return
