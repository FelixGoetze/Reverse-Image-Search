# %%
from reverse_image_search.haltakov_clip import *

# %% Index
import faiss

d = 512
nlist = 100
m = 8  # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
# 8 specifies that each sub-vector is encoded as 8 bits
index.train(photo_features)
index.add(photo_features)
# %%
search_query = "yawning cat on couch"
text_features = encode_search_query(search_query)
index.nprobe = 10
# %%
D, I = index.search(text_features, 50)
print(photo_ids.iloc[I.tolist()[0]])
# %%
faiss.write_index(index, "index.faiss")
# %%
from IPython.core.display import display, HTML

display(HTML(r"""<img src="https://source.unsplash.com/hQgxyi8Oduo">"""))
# %%
imageurl = "https://source.unsplash.com/" + "hQgxyi8Oduo"
# %%

# %%
