# %% Import
from reverse_image_search.haltakov_clip import *
import faiss

# %% Create Index
D = 512
M = 256  # number of subquantizers
nbits = 8
nlist = 1500  # The number of cells (space partition). Typical value is sqrt(N)
hnsw_m = 32
quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)  # this remains the same
index.train(photo_features[:20000, :])
index.add(photo_features)
faiss.write_index(index, "index.faiss")
# %%

index = faiss.read_index("index.faiss")

# %% Using Index
#!%%time
index.nprobe = 200
search_query = "technical debt"
text_features = encode_search_query(search_query)
D, I = index.search(text_features, 10)
best_photo_ids = photo_ids.iloc[I.tolist()[0]]["photo_id"].to_list()
display_image_grid(best_photo_ids)
best_photo_ids
# %% True Result
#!%%time
true_result = search_unsplash(search_query, photo_features, photo_ids, 10)
# %%
list(set(true_result) & set(best_photo_ids))
# %%
index_from_factory = faiss.index_factory(512, "OPQ64_128,IVF65536_HNSW32,PQ64")
index_from_factory.train(photo_features[: 30 * 65536, :])
index_from_factory.add(photo_features)
faiss.write_index(index_from_factory, "index_from_factory.faiss")
# %%
#!%%time
D, I = index_from_factory.search(text_features, 10)
best_photo_ids = photo_ids.iloc[I.tolist()[0]]["photo_id"].to_list()
display_image_grid(best_photo_ids)
# %%
