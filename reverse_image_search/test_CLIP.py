# %% import
import torch
from reverse_image_search.CLIP import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# %% Load Data
image = (
    transform(Image.open("reverse_image_search/CLIP/CLIP.png")).unsqueeze(0).to(device)
)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# %% Encode Data
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# %% Inference
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# %% Measure Distances of embeddings

print(torch.pdist(text_features))
print(torch.cdist(text_features, text_features))

# Distance between embeddings doesn't replace model for text to/from image
print(torch.cdist(image_features, text_features))

# %%
