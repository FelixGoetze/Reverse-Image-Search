# %%
import torch
from reverse_image_search.CLIP import clip
from reverse_image_search.CLIP import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

image = (
    transform(Image.open("reverse_image_search/CLIP/CLIP.png")).unsqueeze(0).to(device)
)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# %%
