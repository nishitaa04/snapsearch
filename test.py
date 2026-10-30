import os
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")

folder = "data/Images"
first_file = sorted(os.listdir(folder))[0]
print("Testing with:", first_file)

img = Image.open(os.path.join(folder, first_file))
img_emb = model.encode(img)
text_emb = model.encode("a dog running on grass")

print(img_emb.shape)
print(text_emb.shape)