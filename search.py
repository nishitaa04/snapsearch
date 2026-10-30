import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")

embeddings = np.load("embeddings.npy")
with open("filenames.txt") as f:
    filenames = f.read().split("\n")

def search_by_text(query, top_k=5):
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(filenames[i], float(scores[i])) for i in top_idx]

def search_by_image(image_path, top_k=5):
    img = Image.open(image_path).convert("RGB")
    query_emb = model.encode(img, normalize_embeddings=True)
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(filenames[i], float(scores[i])) for i in top_idx]

if __name__ == "__main__":
    query = input("What are you looking for? ")
    results = search_by_text(query)
    for name, score in results:
        print(f"{score:.3f}  {name}")
    Image.open("data/Images/" + results[0][0]).show()