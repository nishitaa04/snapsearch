import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")

folder = "data/Images"
files = sorted(os.listdir(folder))

TEST_MODE = False
if TEST_MODE:
    files = files[:200]

all_embeddings = []
batch_size = 32

for i in range(0, len(files), batch_size):
    batch_files = files[i:i + batch_size]
    images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in batch_files]
    emb = model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
    all_embeddings.append(emb)
    print(f"Done {i + len(batch_files)} / {len(files)}")

embeddings = np.vstack(all_embeddings)
np.save("embeddings.npy", embeddings)

with open("filenames.txt", "w") as f:
    f.write("\n".join(files))

print("Saved:", embeddings.shape)