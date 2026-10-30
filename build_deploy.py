import os
import numpy as np
from PIL import Image

embeddings = np.load("embeddings.npy")
with open("filenames.txt") as f:
    filenames = f.read().split("\n")

rng = np.random.default_rng(42)
idx = np.arange(len(filenames))
os.makedirs("deploy/images", exist_ok=True)

for n, i in enumerate(idx):
    img = Image.open(os.path.join("data", "Images", filenames[i])).convert("RGB")
    img.thumbnail((400, 400))
    img.save(os.path.join("deploy", "images", filenames[i]), quality=80)
    if n % 200 == 0:
        print(f"Done {n} / {len(idx)}")

np.save("deploy/embeddings.npy", embeddings[idx])
with open("deploy/filenames.txt", "w") as f:
    f.write("\n".join(filenames[i] for i in idx))

print("Saved:", embeddings[idx].shape)