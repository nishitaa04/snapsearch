import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")

embeddings = np.load("embeddings.npy")
with open("filenames.txt") as f:
    filenames = f.read().split("\n")

df = pd.read_csv("data/captions.txt")
print(df.head())

df = df.sample(2000, random_state=42)
captions = df["caption"].tolist()
true_files = df["image"].tolist()

query_embs = model.encode(
    captions, batch_size=64, convert_to_numpy=True,
    normalize_embeddings=True, show_progress_bar=True
)

scores = query_embs @ embeddings.T

file_to_idx = {name: i for i, name in enumerate(filenames)}
true_idx = np.array([file_to_idx[f] for f in true_files])

true_scores = scores[np.arange(len(scores)), true_idx]
ranks = (scores > true_scores[:, None]).sum(axis=1) + 1

for k in [1, 5, 10]:
    print(f"Recall@{k}: {(ranks <= k).mean() * 100:.1f}%")