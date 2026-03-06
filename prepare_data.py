import pandas as pd
import umap
from sentence_transformers import SentenceTransformer

# 1. Load raw data
df = pd.read_csv("data/processed/cams_long.csv")

# 2. Clean required columns
df["text"] = df["text"].fillna("").astype(str)
df["type"] = df["type"].fillna("Unknown").astype(str)

# 3. Add row id
df = df.reset_index(drop=True)
df["row_id"] = df.index

# 4. Load embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 5. Compute embeddings
texts = df["text"].tolist()
embeddings = model.encode(texts, normalize_embeddings=True)

# 6. Run UMAP
reducer = umap.UMAP(
    n_components=3,
    metric="cosine",
    random_state=42
)
coords = reducer.fit_transform(embeddings)

# 7. Save coordinates back into dataframe
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]
df["z"] = coords[:, 2]

# 8. Save processed data
df.to_csv("data/processed/cams_long_umap.csv", index=False)

print("Done! Saved processed file to data/processed/cams_long_umap.csv")