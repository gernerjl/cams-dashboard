import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/cams_long.csv")

df["text"] = df["text"].fillna("").astype(str)
df["type"] = df["type"].fillna("unknown").astype(str).str.strip().str.lower()
df["id"] = df["id"].astype(str)

# Standardize type labels
type_map = {
    "driver": "driver",
    "drivers": "driver",
    "rfd": "rfd",
    "reason for dying": "rfd",
    "reasons for dying": "rfd"
}
df["type"] = df["type"].replace(type_map)

# Keep only rows we can use
df = df[df["type"].isin(["driver", "rfd"])].reset_index(drop=True)

# -----------------------------
# 2. CREATE EMBEDDINGS
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("Encoding text...")
embeddings = model.encode(
    df["text"].tolist(),
    normalize_embeddings=True
)

df["embedding_index"] = range(len(df))

# -----------------------------
# 3. HELPER FUNCTION
# -----------------------------
def mean_off_diagonal_similarity(vectors: np.ndarray) -> float:
    """
    Mean cosine similarity among all unique within-set pairs.
    Returns NaN if there are fewer than 2 items.
    """
    if vectors.shape[0] < 2:
        return np.nan

    sim_matrix = cosine_similarity(vectors)
    upper_idx = np.triu_indices(sim_matrix.shape[0], k=1)
    return sim_matrix[upper_idx].mean()

# -----------------------------
# 4. CALCULATE PARTICIPANT-LEVEL SIMILARITIES
# -----------------------------
results = []

print("Computing participant-level similarity metrics...")

for pid, group in df.groupby("id"):
    drivers = group[group["type"] == "driver"]
    rfds = group[group["type"] == "rfd"]

    driver_idx = drivers["embedding_index"].to_numpy()
    rfd_idx = rfds["embedding_index"].to_numpy()

    driver_emb = embeddings[driver_idx] if len(driver_idx) > 0 else np.empty((0, embeddings.shape[1]))
    rfd_emb = embeddings[rfd_idx] if len(rfd_idx) > 0 else np.empty((0, embeddings.shape[1]))

    driver_driver_similarity = mean_off_diagonal_similarity(driver_emb)
    rfd_rfd_similarity = mean_off_diagonal_similarity(rfd_emb)

    if len(driver_emb) > 0 and len(rfd_emb) > 0:
        driver_rfd_similarity = cosine_similarity(driver_emb, rfd_emb).mean()
    else:
        driver_rfd_similarity = np.nan

    results.append({
        "id": pid,
        "n_drivers": len(driver_emb),
        "n_rfd": len(rfd_emb),
        "driver_driver_similarity": driver_driver_similarity,
        "rfd_rfd_similarity": rfd_rfd_similarity,
        "driver_rfd_similarity": driver_rfd_similarity
    })

results_df = pd.DataFrame(results)

# -----------------------------
# 5. SAVE OUTPUT
# -----------------------------
output_path = "data/processed/participant_similarity_results.csv"
results_df.to_csv(output_path, index=False)

print(f"Done! Saved participant similarity results to:\n{output_path}")
print("\nPreview:")
print(results_df.head())

print("\nNon-missing counts:")
print(results_df.notna().sum())