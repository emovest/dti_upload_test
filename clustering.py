# ================================================================
# We identified "BERT-based + K-Means" as the champion clustering model,
# achieving the highest Kappa Score, Silhouette Score, V-Measure, and Adjusted Rand Index (ARI).
# Therefore, we adopt this model to group semantically similar papers and 
# identify latent structures (distinct clusters) within the papers 
# belonging to the user's interested domain.
# ================================================================

import pandas as pd
from typing import List



clustered_df = pd.read_csv("clustered_all_labels.csv")
bigrams_df = pd.read_csv("bigrams_summary.csv")


def get_cluster_count(label: str) -> int:

    if label not in clustered_df['label'].unique():
        return 0
    return clustered_df[clustered_df["label"] == label]["cluster"].nunique()


def get_top_bigrams(label: str, cluster_id: int) -> list:

    df = bigrams_df[(bigrams_df["label"] == label) & (bigrams_df["cluster"] == cluster_id)]
    if df.empty:
        return []
    top_bigrams_str = df.iloc[0]["top_bigrams"]
    return top_bigrams_str.split(", ")  # 返回 list




