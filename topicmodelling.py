# ================================================================
# This module loads topic modeling results (from LDA)
# and exposes interfaces for Dialogflow fulfillment to:
# - Get best_k (number of topics) per label
# - Get top words for a specific topic_id within a label
# ================================================================

import pandas as pd
from typing import List

# Load long-format topic modeling output
topic_df = pd.read_csv("topic_summary_long_format.csv")

def get_topic_count(label: str) -> int:
    """Return the best_k (number of topics) for a given label."""
    df = topic_df[topic_df["label"] == label]
    if df.empty:
        return 0
    return df["best_k"].iloc[0]  # all rows have same best_k for same label

def get_top_words(label: str, topic_id: int) -> List[str]:
    """Return top words for a specific topic_id in given label."""
    df = topic_df[(topic_df["label"] == label) & (topic_df["topic_id"] == topic_id)]
    if df.empty:
        return []
    top_words_str = df.iloc[0]["top_words"]
    return top_words_str.split(", ")  # returns a List[str]
