import numpy as np

# Suppose we have two recommendation scores for each movie:
# 1. Collaborative filtering score (e.g., from SVD)
collab_scores = np.array([0.8, 0.6, 0.9, 0.2])

# 2. Content-based score (e.g., from TF-IDF + cosine similarity)
content_scores = np.array([0.7, 0.5, 0.85, 0.3])

# Combine the two scores with a weighted average
hybrid_scores = 0.6 * collab_scores + 0.4 * content_scores

# Recommend the top movie based on the hybrid scores
top_movie_idx = np.argmax(hybrid_scores)
print(f"Top recommended movie index: {top_movie_idx}")
