# =============================================================================
# FILE: utils_cluster.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Clustering utilities for OCR box grouping.
#              Uses KMeans to group bounding boxes by vertical position.
# =============================================================================

from sklearn.cluster import KMeans
import numpy as np

def cluster_boxes_by_vertical_position(boxes: list, n_clusters=2) -> dict:
    y_centers = np.array([[(y1 + y2) / 2] for _, y1, _, y2 in boxes])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(y_centers)
    labels = kmeans.labels_

    clustered = {}
    for label, box in zip(labels, boxes):
        clustered.setdefault(label, []).append(box)

    return clustered
