import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def compute_silhouette_scores(data, cluster_range):
    silhouette_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(data)
        score = silhouette_score(data, clusters)
        silhouette_scores.append(score)
    return silhouette_scores

def purity_score(y_true, y_pred):
    contingency_matrix = np.zeros((np.max(y_pred) + 1, np.max(y_true) + 1))
    for i in range(len(y_pred)):
        contingency_matrix[y_pred[i], y_true[i]] += 1
    dominant_classes = np.max(contingency_matrix, axis=1)
    purity = np.sum(dominant_classes) / np.sum(contingency_matrix)
    return purity
