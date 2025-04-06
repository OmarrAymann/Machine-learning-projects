from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage

def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def hierarchical_clustering(data, method="ward"):
    linkage_matrix = linkage(data, method=method)
    return linkage_matrix
