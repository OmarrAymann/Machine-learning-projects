import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_silhouette_scores(cluster_range, scores, dataset_name):
    plt.plot(cluster_range, scores, marker='x')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Scores for {dataset_name}")
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.savefig(f"results/silhouette_scores_{dataset_name}.png")
    plt.close()

def plot_tsne_clusters(tsne_result, clusters, title, save_path=None):
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Cluster')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_dendrogram(linkage_matrix, title, save_path=None):
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()