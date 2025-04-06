import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os
from sklearn.metrics import silhouette_score

# Import from project modules
from preprocessing import download_nltk_resources, preprocess_text, add_text_features
from feature_extraction import create_tfidf_features, apply_dimensionality_reduction
from clustering import kmeans_clustering, hierarchical_clustering
from evaluation import compute_silhouette_scores, purity_score
from visualization import plot_silhouette_scores, plot_tsne_clusters, plot_dendrogram

def load_datasets():
    # Load 20 Newsgroups
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data
    labels = newsgroups.target
    newsgroups_df = pd.DataFrame({"text": texts, "category": labels})
    # Load Wikipedia dataset (assuming it exists)
    try:
        wiki_df = pd.read_csv('Text clustring\Dataset\people_wiki.csv')
    except FileNotFoundError:
        print("Wikipedia dataset not found. Processing only Newsgroups.")
        wiki_df = None
    return newsgroups_df, wiki_df, labels

def create_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/newsgroups", exist_ok=True)
    os.makedirs("results/wikipedia", exist_ok=True)
def main():
    create_directories()

    download_nltk_resources()

    newsgroups_df, wiki_df, true_labels = load_datasets()


    newsgroups_df = add_text_features(newsgroups_df)
    newsgroups_df["processed_text"] = newsgroups_df["text"].apply(preprocess_text)
    news_tfidf, news_vectorizer = create_tfidf_features(newsgroups_df["processed_text"])
    news_pca = apply_dimensionality_reduction(news_tfidf, method='svd')
    news_tsne = apply_dimensionality_reduction(news_tfidf, method='tsne')
    
    cluster_range = range(2, 10)
    news_silhouette_scores = compute_silhouette_scores(news_pca, cluster_range)
    
    # Visualize silhouette scores
    plot_silhouette_scores(cluster_range, news_silhouette_scores, "20 News Groups")
    
    # Select optimal number of clusters (for newsgroups, using k=3 as in original code)
    optimal_k = 3
    news_clusters, news_kmeans = kmeans_clustering(news_pca, n_clusters=optimal_k)
    
    # Evaluate clustering
    news_silhouette = silhouette_score(news_pca, news_clusters)
    news_purity = purity_score(true_labels, news_clusters)
    
    print(f"Newsgroups Clustering Results:")
    print(f"- Silhouette Score: {news_silhouette:.4f}")
    print(f"- Purity Score: {news_purity:.4f}")
    
    # Visualize newsgroups clusters
    plot_tsne_clusters(news_tsne, news_clusters, 
                    "t-SNE of 20 News Groups Clusters",
                    save_path="results/newsgroups_tsne.png")
    
    # Process Wikipedia dataset if available
    if wiki_df is not None:
        print("Processing Wikipedia dataset...")
        wiki_df = add_text_features(wiki_df)
        wiki_df["processed_text"] = wiki_df["text"].apply(preprocess_text)
        
        # Feature extraction for Wikipedia
        wiki_tfidf, wiki_vectorizer = create_tfidf_features(wiki_df["processed_text"])
        wiki_pca = apply_dimensionality_reduction(wiki_tfidf, method='svd')
        wiki_tsne = apply_dimensionality_reduction(wiki_tfidf, method='tsne')
        
        # Cluster Wikipedia data
        wiki_silhouette_scores = compute_silhouette_scores(wiki_pca, cluster_range)
        
        # Visualize silhouette scores
        plot_silhouette_scores(cluster_range, wiki_silhouette_scores, "Wikipedia")
        
        # Select optimal number of clusters (for Wikipedia, using M=3 as in original code)
        optimal_M = 3
        wiki_clusters, wiki_kmeans = kmeans_clustering(wiki_pca, n_clusters=optimal_M)
        
        # Evaluate clustering
        wiki_silhouette = silhouette_score(wiki_pca, wiki_clusters)
        
        print(f"Wikipedia Clustering Results:")
        print(f"- Silhouette Score: {wiki_silhouette:.4f}")
        
        # Visualize Wikipedia clusters
        plot_tsne_clusters(wiki_tsne, wiki_clusters, 
                        "t-SNE of Wikipedia Clusters",
                        save_path="results/wikipedia_tsne.png")
        # Hierarchical clustering for Wikipedia
        wiki_linkage = hierarchical_clustering(wiki_pca)
        plot_dendrogram(wiki_linkage, "Dendrogram of Wikipedia Dataset", 
                        save_path="results/wikipedia_dendrogram.png")


if __name__ == "__main__":
    main()