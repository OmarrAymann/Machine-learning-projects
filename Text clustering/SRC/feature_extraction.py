from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def create_tfidf_features(texts, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def apply_dimensionality_reduction(tfidf_matrix, method='svd'):

    if method == 'svd':
        svd = TruncatedSVD()
        reduced_features = svd.fit_transform(tfidf_matrix)
        return reduced_features
    elif method == 'tsne':
        # First reduce with SVD to handle sparsity
        svd = TruncatedSVD(n_components=50)  # Reduce to 50 dimensions first
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        
        # Then apply t-SNE
        tsne = TSNE(n_components=2, random_state=1234)
        reduced_features = tsne.fit_transform(reduced_matrix)
        return reduced_features
    else:
        raise ValueError("try using 'svd' or 'tsne'")