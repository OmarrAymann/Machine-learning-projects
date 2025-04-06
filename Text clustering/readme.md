# Document Clustering Project

## Overview
This project applies unsupervised learning techniques to cluster documents from two datasets: the People Wikipedia Dataset and the 20 Newsgroups Dataset. The objective is to identify inherent structures within the datasets and uncover natural groupings of documents.

## Datasets
### 1. People Wikipedia Dataset
#### Description
The People Wikipedia Dataset consists of biographical articles of notable individuals extracted from Wikipedia. It allows for analyzing relationships between individuals based on the content of their biographies, including similarities in professions and historical relevance.

#### Features
- **URI**: Unique identifier for each person’s Wikipedia page.
- **Name**: Full name of the individual.
- **Text**: Extracted content from their Wikipedia biography.

### 2. 20 Newsgroups Dataset
#### Description
The 20 Newsgroups Dataset contains around 20,000 newsgroup documents, categorized into 20 different newsgroups. The content covers a variety of topics, including politics, religion, sports, and technology.

#### Features
- **Content**: Main body of the newsgroup post.
- **Headers**: Metadata such as subject line, author, and date.

## **Structured Code**:
     ├── src/
     │   ├── preprocessing.py    # Code for data cleaning and preprocessing
     │   ├── feature_extraction.py  # Code for vectorization and embedding
     │   ├── clustering.py       # Code for implementing clustering models
     │   ├── evaluation.py       # Code for computing clustering metrics
     │   ├── visualization.py    # Code for plotting results
     │   ├── main.py             # Main script to run the project pipeline
     ├── results/                # Folder to save cluster results and visualizations
     ├── requirements.txt        # List of required Python libraries
     ├── README.md               # Project documentation
     ```

## Methodology
### 1. Data Collection
- Obtain the datasets from their respective sources.

### 2. Data Preprocessing
- **Cleaning**: Remove noise, handle missing values, and normalize text (e.g., lowercasing, stemming).
- **Tokenization**: Split text into tokens (words or phrases).
- **Stop Words Removal**: Eliminate common words that do not contribute to clustering (e.g., "the", "and").

### 3. Feature Extraction
- **TF-IDF Vectorization**: Convert textual data into numerical features.

### 4. Clustering Algorithms
- **K-Means Clustering**: Partition documents into 'k' clusters based on feature similarity.
- **Hierarchical Clustering**: Create a tree-like structure representing nested groupings of documents.

### 5. Evaluation of Clusters
- **Silhouette Score**: Measure how well a document fits within its cluster.
- **Purity Score**: Assess the homogeneity of clusters.

### 6. Visualization
- **t-SNE / PCA**: Reduce dimensionality for cluster visualization.
- **Dendrograms**: Visualize hierarchical clustering results.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - **Data Manipulation**: pandas, NumPy
  - **Text Processing**: NLTK, spaCy
  - **Machine Learning**: scikit-learn
  - **Visualization**: matplotlib, seaborn


