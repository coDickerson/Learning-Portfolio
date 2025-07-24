import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# PCA + KMeans classification 
def classify(interaction_matrix):  
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(interaction_matrix)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)

    clustered_df = interaction_matrix.copy()
    clustered_df['clusters'] = clusters

    for cluster_num in sorted(clustered_df['clusters'].unique()):
        cluster_data = clustered_df[clustered_df['clusters'] == cluster_num].drop(columns='clusters')
        top_companies = cluster_data.sum().sort_values(ascending=False).head(5)
        print(f"\nTop 5 companies for Cluster {cluster_num}:\n{top_companies}")

    #visualize(X_reduced=X_reduced, clusters=clusters) 


#visualization of PCA
def visualize(X_reduced, clusters):
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.title("Investor Cohorts via KMeans")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
