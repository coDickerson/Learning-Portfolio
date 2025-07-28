import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd


# PCA + KMeans classification 
def classify(recommendation_df, interaction_matrix):  
    recommended_companies = recommendation_df['recommended company'].tolist()
    transposed_matrix = interaction_matrix.T

    subset_matrix = transposed_matrix.loc[transposed_matrix.index.intersection(recommended_companies)]

    if subset_matrix.shape[0] < 2:
            print("\nNot enough companies for clustering.")
            return

    pca = PCA(n_components=min(10, subset_matrix.shape[0]))
    X_reduced = pca.fit_transform(subset_matrix)

    kmeans = KMeans(n_clusters=min(3, subset_matrix.shape[0]), random_state=42)
    clusters = kmeans.fit_predict(X_reduced)

    clustered_df = subset_matrix.copy()
    clustered_df['clusters'] = clusters

    for cluster_num in sorted(clustered_df['clusters'].unique()):
        cluster_data = clustered_df[clustered_df['clusters'] == cluster_num].drop(columns='clusters')
        top_companies = cluster_data.sum(axis=1).sort_values(ascending=False).head(5)
        print(f"\nTop 5 companies for Cluster {cluster_num}:\n{top_companies}")

    visualize(X_reduced, clusters, labels=subset_matrix.index.tolist()) 


# visualization of PCA
def visualize(X_reduced, clusters, labels):
     # Build a DataFrame for plotting
    df_plot = pd.DataFrame({
        'PC1': X_reduced[:, 0],
        'PC2': X_reduced[:, 1],
        'Cluster': clusters,
        'Company': labels
    })

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='Company',
        title='Investor Cohorts via KMeans (Hoverable)',
        width=900,
        height=600
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(showlegend=True)
    fig.show()
