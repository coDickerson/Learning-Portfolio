import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#cleans excel spreadsheet into dataframe
df = pd.read_excel("Conf 2024 Request List Update.xlsx")
df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last', 'Request Date Created'])
df = df.rename(columns = 
            {
            'Target Company - who was requested to meet':'target company',''
            'Source Company - who made the request':'source company', 
            }
        )
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
cols = df.columns.to_list()
cols[0], cols[1] = cols[1], cols[0]
df = df[cols] # flips source company and target company columns
df = df.drop_duplicates()
# print(df.head())

# binary interaction matrix
# describes if a target company was requested by a source, 1=yes and 0=no
interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])
# print(interaction_matrix)

# pearson correlation
company_corr = interaction_matrix.corr(method='pearson')
# print(company_corr.head())

# new table showing significant correlation between two companies 
company_corr.columns.name = None
company_corr.index.name = None
company_pairs = company_corr.stack().reset_index() # flattens square matrix
company_pairs.columns = ['company_a', 'company_b', 'correlation']
company_pairs = company_pairs[company_pairs['company_a'] < company_pairs['company_b']] # deals with duplicates and same company correlations
significant_pairs = company_pairs[company_pairs['correlation'] > 0.5] # filters for only significant company pairs; .5 arbitrary
significant_pairs = significant_pairs.sort_values(by='correlation', ascending=False)
# print(significant_pairs.head(10))

# visualization
# top_n = 20
# top_companies = df['target_company'].value_counts().nlargest(top_n).index
# filtered = interaction_matrix[top_companies]
# filtered_corr = filtered.corr()

# plt.figure(figsize=(12, 12))
# sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation of Requested Companies")
# plt.show()

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(interaction_matrix)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

clustered_df = interaction_matrix.copy()
clustered_df['clusters'] = clusters

for cluster_num in sorted(clustered_df['clusters'].unique()):
    cluster_data = clustered_df[clustered_df['clusters'] == cluster_num].drop(columns='clusters')
    top_companies = cluster_data.sum().sort_values(ascending=False).head(5)
    print(f"\nTop 5 companies for Cluster {cluster_num}:\n{top_companies}")

# visualization
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='tab10', alpha=0.7)
plt.title("Investor Cohorts via KMeans")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
