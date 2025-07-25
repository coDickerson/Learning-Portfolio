import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classificationcluster import classify, visualize
from source_company import SourceCompany

#cleans excel spreadsheet into dataframe
df = pd.read_excel("Conf 2024 Request List Update.xlsx")
df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
df = df.rename(columns = 
            {
            'Target Company - who was requested to meet':'target company',
            'Source Company - who made the request':'source company', 
            'Request Date Created' : 'request date'
            }
        )
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
cols = df.columns.to_list()
cols[0], cols[1] = cols[1], cols[0]
df = df[cols] # flips source company and target company columns
df = df.drop_duplicates()
# print(df.head(10))

grouped = df.groupby('source_company')
source_company_map = {}
for source_id, group in grouped:
    requested = group['target_company'].tolist()
    dates = set(group['request_date']) 
    source_company_map[source_id] = SourceCompany(source_id, requested, dates)

print(source_company_map[1001])

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
significant_pairs = company_pairs[company_pairs['correlation'] > 0.8] # filters for only significant company pairs; .5 arbitrary
significant_pairs = significant_pairs.sort_values(by='correlation', ascending=False)
# print(significant_pairs)

# visualization
# top_n = 20
# top_companies = df['target_company'].value_counts().nlargest(top_n).index
# filtered = interaction_matrix[top_companies]
# filtered_corr = filtered.corr()

# plt.figure(figsize=(10, 12))
# sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation of Requested Companies")
# plt.show()


