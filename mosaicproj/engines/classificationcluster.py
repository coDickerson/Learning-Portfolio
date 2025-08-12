from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

def excel_classify(df):
    classify_df = pd.read_excel("data/company-list_cp.xls")
    classify_df = classify_df.drop(columns=['Days Attending', 'Reps'])
    classify_df = classify_df.rename(columns=
        {
            'Company Name' : 'recommended company',
        }
    )
    classify_df.columns = [col.strip().lower().replace(" ", "_") for col in classify_df.columns]
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    combined_df = pd.merge(df, classify_df, on='recommended_company', how='left')
    return combined_df

def new_classify(recommendation_df, interaction_matrix):
    combined_df = excel_classify(recommendation_df)
    
    # Add sector-based classification
    if 'sector' in combined_df.columns:
        # Sector distribution visualization
        
        # Sector-based PCA if enough data
        if len(combined_df) >= 5:
            sector_pca_classify(combined_df, interaction_matrix)
    
    return combined_df

def sector_distribution_chart(combined_df):
    """Bar chart showing distribution of recommended companies by sector"""
    if 'sector' not in combined_df.columns:
        print("No sector information available")
        return
        
    sector_counts = combined_df['sector'].value_counts()
    
    fig = px.bar(
        x=sector_counts.index,
        y=sector_counts.values,
        title='Recommended Companies by Sector',
        labels={'x': 'Sector', 'y': 'Number of Companies'},
        color=sector_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    fig.show()

def sector_pca_classify(combined_df, interaction_matrix):
    """PCA classification colored by sector instead of clusters"""
    # Handle both column name formats
    company_col = 'recommended_company' if 'recommended_company' in combined_df.columns else 'recommended company'
    recommended_companies = combined_df[company_col].tolist()
    transposed_matrix = interaction_matrix.T
    
    subset_matrix = transposed_matrix.loc[transposed_matrix.index.intersection(recommended_companies)]
    
    if subset_matrix.shape[0] < 2:
        print("\nNot enough companies for PCA analysis.")
        return
    
    pca = PCA(n_components=min(10, subset_matrix.shape[0]))
    X_reduced = pca.fit_transform(subset_matrix)
    
    # Get sectors for the companies in the subset, including those without sector info
    company_sectors = combined_df.set_index(company_col)['sector'].to_dict()
    sectors = []
    for company in subset_matrix.index:
        sector = company_sectors.get(company, 'No Sector')
        # Handle NaN, None, or empty string values
        if pd.isna(sector) or sector == '' or sector is None:
            sector = 'No Sector'
        sectors.append(sector)
    
    # Get correlation/similarity scores for hover tooltips
    score_col = None
    for col in combined_df.columns:
        if col in ['similarity', 'correlation', 'recommendation_score']:
            score_col = col
            break
    
    if score_col:
        company_scores = combined_df.set_index(company_col)[score_col].to_dict()
        scores = [company_scores.get(company, 0) for company in subset_matrix.index]
    else:
        scores = [0] * len(subset_matrix.index)
    
    sector_pca_visualize(X_reduced, sectors, subset_matrix.index.tolist(), scores, score_col)

def sector_pca_visualize(X_reduced, sectors, labels, scores=None, score_col_name=None):
    """PCA scatter plot colored by sector with dull color for companies without sector and correlation in hover"""
    df_plot = pd.DataFrame({
        'PC1': X_reduced[:, 0],
        'PC2': X_reduced[:, 1],
        'Sector': sectors,
        'Company': labels
    })
    
    # Add scores if provided
    if scores and score_col_name:
        df_plot['Score'] = scores
        df_plot['Score_Label'] = f"{score_col_name.title()}"
    else:
        df_plot['Score'] = [0] * len(labels)
        df_plot['Score_Label'] = ['N/A'] * len(labels)

    # Create a custom color map that gives "No Sector" companies a dull gray color
    unique_sectors = df_plot['Sector'].unique()
    colors = []
    
    for sector in unique_sectors:
        if sector == 'No Sector':
            colors.append('#808080')  # Dull gray for companies without sector
        else:
            colors.append(None)  # Let plotly choose colors for other sectors
    
    # Create color map
    color_map = dict(zip(unique_sectors, colors))
    
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Sector',
        hover_data=['Company', 'Sector', 'Score', 'Score_Label'],
        title='Company Cohorts by Sector (PCA)',
        width=900,
        height=600,
        color_discrete_map=color_map
    )
    
    # Update hover template to show custom text
    if scores and score_col_name:
        fig.update_traces(
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate='<b>%{customdata[0]}</b><br>Sector: %{customdata[1]}<br>%{customdata[3]}: %{customdata[2]:.3f}<extra></extra>'
        )
    else:
        fig.update_traces(
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate='<b>%{customdata[0]}</b><br>Sector: %{customdata[1]}<extra></extra>'
        )
    
    fig.show()

def PCA_Kmeans_visualize(X_reduced, sectors, labels):
    """PCA scatter plot colored by sector"""
    df_plot = pd.DataFrame({
        'PC1': X_reduced[:, 0],
        'PC2': X_reduced[:, 1],
        'Sector': sectors,
        'Company': labels
    })

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Sector',
        hover_name='Company',
        title='Company Cohorts by Sector (PCA)',
        width=900,
        height=600
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()

# PCA + KMeans classification 
def PCA_Kmeans_classify(recommendation_df, interaction_matrix):  
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

    # for cluster_num in sorted(clustered_df['clusters'].unique()):
    #     cluster_data = clustered_df[clustered_df['clusters'] == cluster_num].drop(columns='clusters')
    #     top_companies = cluster_data.sum(axis=1).sort_values(ascending=False).head(5)
    #     print(f"\nTop 5 companies for Cluster {cluster_num}:\n{top_companies}")

    PCA_Kmeans_visualize(X_reduced, clusters, labels=subset_matrix.index.tolist()) 


# hoverable visualization of PCA
def PCA_Kmeans_visualize(X_reduced, clusters, labels):
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
        title='Company Cohorts via KMeans (Hoverable)',
        width=900,
        height=600
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(showlegend=True)
    fig.show()
