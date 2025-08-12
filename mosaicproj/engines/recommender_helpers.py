import pandas as pd

def convert_to_recommendations_df(scores, method, top_n):
    """
    Convert a dictionary of company scores to a standardized recommendations DataFrame.

    Args:
        scores (dict): Dictionary with company names as keys and scores as values
        method (str): The recommendation method ('pairwise', 'multivector', 'hybrid')
        top_n (int): Number of top recommendations to return
    
    Returns:
        pd.DataFrame: Formatted recommendations with appropriate column names
    """
    score_column = get_score_column_name(method)

    if not scores:
        # Return empty DataFrame with appropriate columns based on method
        return pd.DataFrame(columns=['recommended company', score_column])
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(
        list(scores.items()),
        columns=['recommended company', score_column]
    ).sort_values(score_column, ascending=False)
    recommendations_df = excel_classify(recommendations_df)
    # Print results with method-specific messaging
    print_results(recommendations_df, method, top_n)
    
    return recommendations_df

def print_results(recommendations_df, method, top_n):
    """Print method-specific results."""
    if recommendations_df.empty:
        print(f"\nNo {method} recommendations found.")
        return
        
    method_messages = {
        'pairwise': f"\nPairwise recommendations based on company correlations:",
        'multivector': f"\nMultivector recommendations based on similar investor preferences:",
    }
    
    message = method_messages.get(method.lower(), f"\n{method.title()} recommendations:")
    print(message)
    print(recommendations_df.head(top_n))

def get_score_column_name(method) :
    score_columns = {
        'pairwise' : 'correlation',
        'multivector': 'similarity',
    }
    return score_columns.get(method.lower(), 'score')

def visualize_results(recommendations_df, interaction_matrix, method):
    """
    Visualization function compatible
    """
    if recommendations_df.empty:
        print("No recommendations to visualize.")
        return
    
    print(f"\n--- {method.upper()} METHOD RESULTS ---")
    print(f"Found {len(recommendations_df)} recommendations")
    
    while True:
        user_input = input('\nWould you like a visualization (y/n): ').strip().lower()
        if user_input == 'y':
            try:
                new_classify(recommendations_df, interaction_matrix)
                break
            except ImportError:
                print("Could not import classify function. Skipping visualization.")
                break
        elif user_input == 'n':
            print("Visualization skipped.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    
    return recommendations_df

def filter_companies_by_phrases(df, exclude_phrases, column_name='target_company'):
    """
    Remove companies that contain any of the specified phrases.
    
    Args:
        df: DataFrame to filter
        exclude_phrases: List of phrases to exclude (case-insensitive)
        column_name: Column containing company names to check
    
    Returns:
        Filtered DataFrame
    """
    if not exclude_phrases:
        return df
    
    # Convert to lowercase for case-insensitive matching
    exclude_phrases_lower = [phrase.lower() for phrase in exclude_phrases]
    
    # Create a mask for companies to keep (those that don't contain any excluded phrases)
    mask = ~df[column_name].astype(str).str.lower().str.contains('|'.join(exclude_phrases_lower), na=False)
    
    filtered_df = df[mask].copy()
    
    return filtered_df

def excel_classify(df):
    """Classify companies using external Excel data."""
    try:
        classify_df = pd.read_excel("data/company-list_cp.xls")
        classify_df = classify_df.drop(columns=['Days Attending', 'Reps'])
        classify_df = classify_df.rename(columns={
            'Company Name' : 'recommended company',
        })
        classify_df.columns = [col.strip().lower().replace(" ", "_") for col in classify_df.columns]
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        combined_df = pd.merge(df, classify_df, on='recommended_company', how='left')
        return combined_df
    except FileNotFoundError:
        print("Warning: company-list_cp.xls not found. Returning original DataFrame.")
        return df
    except Exception as e:
        print(f"Warning: Error in excel_classify: {e}. Returning original DataFrame.")
        return df

def new_classify(recommendation_df, interaction_matrix):
    """Enhanced classification with visualization."""
    try:
        from sklearn.decomposition import PCA
        import plotly.express as px
        
        combined_df = excel_classify(recommendation_df)
        
        # Add sector-based classification
        if 'sector' in combined_df.columns:
            # Sector distribution visualization
            if len(combined_df) >= 5:
                sector_pca_classify(combined_df, interaction_matrix)
        
        return combined_df
    except ImportError:
        print("Warning: Required libraries for visualization not available.")
        return excel_classify(recommendation_df)

def sector_pca_classify(combined_df, interaction_matrix):
    """PCA classification colored by sector."""
    try:
        from sklearn.decomposition import PCA
        import plotly.express as px
        
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
        
        # Get sectors for the companies in the subset
        company_sectors = combined_df.set_index(company_col)['sector'].to_dict()
        sectors = []
        for company in subset_matrix.index:
            sector = company_sectors.get(company, 'No Sector')
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
        
    except Exception as e:
        print(f"Error in sector PCA classification: {e}")

def sector_pca_visualize(X_reduced, sectors, labels, scores=None, score_col_name=None):
    """PCA scatter plot colored by sector."""
    try:
        import plotly.express as px
        
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
        
    except Exception as e:
        print(f"Error in sector PCA visualization: {e}")