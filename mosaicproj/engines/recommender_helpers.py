import pandas as pd

def convert_to_recommendations_df(scores, method, top_n):
    """
    Convert a dictionary of company scores to a standardized recommendations DataFrame.

    Args:
        scores (dict): Dictionary with company names as keys and scores as values
        method (str): The recommendation method ('pairwise', 'multivector', 'hybrid')
        top_n (int): Number of top recommendations to return
    
    Returns:
        pd.DataFrame: Formatted recommendations with company names as index
    """
    score_column = get_score_column_name(method)

    if not scores:
        # Return empty DataFrame with appropriate columns based on method
        return pd.DataFrame(columns=[score_column])
    
    # Convert to DataFrame and set company names as index
    recommendations_df = pd.DataFrame(
        list(scores.items()),
        columns=['recommended company', score_column]
    ).sort_values(score_column, ascending=False)
    
    # Set company names as index
    recommendations_df = recommendations_df.set_index('recommended company')
    
    # Apply classification if possible
    try:
        # Reset index for classification
        temp_df = recommendations_df.reset_index()
        temp_df = excel_classify(temp_df)
        # Re-set index after classification
        if 'recommended company' in temp_df.columns:
            recommendations_df = temp_df.set_index('recommended company')
        else:
            # If classification changed column names, try to find the right column
            for col in temp_df.columns:
                if 'company' in col.lower():
                    recommendations_df = temp_df.set_index(col)
                    break
            else:
                # Fallback: use first column as index
                recommendations_df = temp_df.set_index(temp_df.columns[0])
    except Exception as e:
        print(f"Warning: Classification failed: {e}")
        # Ensure we have company names as index
        if 'recommended company' in recommendations_df.columns:
            recommendations_df = recommendations_df.set_index('recommended company')
    
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
    # Escape regex special characters to avoid warnings
    import re
    escaped_phrases = [re.escape(phrase) for phrase in exclude_phrases_lower]
    mask = ~df[column_name].astype(str).str.lower().str.contains('|'.join(escaped_phrases), na=False)
    
    filtered_df = df[mask].copy()
    
    return filtered_df

def load_kbcm_coverage():
    """Load KBCM coverage data from text file."""
    coverage_data = {}
    try:
        with open("data/kbcm_coverage.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line == "KBCM Coverage":
                    continue
                    
                # Check if this is an analyst line (contains parentheses and colon)
                if '(' in line and ')' in line and ':' in line:
                    # Extract analyst name and coverage type
                    analyst_part = line.split('(')[0].strip()
                    coverage_type_part = line.split('(')[1].split(')')[0]
                    
                    # Extract tickers (everything after the colon)
                    ticker_part = line.split('):')[1].strip()
                    tickers = [ticker.strip() for ticker in ticker_part.split(',')]
                    
                    # Store coverage information
                    for ticker in tickers:
                        if ticker:  # Skip empty tickers
                            coverage_data[ticker] = {
                                'analyst': analyst_part,
                                'sector': coverage_type_part,
                                'coverage_type': coverage_type_part
                            }
                            
    except FileNotFoundError:
        print("Warning: kbcm_coverage.txt not found.")
    except Exception as e:
        print(f"Warning: Error loading KBCM coverage: {e}")
    
    return coverage_data

def classify_company_status(df):
    """Classify companies as public/private and add coverage information."""
    try:
        # Load KBCM coverage data
        coverage_data = load_kbcm_coverage()
        
        # Add company status classification
        df['company_status'] = 'Private'  # Default to private
        
        # Check if company has a ticker
        if 'ticker' in df.columns:
            # Companies with tickers are considered public
            df.loc[df['ticker'].notna() & (df['ticker'] != ''), 'company_status'] = 'Public'
            
            # Add coverage information for public companies
            df['analyst'] = None
            df['coverage_sector'] = None
            df['coverage_type'] = None
            
            for idx, row in df.iterrows():
                ticker = row['ticker']
                if pd.notna(ticker) and ticker in coverage_data:
                    df.at[idx, 'analyst'] = coverage_data[ticker]['analyst']
                    df.at[idx, 'coverage_sector'] = coverage_data[ticker]['sector']
                    df.at[idx, 'coverage_type'] = coverage_data[ticker]['coverage_type']
        
        # Create enhanced description
        df['enhanced_description'] = df.apply(lambda row: create_enhanced_description(row), axis=1)
        
        return df
        
    except Exception as e:
        print(f"Warning: Error in company status classification: {e}")
        return df

def create_enhanced_description(row):
    """Create enhanced description based on company status and coverage."""
    company_name = row.get('recommended_company', 'Unknown Company')
    ticker = row.get('ticker', '')
    sector = row.get('sector', 'Unknown Sector')
    status = row.get('company_status', 'Unknown')
    
    if status == 'Public':
        analyst = row.get('analyst', '')
        coverage_sector = row.get('coverage_sector', '')
        coverage_type = row.get('coverage_type', '')
        
        if analyst and coverage_sector:
            return f"{company_name} ({ticker}) - {sector} | Public | Covered by {analyst} ({coverage_sector}: {coverage_type})"
        else:
            return f"{company_name} ({ticker}) - {sector} | Public | No KBCM coverage"
    else:
        return f"{company_name} ({ticker}) - {sector} | Private"

def excel_classify(df):
    """Classify companies using external Excel data with public/private classification."""
    try:
        classify_df = pd.read_excel("data/company-list_cp.xls")
        classify_df = classify_df.drop(columns=['Days Attending', 'Reps'])
        classify_df = classify_df.rename(columns={
            'Company Name' : 'recommended company',
        })
        classify_df.columns = [col.strip().lower().replace(" ", "_") for col in classify_df.columns]
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        combined_df = pd.merge(df, classify_df, on='recommended_company', how='left')
        
        # Add company status classification
        combined_df = classify_company_status(combined_df)
        
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

def batch_mosaic_summit_entries(df, column_name='target_company'):
    """
    Batch Mosaic Summit entries based on the topic/subject after the colon and before the dash.
    
    This groups related sessions together by extracting the core topic:
    - "Mosaic Summit 1x1: AI, ML & Advanced Data Science - Assessing the GenAI Data Stack"
    - "Mosaic Summit Small Group: AI, ML & Advanced Data Science - GenAI Use Cases in the Enterprise"
    
    Will both be converted to: "Mosaic Summit: AI, ML & Advanced Data Science"
    
    Args:
        df: DataFrame containing the data
        column_name: Column containing company names to process
    
    Returns:
        DataFrame with batched Mosaic Summit entries
    """
    df_copy = df.copy()
    
    # Function to extract and standardize Mosaic Summit entries
    def standardize_mosaic_summit(company_name):
        if pd.isna(company_name):
            return company_name
            
        company_str = str(company_name).strip()
        
        # Check if it's a Mosaic Summit entry
        if 'mosaic summit' in company_str.lower():
            # Remove "(Optional)" prefix if present
            if company_str.lower().startswith('(optional)'):
                company_str = company_str[10:].strip()
            
            # Find the colon position
            colon_pos = company_str.find(':')
            if colon_pos != -1:
                # Extract everything after the colon
                after_colon = company_str[colon_pos + 1:].strip()
                
                # Find the dash position to extract just the topic
                dash_pos = after_colon.find(' - ')
                if dash_pos != -1:
                    # Extract topic (everything before the dash)
                    topic = after_colon[:dash_pos].strip()
                    return f"Mosaic Summit: {topic}"
                else:
                    # No dash found, use everything after colon
                    return f"Mosaic Summit: {after_colon}"
            else:
                # No colon found, handle special cases
                if 'mosaic summit thematic dinner' in company_str.lower():
                    return "Mosaic Summit: Thematic Dinner"
                elif 'mosaic summit birds of a feather' in company_str.lower():
                    return "Mosaic Summit: Birds of a Feather"
                else:
                    # Try to extract meaningful content after "Mosaic Summit"
                    mosaic_pos = company_str.lower().find('mosaic summit') + len('mosaic summit')
                    after_mosaic = company_str[mosaic_pos:].strip()
                    if after_mosaic.startswith(':') or after_mosaic.startswith(' -'):
                        after_mosaic = after_mosaic[1:].strip()
                    if after_mosaic:
                        return f"Mosaic Summit: {after_mosaic}"
                    else:
                        return "Mosaic Summit: General"
        
        return company_name
    
    # Apply the standardization
    df_copy[column_name] = df_copy[column_name].apply(standardize_mosaic_summit)
    
    return df_copy