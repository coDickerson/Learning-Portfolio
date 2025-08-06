import pandas as pd
from classificationcluster import new_classify, excel_classify

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


# Backward compatibility functions for existing code
# def pairwise(company_corr, requested_companies, threshold):
#     """Legacy function - replaced by unified recommend() function"""
#     print("Note: pairwise() function is deprecated. Use recommend() with method='pairwise'")
#     return pd.DataFrame()

# def multivector(interaction_matrix, source_company_id, requested_companies, top_n=10):
#     """Legacy function - replaced by unified recommend() function"""  
#     print("Note: multivector() function is deprecated. Use recommend() with method='multivector'")
#     return pd.DataFrame()

# def visualize(interaction_matrix, recommendations_df):
#     """Legacy function - now handled within recommend() function"""
#     print("Note: visualize() function is deprecated. Visualization is now handled automatically.")
#     return recommendations_df