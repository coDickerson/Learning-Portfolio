import pandas as pd
from classificationcluster import classify

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
        'hybrid': f"\nHybrid recommendations combining multiple approaches:"
    }
    
    message = method_messages.get(method.lower(), f"\n{method.title()} recommendations:")
    print(message)
    print(recommendations_df.head(top_n))

def get_score_column_name(method) :
    score_columns = {
        'pairwise' : 'correlation',
        'multivector': 'similarity',
        'hybrid': 'hybrid_score'
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
                classify(recommendations_df, interaction_matrix)
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