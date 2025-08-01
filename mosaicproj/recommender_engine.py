# import pandas as pd
# from sklearn.cluster import KMeans 
# from sklearn.metrics.pairwise import cosine_similarity
# from classificationcluster import classify

# # recommend other correlated companies to meet based on previous behavior
# # ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
# #     do you want to go ask your client if they might be interested?

# def recommend(df, source_company_id, source_company_map, method='multivector', threshold=.4, top_n=10):
#     # binary interaction matrix
#     # describes if a target company was requested by a source, 1=yes and 0=no
#     interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])

#     # checks if company id exists
#     if source_company_id not in interaction_matrix.index:
#         print(f'Investor {source_company_id} not found.')
#         return pd.DataFrame()

#     # source company row from map and list of requested companies
#     source_company_row = source_company_map[source_company_id]
#     requested_companies = source_company_row.requested

#     if method == 'pairwise':
#         recommendations_df = pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n)
#     elif method == 'multivector':
#         recommendations_df = multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n)
#     else:
#         raise ValueError("Method must be 'pairwise' or 'multivector'")

#     visualize(recommendations_df, interaction_matrix, method)

#     return recommendations_df
    
    
# def pairwise_recommend(company_corr, requested_companies, threshold):
#     """
#     Optimized pairwise correlation method.
#     Finds companies correlated with requested companies.
#     """
#     if not requested_companies:
#         print("No requested companies to base recommendations on.")
#         return pd.DataFrame()
    
#     # Calculate correlation matrix more efficiently
#     company_corr = interaction_matrix.corr(method='pearson')
    
#     # Get correlations for requested companies only (more efficient)
#     relevant_corrs = company_corr.loc[requested_companies]
    
#     # Aggregate recommendations across all requested companies
#     recommendation_scores = {}
    
#     for requested_company in requested_companies:
#         if requested_company not in relevant_corrs.index:
#             continue
            
#         # Get correlations for this requested company
#         correlations = relevant_corrs.loc[requested_company]
        
#         # Filter by threshold and exclude already requested companies
#         valid_correlations = correlations[
#             (correlations > threshold) & 
#             (~correlations.index.isin(requested_companies))
#         ]
        
#         # Aggregate scores (taking maximum correlation seen)
#         for company, corr in valid_correlations.items():
#             if company not in recommendation_scores:
#                 recommendation_scores[company] = corr
#             else:
#                 recommendation_scores[company] = max(recommendation_scores[company], corr)
    
#     # Convert to DataFrame
#     if not recommendation_scores:
#         print("No significant correlations found above threshold.")
#         return pd.DataFrame(columns=['company', 'correlation'])
    
#     recommendations_df = pd.DataFrame(
#         list(recommendation_scores.items()),
#         columns=['company', 'correlation']
#     ).sort_values('correlation', ascending=False).head(top_n)
    
#     print(f"\nPairwise recommendations based on correlations > {threshold}:\n")
#     print(recommendations_df)
    


#     # flattens correlation matrix into a workable table
#     # filters out correlation below threshold and finds pairs with one company from requsted companies list
#     company_corr.columns.name = None
#     company_corr.index.name = None
#     company_pairs = company_corr.stack().reset_index() # flattens square matrix
#     company_pairs.columns = ['company_a', 'company_b', 'correlation']
#     company_pairs = company_pairs[company_pairs['company_a'] < company_pairs['company_b']] # deals with duplicates and same company correlations
#     significant_pairs = company_pairs[company_pairs['correlation'] > threshold] 
#     filtered = significant_pairs [
#     ((significant_pairs['company_a'].isin(requested_companies)) & (~significant_pairs['company_b'].isin(requested_companies))) | 
#     ((significant_pairs['company_b'].isin(requested_companies)) & (~significant_pairs['company_a'].isin(requested_companies)))
#     ]
#     filtered = filtered.sort_values(by='correlation', ascending=False)
#     # adds other company and correlation to suggestions list
#     suggestions = []
#     for _, row in filtered.iterrows():
#         if row['company_a'] in requested_companies and row['company_b'] not in requested_companies:
#             suggestions.append((row['company_b'], row['correlation']))
#         elif row['company_b'] in requested_companies and row['company_a'] not in requested_companies:
#             suggestions.append((row['company_a'], row['correlation']))
#     # adds suggestions to data frame to be printed
#     recommendations_df = pd.DataFrame(suggestions, columns=['recommended company', 'correlation'])
#     recommendations_df = recommendations_df.drop_duplicates(subset='recommended company')
#     recommendations_df = recommendations_df.sort_values(by='correlation', ascending=False)
#     return recommendations_df

    
# def multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n = 10):
#     """
#     Collaborative filtering using cosine similarity between investor profiles.
#     """
#     # Get source investor's profile
#     source_profile = interaction_matrix.loc[source_company_id].values.reshape(1, -1)
    
#     # Calculate similarity with all other investors
#     similarities = cosine_similarity(source_profile, interaction_matrix.values).flatten()
    
#     # Create investor similarity mapping
#     investor_similarities = pd.Series(similarities, index=interaction_matrix.index)
#     investor_similarities = investor_similarities[investor_similarities.index != source_company_id]
    
#     # Calculate weighted recommendation scores
#     company_scores = {}
    
#     for company in interaction_matrix.columns:
#         if company in requested_companies:
#             continue
            
#         # Weighted score based on similar investors who requested this company
#         weighted_score = 0
#         total_weight = 0
        
#         for investor, similarity in investor_similarities.items():
#             if interaction_matrix.loc[investor, company] == 1:  # If investor requested this company
#                 weighted_score += similarity
#                 total_weight += similarity
        
#         # Normalize score
#         if total_weight > 0:
#             company_scores[company] = weighted_score / total_weight
    
#     # Convert to DataFrame
#     if not company_scores:
#         print("No recommendations found based on similar investors.")
#         return pd.DataFrame(columns=['company', 'similarity'])
    
#     recommendations_df = pd.DataFrame(
#         list(company_scores.items()),
#         columns=['company', 'similarity']
#     ).sort_values('similarity', ascending=False).head(top_n)
    
#     print("\nMultivector recommendations based on similar investor preferences:\n")
#     print(recommendations_df)
    
#     return recommendations_df







    
#     source_profile = interaction_matrix.loc[source_company_id]
#     investor_similarities = cosine_similarity(source_profile.values.reshape(1, -1), interaction_matrix.values).flatten()

#     investor_sim_df = pd.DataFrame({
#         'investor': interaction_matrix.index,
#         'similarity': investor_similarities
#     })

#     investor_sim_df = investor_sim_df[investor_sim_df['investor'] != source_company_id]
#     similar_investors = investor_sim_df.sort_values('similarity', ascending=False)
#     company_scores = {}
    
#     for company in interaction_matrix.columns:
#         if company in requested_companies:
#             continue  # Skip already requested companies
            
#         weighted_score = 0
#         total_weight = 0
        
#         for _, row in similar_investors.iterrows():
#             investor = row['investor']
#             similarity = row['similarity']
            
#             # Check if this similar investor requested this company
#             if interaction_matrix.loc[investor, company] == 1:
#                 weighted_score += similarity
#                 total_weight += similarity
        
#         # Normalize by total weight to get average weighted interest
#         if total_weight > 0:
#             company_scores[company] = weighted_score / total_weight
#         else:
#             company_scores[company] = 0
    
#     # Convert to DataFrame and sort
#     recommendations_df = pd.DataFrame(
#         list(company_scores.items()), 
#         columns=['company', 'recommendation_score']
#     )
    
#     recommendations_df = recommendations_df.sort_values(
#         'recommendation_score', 
#         ascending=False
#     ).head(top_n)
    
#     visualize(interaction_matrix, recommendations_df)
#     return recommendations_df




# def visualize(interaction_matrix, recommendations_df):   
#     print("\nHere are a list of similar companies that are based on your previous company requests:\n")
#     print(recommendations_df.head(10))
#     while(True) :
#         user_input = input('Would you like a visualization (y/n):\n').strip().lower()
#         if user_input == 'y':
#             classify(recommendations_df, interaction_matrix)
#             break
#         elif user_input == 'n':
#             print("Visualization skipped.")
#             break
#         else:
#             print("Invalid input. Please enter 'y' or 'n'.")
#     return (recommendations_df)


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend(df, source_company_id, source_company_map, method='multivector', threshold=0.4, top_n=10):
    """
    Unified recommendation function supporting multiple methods.
    
    Args:
        df: DataFrame with columns ['source_company', 'target_company']
        source_company_id: ID of the investor to generate recommendations for
        source_company_map: Mapping of company IDs to their data
        method: 'pairwise', 'multivector', or 'hybrid'
        threshold: Minimum correlation threshold for pairwise method
        top_n: Number of recommendations to return
    """
    # Create interaction matrix once
    interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])
    
    # Validate input
    if source_company_id not in interaction_matrix.index:
        print(f'Investor {source_company_id} not found.')
        return pd.DataFrame()
    
    # Get requested companies
    source_company_row = source_company_map[source_company_id]
    requested_companies = source_company_row.requested
    
    print(f"\nGenerating recommendations for investor {source_company_id}")
    print(f"Already requested: {len(requested_companies)} companies")
    
    # Generate recommendations based on method
    if method == 'pairwise':
        recommendations_df = _pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n)
    elif method == 'multivector':
        recommendations_df = _multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n)
    elif method == 'hybrid':
        recommendations_df = _hybrid_recommend(interaction_matrix, source_company_id, requested_companies, threshold, top_n)
    else:
        raise ValueError("Method must be 'pairwise', 'multivector', or 'hybrid'")
    
    # Import and call visualization - compatible with your existing classify function
    if not recommendations_df.empty:
        _visualize_results(recommendations_df, interaction_matrix, method)
    
    return recommendations_df


def _pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n):
    """
    Optimized pairwise correlation method.
    """
    if not requested_companies:
        print("No requested companies to base recommendations on.")
        return pd.DataFrame()
    
    # Calculate correlation matrix
    company_corr = interaction_matrix.corr(method='pearson')
    
    # Get correlations for requested companies only
    relevant_corrs = company_corr.loc[requested_companies]
    
    # Aggregate recommendations across all requested companies
    recommendation_scores = {}
    
    for requested_company in requested_companies:
        if requested_company not in relevant_corrs.index:
            continue
            
        # Get correlations for this requested company
        correlations = relevant_corrs.loc[requested_company]
        
        # Filter by threshold and exclude already requested companies
        valid_correlations = correlations[
            (correlations > threshold) & 
            (~correlations.index.isin(requested_companies))
        ]
        
        # Aggregate scores (taking maximum correlation seen)
        for company, corr in valid_correlations.items():
            if company not in recommendation_scores:
                recommendation_scores[company] = corr
            else:
                recommendation_scores[company] = max(recommendation_scores[company], corr)
    
    # Convert to DataFrame - using column name that matches your classify function
    if not recommendation_scores:
        print("No significant correlations found above threshold.")
        return pd.DataFrame(columns=['recommended company', 'correlation'])
    
    recommendations_df = pd.DataFrame(
        list(recommendation_scores.items()),
        columns=['recommended company', 'correlation']  # Match your classify function expectation
    ).sort_values('correlation', ascending=False).head(top_n)
    
    print(f"\nPairwise recommendations based on correlations > {threshold}:")
    print(recommendations_df)
    
    return recommendations_df


def _multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n):
    """
    Collaborative filtering using cosine similarity between investor profiles.
    """
    # Get source investor's profile
    source_profile = interaction_matrix.loc[source_company_id].values.reshape(1, -1)
    
    # Calculate similarity with all other investors
    similarities = cosine_similarity(source_profile, interaction_matrix.values).flatten()
    
    # Create investor similarity mapping
    investor_similarities = pd.Series(similarities, index=interaction_matrix.index)
    investor_similarities = investor_similarities[investor_similarities.index != source_company_id]
    
    # Sort by similarity and get top similar investors for debugging
    top_similar = investor_similarities.sort_values(ascending=False).head(5)
    print(f"\nTop 5 similar investors:")
    for investor, sim in top_similar.items():
        print(f"  Investor {investor}: {sim:.3f} similarity")
    
    # Calculate weighted recommendation scores
    company_scores = {}
    
    for company in interaction_matrix.columns:
        if company in requested_companies:
            continue
            
        # Weighted score based on similar investors who requested this company
        weighted_score = 0
        total_weight = 0
        
        for investor, similarity in investor_similarities.items():
            if interaction_matrix.loc[investor, company] == 1:  # If investor requested this company
                weighted_score += similarity
                total_weight += similarity
        
        # Normalize score
        if total_weight > 0:
            company_scores[company] = weighted_score / total_weight
    
    # Convert to DataFrame - using column name that matches your classify function
    if not company_scores:
        print("No recommendations found based on similar investors.")
        return pd.DataFrame(columns=['recommended company', 'similarity'])
    
    recommendations_df = pd.DataFrame(
        list(company_scores.items()),
        columns=['recommended company', 'similarity']  # Match your classify function expectation
    ).sort_values('similarity', ascending=False).head(top_n)
    
    print("\nMultivector recommendations based on similar investor preferences:")
    print(recommendations_df)
    
    return recommendations_df


def _hybrid_recommend(interaction_matrix, source_company_id, requested_companies, threshold, top_n):
    """
    Hybrid approach combining pairwise and multivector methods.
    """
    # Get both types of recommendations
    pairwise_recs = _pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n * 2)
    multivector_recs = _multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n * 2)
    
    if pairwise_recs.empty and multivector_recs.empty:
        return pd.DataFrame(columns=['recommended company', 'hybrid_score'])
    
    # Normalize scores to 0-1 range for combining
    hybrid_scores = {}
    
    # Add pairwise recommendations with weight (60%)
    if not pairwise_recs.empty:
        max_corr = pairwise_recs['correlation'].max()
        for _, row in pairwise_recs.iterrows():
            company = row['recommended company']
            normalized_score = row['correlation'] / max_corr
            hybrid_scores[company] = hybrid_scores.get(company, 0) + (normalized_score * 0.6)
    
    # Add multivector recommendations with weight (40%)
    if not multivector_recs.empty:
        max_sim = multivector_recs['similarity'].max()
        for _, row in multivector_recs.iterrows():
            company = row['recommended company']
            normalized_score = row['similarity'] / max_sim
            hybrid_scores[company] = hybrid_scores.get(company, 0) + (normalized_score * 0.4)
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(
        list(hybrid_scores.items()),
        columns=['recommended company', 'hybrid_score']  # Match your classify function expectation
    ).sort_values('hybrid_score', ascending=False).head(top_n)
    
    print("\nHybrid recommendations combining pairwise and multivector approaches:")
    print(recommendations_df)
    
    return recommendations_df


def _visualize_results(recommendations_df, interaction_matrix, method):
    """
    Visualization function compatible with your existing classify function.
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
                # Import your classify function and call it
                from classificationcluster import classify
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


# Backward compatibility functions for your existing code
def pairwise(company_corr, requested_companies, threshold):
    """Legacy function - replaced by unified recommend() function"""
    print("Note: pairwise() function is deprecated. Use recommend() with method='pairwise'")
    return pd.DataFrame()

def multivector(interaction_matrix, source_company_id, requested_companies, top_n=10):
    """Legacy function - replaced by unified recommend() function"""  
    print("Note: multivector() function is deprecated. Use recommend() with method='multivector'")
    return pd.DataFrame()

def visualize(interaction_matrix, recommendations_df):
    """Legacy function - now handled within recommend() function"""
    print("Note: visualize() function is deprecated. Visualization is now handled automatically.")
    return recommendations_df