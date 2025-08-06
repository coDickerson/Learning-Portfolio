import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommender_helpers import convert_to_recommendations_df, visualize_results

def recommend(df, source_company_id, source_company_map, method='multivector', threshold=0.4, top_n=10):
    """
    Unified recommendation function supporting multiple methods.
    
    Args:
        df: DataFrame with columns ['source_company', 'target_company']
        source_company_id: ID of the investor to generate recommendations for
        source_company_map: Mapping of company IDs to their data
        method: 'pairwise', 'multivector'
        threshold: Minimum correlation threshold for pairwise method
        top_n: Number of recommendations to return

    Recommends other correlated companies to meet based on previous behavior
    ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
    do you want to go ask your client if they might be interested?    
    """

    # Create interaction matrix
    interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])
    
    # Validate input
    if source_company_id not in interaction_matrix.index:
        print(f'Investor {source_company_id} not found.')
        return pd.DataFrame()
    
    # Get requested companies
    source_company_row = source_company_map[source_company_id]
    requested_companies = source_company_row.requested
    
    print(f"\nGenerating recommendations for investor {source_company_id}")
    
    # Generate recommendations based on method
    if method == 'pairwise':
        recommendations_df = pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n)
    elif method == 'multivector':
        recommendations_df = multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n)
    else:
        raise ValueError("Method must be 'pairwise' or 'multivector'")
    
    # Call visualization
    if not recommendations_df.empty:
        visualize_results(recommendations_df, interaction_matrix, method)
    
    return recommendations_df


def pairwise_recommend(interaction_matrix, requested_companies, threshold, top_n):
    """
    Optimized pairwise correlation method.
    """
    if not requested_companies:
        print("No requested companies to base recommendations on.")
        return pd.DataFrame()
    
    # Calculate correlation matrix + relevent correlations
    company_corr = interaction_matrix.corr(method='pearson')
    relevant_corrs = company_corr.loc[requested_companies]
    
    # Aggregate recommendations across all requested companies
    recommendation_scores = {}
    
    for requested_company in requested_companies:
        if requested_company not in relevant_corrs.index:
            continue
            
        correlations = relevant_corrs.loc[requested_company]
        
        # Check if correlations is a Series or DataFrame (handles duplicate companies)
        if isinstance(correlations, pd.DataFrame):
            # If it's a DataFrame, we need to handle it differently
            for col in correlations.columns:
                if col not in requested_companies:
                    corr_value = correlations[col].iloc[0] if len(correlations) > 0 else 0
                    if corr_value > threshold:
                        if col not in recommendation_scores:
                            recommendation_scores[col] = corr_value
                        else:
                            recommendation_scores[col] = max(recommendation_scores[col], corr_value)
        else:
            # Original logic for Series
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
        
    return convert_to_recommendations_df(recommendation_scores, 'pairwise', top_n)


def multivector_recommend(interaction_matrix, source_company_id, requested_companies, top_n):
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
    
    # Sort by similarity and get top similar investors 
    # top_similar = investor_similarities.sort_values(ascending=False).head(5)
    # print(f"\nTop 5 similar investors:")
    # for investor, sim in top_similar.items():
    #     print(f"  Investor {investor}: {sim:.3f} similarity")
    
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

    return convert_to_recommendations_df(company_scores, 'multivector', top_n)

