import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from engines.recommender_helpers import convert_to_recommendations_df, visualize_results
from engines.source_company import SourceCompany

def recommend(df, source_company_id, method='multivector', threshold=0.4, top_n=10):
    """
    Unified recommendation function supporting multiple methods.
    
    Args:
        df: DataFrame with columns ['source_company', 'target_company']
        source_company_id: ID of the investor to generate recommendations for
        method: 'pairwise', 'multivector'
        threshold: Minimum correlation threshold for pairwise method
        top_n: Number of recommendations to return

    Recommends other correlated companies to meet based on previous behavior
    ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
    do you want to go ask your client if they might be interested?    
    """
    # creates the source company map containing id, requested, dates, and recommended
    grouped = df.groupby('source_company')
    source_company_map = {}
    for source_id, group in grouped:
        requested = group['target_company'].tolist()
        dates = set(group['request_date'])        
        source_company_map[source_id] = SourceCompany(source_id, requested, dates)

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
    
    # Remove duplicates to ensure we always get a Series
    unique_requested_companies = list(set(requested_companies))
    
    # Calculate correlation matrix + relevant correlations
    company_corr = interaction_matrix.corr(method='pearson')
    relevant_corrs = company_corr.loc[unique_requested_companies]
    
    # Aggregate recommendations across all requested companies
    recommendation_scores = {}
    
    for requested_company in unique_requested_companies:
        if requested_company not in relevant_corrs.index:
            continue
            
        correlations = relevant_corrs.loc[requested_company]
        
        # Now correlations will always be a Series
        valid_correlations = correlations[
            (correlations > threshold) & 
            (~correlations.index.isin(unique_requested_companies))
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

