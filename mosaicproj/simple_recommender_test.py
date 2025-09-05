#!/usr/bin/env python3
"""
Simple recommender engine accuracy test.
This script tests if the validation set companies are contained within 
the top 10 suggested companies from the recommender engine.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from engines.recommender_engine import recommend

def test_recommender_accuracy(company_df, company_id, method='multivector', top_n=10):
    """
    Test recommender accuracy for a specific company.
    
    Args:
        company_df: DataFrame with company interaction data
        company_id: ID of the company to test
        method: Recommendation method ('pairwise' or 'multivector')
        top_n: Number of top recommendations to consider
        
    Returns:
        Dictionary with test results
    """
    print(f"\n=== Testing Company {company_id} ===")
    
    # Get all interactions for this company
    company_data = company_df[company_df['source_company'] == company_id].copy()
    
    if len(company_data) < 2:
        print(f"Company {company_id} has insufficient data (need at least 2 interactions)")
        return None
    
    print(f"Total interactions for company {company_id}: {len(company_data)}")
    
    # Split into training and validation sets (80% train, 20% validation)
    train_data, val_data = train_test_split(
        company_data, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Get validation set companies
    val_companies = set(val_data['target_company'].unique())
    print(f"Validation companies: {list(val_companies)}")
    
    # Create temporary dataset with only training data for this company
    temp_df = company_df[company_df['source_company'] != company_id].copy()
    temp_df = pd.concat([temp_df, train_data])
    
    # Generate recommendations using training data
    print(f"\nGenerating recommendations using {method} method...")
    recommendations = recommend(
        temp_df, 
        company_id, 
        method=method, 
        threshold=0.4, 
        top_n=top_n
    )
    
    if recommendations.empty:
        print("No recommendations generated!")
        return None
    
    # Get top N recommended companies
    recommended_companies = set(recommendations.head(top_n).index.tolist())
    print(f"Top {top_n} recommended companies: {list(recommended_companies)}")
    
    # Check if validation companies are in recommendations
    hits = val_companies.intersection(recommended_companies)
    misses = val_companies - recommended_companies
    
    print(f"\n=== RESULTS ===")
    print(f"Hits (validation companies found in recommendations): {list(hits)}")
    print(f"Misses (validation companies NOT found): {list(misses)}")
    print(f"Hit rate: {len(hits)}/{len(val_companies)} = {len(hits)/len(val_companies):.2%}")
    
    # Determine if this is a "good" recommendation
    is_good = len(hits) > 0
    result = "GOOD" if is_good else "INVALID"
    print(f"Recommendation quality: {result}")
    
    return {
        'company_id': company_id,
        'method': method,
        'is_good': is_good,
        'hits': list(hits),
        'misses': list(misses),
        'hit_rate': len(hits) / len(val_companies),
        'total_val_companies': len(val_companies),
        'total_recommendations': len(recommended_companies),
        'train_size': len(train_data),
        'val_size': len(val_data)
    }

def main():
    """Main function to test recommender accuracy."""
    print("=== SIMPLE RECOMMENDER ACCURACY TEST ===")
    
    # Load and process data
    print("\nLoading and processing data...")
    company_df = pd.read_excel("data/Conf 2024 Request List Update.xlsx")
    company_df = company_df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
    company_df = company_df.rename(columns = 
        {  
            'Target Company - who was requested to meet':'target_company',
            'Source Company - who made the request':'source_company', 
            'Request Date Created' : 'request_date'
        }
    )
    company_df.columns = [col.strip().lower().replace(" ", "_") for col in company_df.columns]
    cols = company_df.columns.to_list()
    cols[0], cols[1] = cols[1], cols[0]
    company_df = company_df[cols]
    company_df = company_df.drop_duplicates()
    
    # Batch Mosaic Summit entries
    from engines.recommender_helpers import batch_mosaic_summit_entries
    company_df = batch_mosaic_summit_entries(company_df, 'target_company')
    
    # Filter out excluded phrases
    excluded_phrases = [
        'Test Company',
        'Floor Monitor',
        'KeyBank',
        'KeyBanc Capital',
        'Mosaic Summit: Thematic Dinner',
        '(Optional)'
    ]
    
    from engines.recommender_helpers import filter_companies_by_phrases
    company_df = filter_companies_by_phrases(company_df, excluded_phrases, 'target_company')
    
    print(f"Data loaded: {len(company_df)} records, {company_df['source_company'].nunique()} unique investors")
    
    # Find companies with sufficient data (at least 3 interactions)
    company_counts = company_df['source_company'].value_counts()
    eligible_companies = company_counts[company_counts >= 3].index.tolist()
    
    print(f"\nFound {len(eligible_companies)} companies with sufficient data for testing (≥3 interactions)")
    
    if not eligible_companies:
        print("No companies have sufficient data for testing. Exiting.")
        return
    
    # Test a few companies
    test_companies = eligible_companies[:3]  # Test first 3 companies
    print(f"\nTesting companies: {test_companies}")
    
    all_results = []
    
    # Test each company with both methods
    for company_id in test_companies:
        print(f"\n{'='*50}")
        
        # Test multivector method
        result_multivector = test_recommender_accuracy(
            company_df, company_id, method='multivector', top_n=10
        )
        
        if result_multivector:
            all_results.append(result_multivector)
        
        print(f"\n{'-'*30}")
        
        # Test pairwise method
        result_pairwise = test_recommender_accuracy(
            company_df, company_id, method='pairwise', top_n=10
        )
        
        if result_pairwise:
            all_results.append(result_pairwise)
    
    # Summary
    print(f"\n{'='*50}")
    print("=== FINAL SUMMARY ===")
    
    if all_results:
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(all_results)
        
        # Group by method
        for method in ['multivector', 'pairwise']:
            method_results = results_df[results_df['method'] == method]
            if not method_results.empty:
                good_count = method_results['is_good'].sum()
                total_count = len(method_results)
                avg_hit_rate = method_results['hit_rate'].mean()
                
                print(f"\n{method.upper()} Method:")
                print(f"  Good recommendations: {good_count}/{total_count} ({good_count/total_count:.1%})")
                print(f"  Average hit rate: {avg_hit_rate:.2%}")
        
        # Save results
        results_df.to_csv('simple_recommender_test_results.csv', index=False)
        print(f"\nDetailed results saved to: simple_recommender_test_results.csv")
    
    print(f"\n=== TEST COMPLETE ===")
    print("The recommender engine has been tested to see if validation set companies")
    print("are contained within the top 10 suggested companies.")
    print("Results show whether each recommendation is 'GOOD' (contains validation companies)")
    print("or 'INVALID' (does not contain any validation companies).")

if __name__ == "__main__":
    main()

