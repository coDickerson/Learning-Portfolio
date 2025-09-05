#!/usr/bin/env python3
"""
Test script for evaluating recommender engine accuracy using scikit-learn.
This script demonstrates how to:
1. Split company data into training and validation sets
2. Test recommendation accuracy
3. Perform cross-validation
4. Compare different recommendation methods
"""

import pandas as pd
import numpy as np
from engines.model_evaluator import RecommenderEvaluator

def main():
    """Main function to test recommender engine accuracy."""
    print("=== RECOMMENDER ENGINE ACCURACY TESTING ===")
    
    # Load and process data (same as main.py)
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
    company_df = company_df[cols]  # flips source company and target company columns
    company_df = company_df.drop_duplicates()
    
    # Batch Mosaic Summit entries based on content after colon
    from engines.recommender_helpers import batch_mosaic_summit_entries
    company_df = batch_mosaic_summit_entries(company_df, 'target_company')
    
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
    
    # Find companies with sufficient data for evaluation (at least 3 interactions)
    company_counts = company_df['source_company'].value_counts()
    eligible_companies = company_counts[company_counts >= 3].index.tolist()
    
    print(f"\nFound {len(eligible_companies)} companies with sufficient data for evaluation (≥3 interactions)")
    
    if not eligible_companies:
        print("No companies have sufficient data for evaluation. Exiting.")
        return
    
    # Initialize the evaluator
    evaluator = RecommenderEvaluator(company_df, test_size=0.2, random_state=42)
    
    # Test single company evaluation
    print(f"\n=== TESTING SINGLE COMPANY EVALUATION ===")
    test_company = eligible_companies[0]
    print(f"Testing company ID: {test_company}")
    
    # Test multivector method
    result = evaluator.evaluate_recommendations(
        company_id=test_company,
        method='multivector',
        top_n=10
    )
    
    if result['success']:
        print(f"Evaluation successful!")
        print(f"Training set size: {result['train_size']}")
        print(f"Validation set size: {result['val_size']}")
        print(f"Precision: {result['metrics']['precision']:.3f}")
        print(f"Recall: {result['metrics']['recall']:.3f}")
        print(f"F1-Score: {result['metrics']['f1_score']:.3f}")
        print(f"Hit Rate: {result['metrics']['hit_rate']:.3f}")
        print(f"Hits: {result['metrics']['hits']}/{result['metrics']['total_val_companies']}")
    else:
        print(f"Evaluation failed: {result['error']}")
    
    # Test multiple companies
    print(f"\n=== TESTING MULTIPLE COMPANIES ===")
    # Use first 5 eligible companies for demonstration
    test_companies = eligible_companies[:5]
    print(f"Testing {len(test_companies)} companies: {test_companies}")
    
    # Test multivector method on multiple companies
    multivector_results = evaluator.evaluate_multiple_companies(
        company_ids=test_companies,
        method='multivector',
        top_n=10
    )
    
    # Test pairwise method on multiple companies
    pairwise_results = evaluator.evaluate_multiple_companies(
        company_ids=test_companies,
        method='pairwise',
        top_n=10,
        threshold=0.4
    )
    
    # Compare methods
    print(f"\n=== COMPARING RECOMMENDATION METHODS ===")
    comparison = evaluator.compare_methods(
        company_ids=test_companies,
        top_n=10,
        threshold=0.4
    )
    
    # Perform cross-validation
    print(f"\n=== PERFORMING CROSS-VALIDATION ===")
    cv_results = evaluator.cross_validate_recommendations(
        company_ids=test_companies,
        method='multivector',
        top_n=10,
        cv_folds=3  # Using 3 folds for demonstration
    )
    
    # Save results to CSV
    print(f"\n=== SAVING RESULTS ===")
    
    # Save detailed results
    multivector_results.to_csv('multivector_evaluation_results.csv', index=False)
    pairwise_results.to_csv('pairwise_evaluation_results.csv', index=False)
    comparison.to_csv('method_comparison_results.csv', index=False)
    
    print("Results saved to CSV files:")
    print("- multivector_evaluation_results.csv")
    print("- pairwise_evaluation_results.csv") 
    print("- method_comparison_results.csv")
    
    # Display summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    
    # Multivector summary
    successful_multivector = multivector_results[multivector_results['success'] == True]
    if not successful_multivector.empty:
        metrics_df = pd.DataFrame(successful_multivector['metrics'].tolist())
        print(f"Multivector Method:")
        print(f"  Successful evaluations: {len(successful_multivector)}/{len(test_companies)}")
        print(f"  Average Precision: {metrics_df['precision'].mean():.3f}")
        print(f"  Average Recall: {metrics_df['recall'].mean():.3f}")
        print(f"  Average F1-Score: {metrics_df['f1_score'].mean():.3f}")
        print(f"  Average Hit Rate: {metrics_df['hit_rate'].mean():.3f}")
    
    # Pairwise summary
    successful_pairwise = pairwise_results[pairwise_results['success'] == True]
    if not successful_pairwise.empty:
        metrics_df = pd.DataFrame(successful_pairwise['metrics'].tolist())
        print(f"\nPairwise Method:")
        print(f"  Successful evaluations: {len(successful_pairwise)}/{len(test_companies)}")
        print(f"  Average Precision: {metrics_df['precision'].mean():.3f}")
        print(f"  Average Recall: {metrics_df['recall'].mean():.3f}")
        print(f"  Average F1-Score: {metrics_df['f1_score'].mean():.3f}")
        print(f"  Average Hit Rate: {metrics_df['hit_rate'].mean():.3f}")
    
    print(f"\n=== EVALUATION COMPLETE ===")
    print("The recommender engine has been tested using scikit-learn evaluation methods.")
    print("Results show how well the engine can predict which companies a source company")
    print("would request to meet with, based on their historical behavior patterns.")

if __name__ == "__main__":
    main()

