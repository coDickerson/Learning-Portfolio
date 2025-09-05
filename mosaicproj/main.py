import numpy as np
import pandas as pd
import traceback
from engines.recommender_engine import recommend
from engines.recommender_helpers import filter_companies_by_phrases, batch_mosaic_summit_entries
from engines.cohort_analysis import CohortAnalyzer

def run_recommendation_system(company_df):
    """Run the company recommendation system."""
    print("\n=== COMPANY RECOMMENDATION SYSTEM ===")
    
    # obtains input from user
    user_input = input("\nEnter source company ID (e.g. 1000-1342): ").strip()
    print("Available recommendation methods:")
    print("1. Pairwise (correlation-based)")
    print("2. Multivector (collaborative filtering)")
    method_input = input("Enter method number (1-2) or press Enter for multivector: ").strip()

    # Parse inputs
    try:
        source_id = int(user_input)
    except ValueError:
        print("Invalid company ID format. Please enter a numeric ID.")
        return

    # Method selection
    method_map = {'1': 'pairwise', '2': 'multivector'}
    method = method_map.get(method_input, 'multivector')  # Default to multivector

    print(f"\nUsing {method} method for recommendations...")

    # Generate recommendations
    try:
        recommendations = recommend(
            company_df, 
            source_id,  
            method=method,
            threshold=0.4,  # For pairwise method
            top_n=10
        )
        
        if not recommendations.empty:
            print(f"\n=== FINAL RECOMMENDATIONS ===")
            
            # Display enhanced descriptions if available
            if 'enhanced_description' in recommendations.columns:
                print("\nTop Recommendations:")
                for i, (_, row) in enumerate(recommendations.head(10).iterrows(), 1):
                    print(f"{i}. {row['enhanced_description']}")
                    if 'correlation' in row:
                        print(f"   Correlation: {row['correlation']:.3f}")
                    elif 'similarity' in row:
                        print(f"   Similarity: {row['similarity']:.3f}")
                    print()
            else:
                print(recommendations.head(10))
        else:
            print("\nNo recommendations could be generated for this investor.")

    except Exception as e:
        traceback.print_exc()

def run_cohort_analysis(company_df):
    """Run the cohort analysis system."""
    print("\n=== COHORT ANALYSIS SYSTEM ===")
    
    # Initialize cohort analyzer
    analyzer = CohortAnalyzer(company_df)
    
    # Get user preferences
    print("\nCohort Analysis Options:")
    print("1. Monthly cohorts (default)")
    print("2. Weekly cohorts")
    print("3. Quarterly cohorts")
    period_input = input("Enter period choice (1-3) or press Enter for monthly: ").strip()
    
    period_map = {'1': 'M', '2': 'W', '3': 'Q'}
    cohort_period = period_map.get(period_input, 'M')
    
    min_requests = input("Enter minimum requests per investor (default: 2): ").strip()
    min_requests = int(min_requests) if min_requests.isdigit() else 2
    
    print(f"\nDefining cohorts with {cohort_period} period grouping...")
    
    # Define cohorts
    cohorts = analyzer.define_cohorts(cohort_period=cohort_period, min_requests=min_requests)
    print(f"\nCohort summary:")
    print(cohorts['cohort'].value_counts().sort_index())
    
    # Create cohort matrix
    cohort_matrix = analyzer.create_cohort_matrix()
    print(f"\nCohort retention matrix:")
    print(cohort_matrix[['cohort', 'period_0', 'retention_30', 'retention_90']].head())
    
    # Analyze cohort behavior
    behavior = analyzer.analyze_cohort_behavior()
    print(f"\nCohort behavior analysis (first cohort):")
    first_cohort = list(behavior.keys())[0]
    print(f"Cohort: {first_cohort}")
    print(f"Total meetings: {behavior[first_cohort]['total_meetings']}")
    print(f"Unique investors: {behavior[first_cohort]['unique_investors']}")
    print(f"Avg meetings per investor: {behavior[first_cohort]['avg_meetings_per_investor']:.2f}")
    
    # Ask if user wants recommendations
    get_recommendations = input("\nWould you like cohort-based recommendations? (y/n): ").strip().lower()
    if get_recommendations == 'y':
        target_cohort = input(f"Enter target cohort (e.g., {first_cohort}): ").strip()
        if not target_cohort:
            target_cohort = first_cohort
            
        print("Recommendation methods:")
        print("1. Similar cohorts (default)")
        print("2. Early adopter patterns")
        method_input = input("Enter method (1-2) or press Enter for similar cohorts: ").strip()
        
        method_map = {'1': 'similar_cohorts', '2': 'early_adopter_patterns'}
        method = method_map.get(method_input, 'similar_cohorts')
        
        try:
            recommendations = analyzer.generate_cohort_recommendations(target_cohort, method=method)
            print(f"\nTop 5 recommendations for cohort {target_cohort}:")
            
            # Convert recommendations dict to DataFrame for enhanced display
            rec_df = pd.DataFrame(list(recommendations.items()), columns=['recommended_company', 'score'])
            rec_df = rec_df.head(5)
            
            # Apply enhanced classification
            from engines.recommender_helpers import excel_classify
            rec_df = excel_classify(rec_df)
            
            for i, (_, row) in enumerate(rec_df.iterrows(), 1):
                if 'enhanced_description' in row:
                    print(f"{i}. {row['enhanced_description']}")
                    print(f"   Score: {row['score']:.2f}")
                else:
                    print(f"{i}. {row['recommended_company']}: {row['score']:.2f}")
                print()
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            traceback.print_exc()

def main():
    """Main function to run the analysis system."""
    print("=== MOSAIC SUMMIT ANALYSIS SYSTEM ===")
    print("Choose an analysis type:")
    print("1. Company Recommendations")
    print("2. Cohort Analysis")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    # Data cleaning; cleans excel spreadsheet into dataframe
    print("\nLoading and processing data...")
    company_df = pd.read_excel("data/Conf 2024 Request List Update.xlsx")
    company_df = company_df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
    company_df = company_df.rename(columns = 
        {  
            'Target Company - who was requested to meet':'target company',
            'Source Company - who made the request':'source company', 
            'Request Date Created' : 'request date'
        }
    )
    company_df.columns = [col.strip().lower().replace(" ", "_") for col in company_df.columns]
    cols = company_df.columns.to_list()
    cols[0], cols[1] = cols[1], cols[0]
    company_df = company_df[cols] # flips source company and target company columns
    company_df = company_df.drop_duplicates()

    # Batch Mosaic Summit entries based on content after colon
    company_df = batch_mosaic_summit_entries(company_df, 'target_company')

    excluded_phrases = [
        'Test Company',
        'Floor Monitor',
        'KeyBank',
        'KeyBanc Capital',
        'Mosaic Summit: Thematic Dinner',
        '(Optional)'
    ]

    company_df = filter_companies_by_phrases(company_df, excluded_phrases, 'target_company')
    
    print(f"Data loaded: {len(company_df)} records, {company_df['source_company'].nunique()} unique investors")
    
    # Route to appropriate analysis
    if choice == '1':
        run_recommendation_system(company_df)
    elif choice == '2':
        run_cohort_analysis(company_df)
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")

if __name__ == "__main__":
    main()
