import pandas as pd
import numpy as np
from cohort_analysis import CohortAnalyzer
from recommender_engine import recommend
from recommender_helpers import convert_to_recommendations_df, filter_companies_by_phrases

class CohortEnhancedRecommender:
    """
    Enhanced recommendation system that combines traditional methods with cohort analysis.
    """
    
    def __init__(self, df):
        """
        Initialize with meeting request data.
        
        Args:
            df: DataFrame with columns ['source_company', 'target_company', 'request_date']
        """
        self.df = df.copy()
        self.cohort_analyzer = CohortAnalyzer(df)
        self.cohort_data = None
        self.cohort_behavior = None
        
    def setup_cohort_analysis(self, cohort_period='M', min_requests=2):
        """
        Set up cohort analysis for enhanced recommendations.
        
        Args:
            cohort_period: Time period for cohort grouping
            min_requests: Minimum requests required for analysis
        """
        print("Setting up cohort analysis...")
        
        # Define cohorts
        self.cohort_data = self.cohort_analyzer.define_cohorts(
            cohort_period=cohort_period, 
            min_requests=min_requests
        )
        
        # Analyze cohort behavior
        self.cohort_behavior = self.cohort_analyzer.analyze_cohort_behavior()
        
        print("Cohort analysis setup complete!")
        
    def get_investor_cohort(self, source_company_id):
        """
        Get the cohort for a specific investor.
        
        Args:
            source_company_id: ID of the investor
            
        Returns:
            Cohort period or None if not found
        """
        if self.cohort_data is None:
            raise ValueError("Must run setup_cohort_analysis() first")
            
        investor_cohort = self.cohort_data[
            self.cohort_data['source_company'] == source_company_id
        ]
        
        if investor_cohort.empty:
            return None
            
        return investor_cohort.iloc[0]['cohort']
    
    def cohort_enhanced_recommend(self, source_company_id, method='hybrid', 
                                threshold=0.4, top_n=10, cohort_weight=0.3):
        """
        Generate recommendations using both traditional methods and cohort analysis.
        
        Args:
            source_company_id: ID of the investor to generate recommendations for
            method: 'traditional', 'cohort_only', or 'hybrid'
            threshold: Minimum correlation threshold for pairwise method
            top_n: Number of recommendations to return
            cohort_weight: Weight given to cohort-based recommendations in hybrid mode (0-1)
        """
        if self.cohort_data is None:
            raise ValueError("Must run setup_cohort_analysis() first")
        
        # Get investor's cohort
        investor_cohort = self.get_investor_cohort(source_company_id)
        
        if investor_cohort is None:
            print(f"Investor {source_company_id} not found in cohort data. Using traditional methods only.")
            return recommend(self.df, source_company_id, method='multivector', threshold=threshold, top_n=top_n)
        
        print(f"Investor {source_company_id} belongs to cohort {investor_cohort}")
        
        if method == 'traditional':
            # Use only traditional recommendation methods
            return recommend(self.df, source_company_id, method='multivector', threshold=threshold, top_n=top_n)
            
        elif method == 'cohort_only':
            # Use only cohort-based recommendations
            return self._cohort_only_recommend(source_company_id, investor_cohort, top_n)
            
        elif method == 'hybrid':
            # Combine traditional and cohort-based recommendations
            return self._hybrid_recommend(source_company_id, investor_cohort, threshold, top_n, cohort_weight)
            
        else:
            raise ValueError("Method must be 'traditional', 'cohort_only', or 'hybrid'")
    
    def _cohort_only_recommend(self, source_company_id, investor_cohort, top_n):
        """
        Generate recommendations based solely on cohort analysis.
        """
        # Get companies that similar cohorts prefer
        cohort_recommendations = self.cohort_analyzer.generate_cohort_recommendations(
            investor_cohort, method='similar_cohorts'
        )
        
        # Get companies that show early adoption patterns
        early_adopter_recommendations = self.cohort_analyzer.generate_cohort_recommendations(
            investor_cohort, method='early_adopter_patterns'
        )
        
        # Combine and weight the recommendations
        combined_scores = {}
        
        # Add cohort similarity scores
        for company, score in cohort_recommendations.items():
            combined_scores[company] = score * 0.6  # 60% weight to cohort similarity
        
        # Add early adopter scores
        for company, score in early_adopter_recommendations.items():
            if company in combined_scores:
                combined_scores[company] += score * 0.4  # 40% weight to early adoption
            else:
                combined_scores[company] = score * 0.4
        
        return convert_to_recommendations_df(combined_scores, 'cohort_analysis', top_n)
    
    def _hybrid_recommend(self, source_company_id, investor_cohort, threshold, top_n, cohort_weight):
        """
        Combine traditional and cohort-based recommendations.
        """
        # Get traditional recommendations
        traditional_recs = recommend(self.df, source_company_id, method='multivector', threshold=threshold, top_n=top_n*2)
        
        # Get cohort-based recommendations
        cohort_recs = self._cohort_only_recommend(source_company_id, investor_cohort, top_n*2)
        
        # Combine scores
        combined_scores = {}
        
        # Add traditional scores
        if not traditional_recs.empty:
            traditional_weight = 1 - cohort_weight
            for _, row in traditional_recs.iterrows():
                company = row['recommended company']
                score = row['similarity']
                combined_scores[company] = score * traditional_weight
        
        # Add cohort scores
        if not cohort_recs.empty:
            for _, row in cohort_recs.iterrows():
                company = row['recommended company']
                score = row['cohort_analysis']
                if company in combined_scores:
                    combined_scores[company] += score * cohort_weight
                else:
                    combined_scores[company] = score * cohort_weight
        
        return convert_to_recommendations_df(combined_scores, 'hybrid_score', top_n)
    
    def analyze_cohort_trends(self, source_company_id):
        """
        Analyze trends for the investor's cohort to provide insights.
        """
        if self.cohort_data is None:
            raise ValueError("Must run setup_cohort_analysis() first")
        
        investor_cohort = self.get_investor_cohort(source_company_id)
        
        if investor_cohort is None:
            return None
        
        # Get cohort behavior
        cohort_info = self.cohort_behavior[investor_cohort]
        
        # Get seasonal patterns
        seasonal_analysis = self.cohort_analyzer.seasonal_cohort_analysis()
        cohort_seasonal = seasonal_analysis[investor_cohort]
        
        # Find early adopter companies for this cohort
        early_adopter_companies = []
        for company in list(cohort_info['top_companies'].keys())[:5]:
            adoption_analysis = self.cohort_analyzer.identify_early_adopters(company)
            if adoption_analysis and adoption_analysis['early_adoption_rate'] > 30:
                early_adopter_companies.append({
                    'company': company,
                    'early_adoption_rate': adoption_analysis['early_adoption_rate']
                })
        
        return {
            'cohort': investor_cohort,
            'total_meetings': cohort_info['total_meetings'],
            'unique_investors': cohort_info['unique_investors'],
            'avg_meetings_per_investor': cohort_info['avg_meetings_per_investor'],
            'top_companies': cohort_info['top_companies'],
            'seasonal_patterns': cohort_seasonal,
            'early_adopter_companies': early_adopter_companies
        }
    
    def get_cohort_insights(self, source_company_id):
        """
        Generate actionable insights based on cohort analysis.
        """
        trends = self.analyze_cohort_trends(source_company_id)
        
        if trends is None:
            return "No cohort data available for this investor."
        
        insights = []
        
        # Activity level insights
        if trends['avg_meetings_per_investor'] > 5:
            insights.append(f"Your cohort is highly active, averaging {trends['avg_meetings_per_investor']:.1f} meetings per investor")
        elif trends['avg_meetings_per_investor'] < 2:
            insights.append(f"Your cohort shows lower activity, averaging {trends['avg_meetings_per_investor']:.1f} meetings per investor")
        
        # Early adopter insights
        if trends['early_adopter_companies']:
            insights.append(f"Your cohort has been an early adopter of {len(trends['early_adopter_companies'])} companies")
        
        # Seasonal insights
        monthly_dist = trends['seasonal_patterns']['monthly_distribution']
        peak_month = max(monthly_dist, key=monthly_dist.get)
        insights.append(f"Your cohort's peak activity is in month {peak_month}")
        
        return insights
    
    def visualize_cohort_analysis(self, source_company_id):
        """
        Create visualizations for the investor's cohort analysis.
        """
        if self.cohort_data is None:
            raise ValueError("Must run setup_cohort_analysis() first")
        
        investor_cohort = self.get_investor_cohort(source_company_id)
        
        if investor_cohort is None:
            print("No cohort data available for visualization")
            return
        
        # Create cohort retention visualization
        self.cohort_analyzer.create_cohort_matrix()
        self.cohort_analyzer.visualize_cohort_retention()
        
        # Create cohort size trend visualization
        self.cohort_analyzer.visualize_cohort_size_trend()
        
        print(f"Visualizations created for cohort {investor_cohort}")


def run_enhanced_recommendation_example():
    """Example usage of the enhanced recommendation system."""
    print("=== Enhanced Recommendation System Example ===")
    
    # Load and prepare data
    df = pd.read_excel("Conf 2024 Request List Update.xlsx")
    df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
    df = df.rename(columns={
        'Target Company - who was requested to meet': 'target_company',
        'Source Company - who made the request': 'source_company',
        'Request Date Created': 'request_date'
    })
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    cols = df.columns.to_list()
    cols[0], cols[1] = cols[1], cols[0]
    df = df[cols]
    df = df.drop_duplicates()
    
    # Filter out test companies
    excluded_phrases = [
        'Test Company', 'Floor Monitor', 'KeyBank', 'KeyBanc Capital',
        'Mosaic Summit Thematic Dinner', '(Optional)'
    ]
    df = filter_companies_by_phrases(df, excluded_phrases, 'target_company')
    
    # Initialize enhanced recommender
    enhanced_recommender = CohortEnhancedRecommender(df)
    
    # Setup cohort analysis
    enhanced_recommender.setup_cohort_analysis(cohort_period='M', min_requests=2)
    
    # Get user input
    user_input = input("\nEnter source company ID (e.g. 1000-1342): ").strip()
    
    try:
        source_id = int(user_input)
    except ValueError:
        print("Invalid company ID format. Please enter a numeric ID.")
        return
    
    # Generate insights
    insights = enhanced_recommender.get_cohort_insights(source_id)
    print(f"\n=== COHORT INSIGHTS ===")
    for insight in insights:
        print(f"• {insight}")
    
    # Generate recommendations using different methods
    print(f"\n=== RECOMMENDATION COMPARISON ===")
    
    # Traditional method
    print("\n1. Traditional Method (Multivector):")
    traditional_recs = enhanced_recommender.cohort_enhanced_recommend(source_id, method='traditional')
    if not traditional_recs.empty:
        print(traditional_recs.head(5))
    
    # Cohort-only method
    print("\n2. Cohort-Only Method:")
    cohort_recs = enhanced_recommender.cohort_enhanced_recommend(source_id, method='cohort_only')
    if not cohort_recs.empty:
        print(cohort_recs.head(5))
    
    # Hybrid method
    print("\n3. Hybrid Method (Traditional + Cohort):")
    hybrid_recs = enhanced_recommender.cohort_enhanced_recommend(source_id, method='hybrid', cohort_weight=0.4)
    if not hybrid_recs.empty:
        print(hybrid_recs.head(5))
    
    # Ask if user wants visualizations
    viz_input = input("\nWould you like to see cohort visualizations? (y/n): ").strip().lower()
    if viz_input == 'y':
        enhanced_recommender.visualize_cohort_analysis(source_id)
    
    return enhanced_recommender


if __name__ == "__main__":
    run_enhanced_recommendation_example()

