import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CohortAnalyzer:
    """
    True cohort analysis for investor-company meeting requests.
    
    Defines cohorts based on when investors first started requesting meetings
    and analyzes their behavior patterns over time.
    """
    
    def __init__(self, df):
        """
        Initialize with meeting request data.
        
        Args:
            df: DataFrame with columns ['source_company', 'target_company', 'request_date']
        """
        self.df = df.copy()
        self.df['request_date'] = pd.to_datetime(self.df['request_date'])
        self.cohort_data = None
        self.cohort_matrix = None
        
    def define_cohorts(self, cohort_period='M', min_requests=1):
        """
        Define investor cohorts based on their first meeting request.
        
        Args:
            cohort_period: Time period for cohort grouping ('D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly)
            min_requests: Minimum number of requests required to be included in analysis
        """
        print(f"Defining cohorts with {cohort_period} period grouping...")
        
        # Find first request date for each investor
        first_requests = self.df.groupby('source_company')['request_date'].min().reset_index()
        first_requests.columns = ['source_company', 'first_request_date']
        
        # Define cohort based on first request period
        first_requests['cohort'] = first_requests['first_request_date'].dt.to_period(cohort_period)
        
        # Filter investors with minimum requests
        request_counts = self.df.groupby('source_company').size().reset_index(name='total_requests')
        active_investors = request_counts[request_counts['total_requests'] >= min_requests]['source_company']
        
        # Merge cohort info with active investors
        self.cohort_data = first_requests[first_requests['source_company'].isin(active_investors)]
        
        # Add cohort info to main dataframe
        self.df = self.df.merge(self.cohort_data[['source_company', 'cohort', 'first_request_date']], 
                               on='source_company', how='inner')
        
        # Calculate days since first request for each meeting request
        self.df['days_since_first'] = (self.df['request_date'] - self.df['first_request_date']).dt.days
        
        print(f"Created {self.cohort_data['cohort'].nunique()} cohorts")
        print(f"Active investors: {len(self.cohort_data)}")
        
        return self.cohort_data
    
    def create_cohort_matrix(self, time_periods=None):
        """
        Create cohort retention matrix showing investor activity over time.
        
        Args:
            time_periods: List of time periods to analyze (e.g., [0, 7, 14, 30, 60, 90])
        """
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        if time_periods is None:
            time_periods = [0, 7, 14, 30, 60, 90, 120, 180]
        
        # Create cohort matrix
        cohort_matrix = []
        
        for cohort in sorted(self.cohort_data['cohort'].unique()):
            cohort_investors = self.cohort_data[self.cohort_data['cohort'] == cohort]['source_company']
            cohort_row = {'cohort': cohort}
            
            for period in time_periods:
                if period == 0:
                    # Initial cohort size
                    active_count = len(cohort_investors)
                else:
                    # Investors active within this period
                    period_end = self.df['first_request_date'].iloc[0] + timedelta(days=period)
                    active_in_period = self.df[
                        (self.df['source_company'].isin(cohort_investors)) &
                        (self.df['request_date'] <= period_end)
                    ]['source_company'].nunique()
                    active_count = active_in_period
                
                cohort_row[f'period_{period}'] = active_count
            
            cohort_matrix.append(cohort_row)
        
        self.cohort_matrix = pd.DataFrame(cohort_matrix)
        
        # Calculate retention rates
        for period in time_periods[1:]:
            initial_size = self.cohort_matrix['period_0']
            period_size = self.cohort_matrix[f'period_{period}']
            self.cohort_matrix[f'retention_{period}'] = (period_size / initial_size * 100).round(2)
        
        return self.cohort_matrix
    
    def analyze_cohort_behavior(self, top_n_companies=10):
        """
        Analyze which companies different cohorts prefer.
        
        Args:
            top_n_companies: Number of top companies to analyze per cohort
        """
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        cohort_company_analysis = {}
        
        for cohort in sorted(self.cohort_data['cohort'].unique()):
            cohort_investors = self.cohort_data[self.cohort_data['cohort'] == cohort]['source_company']
            cohort_meetings = self.df[self.df['source_company'].isin(cohort_investors)]
            
            # Top companies requested by this cohort
            top_companies = cohort_meetings['target_company'].value_counts().head(top_n_companies)
            
            # Calculate cohort-specific metrics
            total_meetings = len(cohort_meetings)
            unique_investors = len(cohort_investors)
            avg_meetings_per_investor = total_meetings / unique_investors
            
            cohort_company_analysis[cohort] = {
                'total_meetings': total_meetings,
                'unique_investors': unique_investors,
                'avg_meetings_per_investor': avg_meetings_per_investor,
                'top_companies': top_companies.to_dict()
            }
        
        return cohort_company_analysis
    
    def identify_early_adopters(self, company_name, threshold_days=30):
        """
        Identify which cohorts were early adopters of a specific company.
        
        Args:
            company_name: Name of the company to analyze
            threshold_days: Days threshold to consider "early adoption"
        """
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        # Get all requests for this company
        company_requests = self.df[self.df['target_company'] == company_name].copy()
        
        if company_requests.empty:
            print(f"No requests found for company: {company_name}")
            return None
        
        # Find first request date for this company
        company_first_request = company_requests['request_date'].min()
        
        # Calculate days from company launch to each request
        company_requests['days_from_company_launch'] = (
            company_requests['request_date'] - company_first_request
        ).dt.days
        
        # Identify early adopters
        early_adopters = company_requests[
            company_requests['days_from_company_launch'] <= threshold_days
        ]
        
        # Analyze by cohort
        early_adopter_analysis = early_adopters.groupby('cohort').agg({
            'source_company': 'nunique',
            'days_from_company_launch': 'mean'
        }).rename(columns={
            'source_company': 'early_adopter_count',
            'days_from_company_launch': 'avg_days_to_adopt'
        })
        
        return {
            'company_name': company_name,
            'total_requests': len(company_requests),
            'early_adopter_requests': len(early_adopters),
            'early_adoption_rate': len(early_adopters) / len(company_requests) * 100,
            'cohort_analysis': early_adopter_analysis
        }
    
    def seasonal_cohort_analysis(self):
        """
        Analyze if there are seasonal patterns in cohort behavior.
        """
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        # Add seasonal information
        self.df['month'] = self.df['request_date'].dt.month
        self.df['quarter'] = self.df['request_date'].dt.quarter
        self.df['day_of_week'] = self.df['request_date'].dt.dayofweek
        
        # Analyze seasonal patterns by cohort
        seasonal_analysis = {}
        
        for cohort in sorted(self.cohort_data['cohort'].unique()):
            cohort_investors = self.cohort_data[self.cohort_data['cohort'] == cohort]['source_company']
            cohort_meetings = self.df[self.df['source_company'].isin(cohort_investors)]
            
            seasonal_analysis[cohort] = {
                'monthly_distribution': cohort_meetings['month'].value_counts().sort_index().to_dict(),
                'quarterly_distribution': cohort_meetings['quarter'].value_counts().sort_index().to_dict(),
                'day_of_week_distribution': cohort_meetings['day_of_week'].value_counts().sort_index().to_dict()
            }
        
        return seasonal_analysis
    
    def visualize_cohort_retention(self):
        """Create cohort retention heatmap visualization."""
        if self.cohort_matrix is None:
            raise ValueError("Must create cohort matrix first using create_cohort_matrix()")
        
        # Get retention columns
        retention_cols = [col for col in self.cohort_matrix.columns if col.startswith('retention_')]
        
        if not retention_cols:
            print("No retention data available for visualization")
            return
        
        # Prepare data for heatmap
        heatmap_data = self.cohort_matrix.set_index('cohort')[retention_cols]
        
        # Extract period numbers for x-axis labels
        periods = [int(col.split('_')[1]) for col in retention_cols]
        
        fig = px.imshow(
            heatmap_data.values,
            x=periods,
            y=heatmap_data.index.astype(str),
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title='Cohort Retention Heatmap',
            labels={'x': 'Days Since First Request', 'y': 'Cohort', 'color': 'Retention Rate (%)'}
        )
        
        fig.update_layout(
            xaxis_title='Days Since First Request',
            yaxis_title='Cohort',
            height=500
        )
        
        fig.show()
        return fig
    
    def visualize_cohort_size_trend(self):
        """Visualize cohort sizes over time."""
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        cohort_sizes = self.cohort_data['cohort'].value_counts().sort_index()
        
        fig = px.bar(
            x=cohort_sizes.index.astype(str),
            y=cohort_sizes.values,
            title='Cohort Sizes Over Time',
            labels={'x': 'Cohort', 'y': 'Number of Investors'},
            color=cohort_sizes.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Cohort',
            yaxis_title='Number of Investors',
            showlegend=False
        )
        
        fig.show()
        return fig
    
    def generate_cohort_recommendations(self, target_cohort, method='similar_cohorts'):
        """
        Generate company recommendations based on cohort analysis.
        
        Args:
            target_cohort: The cohort to generate recommendations for
            method: 'similar_cohorts' or 'early_adopter_patterns'
        """
        if self.cohort_data is None:
            raise ValueError("Must define cohorts first using define_cohorts()")
        
        if method == 'similar_cohorts':
            # Find cohorts with similar behavior patterns
            cohort_behavior = self.analyze_cohort_behavior()
            
            # Calculate similarity based on top companies
            target_companies = set(cohort_behavior[target_cohort]['top_companies'].keys())
            
            cohort_similarities = {}
            for cohort, behavior in cohort_behavior.items():
                if cohort == target_cohort:
                    continue
                
                cohort_companies = set(behavior['top_companies'].keys())
                similarity = len(target_companies.intersection(cohort_companies)) / len(target_companies.union(cohort_companies))
                cohort_similarities[cohort] = similarity
            
            # Get companies from most similar cohorts
            similar_cohorts = sorted(cohort_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            recommendations = {}
            for cohort, similarity in similar_cohorts:
                cohort_companies = cohort_behavior[cohort]['top_companies']
                for company, count in cohort_companies.items():
                    if company not in target_companies:
                        if company not in recommendations:
                            recommendations[company] = 0
                        recommendations[company] += count * similarity
            
            return dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
        
        elif method == 'early_adopter_patterns':
            # Analyze early adoption patterns across cohorts
            all_companies = self.df['target_company'].unique()
            
            early_adopter_scores = {}
            for company in all_companies:
                adoption_analysis = self.identify_early_adopters(company)
                if adoption_analysis and adoption_analysis['early_adoption_rate'] > 20:
                    early_adopter_scores[company] = adoption_analysis['early_adoption_rate']
            
            return dict(sorted(early_adopter_scores.items(), key=lambda x: x[1], reverse=True))
        
        else:
            raise ValueError("Method must be 'similar_cohorts' or 'early_adopter_patterns'")


def run_cohort_analysis_example():
    """Example usage of the cohort analysis system."""
    print("=== Cohort Analysis Example ===")
    
    # Load data
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
    
    # Initialize cohort analyzer
    analyzer = CohortAnalyzer(df)
    
    # Define cohorts (monthly)
    cohorts = analyzer.define_cohorts(cohort_period='M', min_requests=2)
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
    
    # Generate recommendations
    recommendations = analyzer.generate_cohort_recommendations(first_cohort, method='similar_cohorts')
    print(f"\nTop 5 recommendations for cohort {first_cohort}:")
    for i, (company, score) in enumerate(list(recommendations.items())[:5], 1):
        print(f"{i}. {company}: {score:.2f}")
    
    return analyzer


if __name__ == "__main__":
    run_cohort_analysis_example()

