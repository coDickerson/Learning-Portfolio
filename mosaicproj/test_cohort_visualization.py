import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the cohort analyzer
from engines.cohort_analysis import CohortAnalyzer
from engines.recommender_helpers import batch_mosaic_summit_entries

def load_and_prepare_data():
    """Load and prepare the meeting request data."""
    print("Loading data...")
    
    # Load the Excel file
    df = pd.read_excel("data/Conf 2024 Request List Update.xlsx")
    
    # Clean and rename columns
    df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
    df = df.rename(columns={
        'Target Company - who was requested to meet': 'target_company',
        'Source Company - who made the request': 'source_company',
        'Request Date Created': 'request_date'
    })
    
    # Clean column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    # Reorder columns
    cols = df.columns.to_list()
    cols[0], cols[1] = cols[1], cols[0]
    df = df[cols]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Batch Mosaic Summit entries based on content after colon
    df = batch_mosaic_summit_entries(df, 'target_company')
    
    print(f"Loaded {len(df)} meeting requests")
    print(f"Date range: {df['request_date'].min()} to {df['request_date'].max()}")
    
    return df

def create_cohort_summary_table(analyzer):
    """Create a summary table of all cohorts."""
    cohort_data = analyzer.cohort_data
    
    # Create summary statistics
    summary_stats = cohort_data.groupby('cohort').agg({
        'source_company': 'count',
        'first_request_date': ['min', 'max']
    }).round(2)
    
    summary_stats.columns = ['Investor_Count', 'First_Request_Min', 'First_Request_Max']
    summary_stats = summary_stats.reset_index()
    
    # Add cohort period info
    summary_stats['Cohort_Period'] = summary_stats['cohort'].astype(str)
    
    # Format dates properly - handle both datetime and Period objects
    def format_date_range(row):
        min_date = row['First_Request_Min']
        max_date = row['First_Request_Max']
        
        # Convert to string format if needed
        if hasattr(min_date, 'strftime'):
            min_str = min_date.strftime('%Y-%m-%d')
        else:
            min_str = str(min_date)[:10]  # Take first 10 characters for YYYY-MM-DD
            
        if hasattr(max_date, 'strftime'):
            max_str = max_date.strftime('%Y-%m-%d')
        else:
            max_str = str(max_date)[:10]  # Take first 10 characters for YYYY-MM-DD
            
        return f"{min_str} to {max_str}"
    
    summary_stats['Date_Range'] = summary_stats.apply(format_date_range, axis=1)
    
    return summary_stats[['Cohort_Period', 'Investor_Count', 'Date_Range']]

def create_retention_table(analyzer):
    """Create a formatted retention table."""
    if analyzer.cohort_matrix is None:
        analyzer.create_cohort_matrix()
    
    # Get retention columns
    retention_cols = [col for col in analyzer.cohort_matrix.columns if col.startswith('retention_')]
    
    # Create formatted table
    retention_table = analyzer.cohort_matrix[['cohort'] + retention_cols].copy()
    
    # Rename columns for better display
    retention_table.columns = ['Cohort'] + [f'{col.split("_")[1]}_Days' for col in retention_cols]
    
    # Format retention rates as percentages
    for col in retention_table.columns[1:]:
        retention_table[col] = retention_table[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    return retention_table

def create_cohort_behavior_table(analyzer, top_n=5):
    """Create a table showing top companies for each cohort."""
    behavior = analyzer.analyze_cohort_behavior(top_n_companies=top_n)
    
    # Create a comprehensive table
    behavior_data = []
    
    for cohort, data in behavior.items():
        row = {
            'Cohort': str(cohort),
            'Total_Meetings': data['total_meetings'],
            'Unique_Investors': data['unique_investors'],
            'Avg_Meetings_Per_Investor': round(data['avg_meetings_per_investor'], 2)
        }
        
        # Add top companies
        for i, (company, count) in enumerate(list(data['top_companies'].items())[:top_n], 1):
            row[f'Top_Company_{i}'] = company
            row[f'Company_{i}_Count'] = count
        
        behavior_data.append(row)
    
    return pd.DataFrame(behavior_data)

def create_early_adopters_table(analyzer, top_companies=10):
    """Create a table of early adopters for top companies."""
    # Get top companies by total requests
    top_companies_list = analyzer.df['target_company'].value_counts().head(top_companies).index.tolist()
    
    early_adopters_data = []
    
    for company in top_companies_list:
        analysis = analyzer.identify_early_adopters(company, threshold_days=30)
        
        if analysis:
            row = {
                'Company': company,
                'Total_Requests': analysis['total_requests'],
                'Early_Adopter_Requests': analysis['early_adopter_requests'],
                'Early_Adoption_Rate': f"{analysis['early_adoption_rate']:.1f}%"
            }
            
            # Add cohort breakdown
            cohort_analysis = analysis['cohort_analysis']
            for cohort in cohort_analysis.index:
                row[f'Cohort_{cohort}_Early_Adopters'] = cohort_analysis.loc[cohort, 'early_adopter_count']
                row[f'Cohort_{cohort}_Avg_Days'] = round(cohort_analysis.loc[cohort, 'avg_days_to_adopt'], 1)
            
            early_adopters_data.append(row)
    
    return pd.DataFrame(early_adopters_data)

def create_visualization_dashboard():
    """Create a comprehensive visualization dashboard."""
    print("Creating Cohort Analysis Visualization Dashboard...")
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize analyzer
    analyzer = CohortAnalyzer(df)
    
    # Define cohorts
    print("\nDefining cohorts...")
    cohorts = analyzer.define_cohorts(cohort_period='M', min_requests=2)
    
    # Create cohort matrix
    print("Creating cohort matrix...")
    cohort_matrix = analyzer.create_cohort_matrix()
    
    # Create tables
    print("Creating visualization tables...")
    
    # 1. Cohort Summary Table
    summary_table = create_cohort_summary_table(analyzer)
    print("\n=== COHORT SUMMARY TABLE ===")
    print(summary_table.to_string(index=False))
    
    # 2. Retention Table
    retention_table = create_retention_table(analyzer)
    print("\n=== COHORT RETENTION TABLE ===")
    print(retention_table.to_string(index=False))
    
    # 3. Cohort Behavior Table
    behavior_table = create_cohort_behavior_table(analyzer, top_n=3)
    print("\n=== COHORT BEHAVIOR TABLE ===")
    print(behavior_table.to_string(index=False))
    
    # 4. Early Adopters Table
    early_adopters_table = create_early_adopters_table(analyzer, top_companies=5)
    print("\n=== EARLY ADOPTERS TABLE ===")
    print(early_adopters_table.to_string(index=False))
    
    # Create interactive visualizations
    print("\nCreating interactive visualizations...")
    
    # 1. Cohort Size Bar Chart
    fig1 = analyzer.visualize_cohort_size_trend()
    
    # 2. Retention Heatmap
    fig2 = analyzer.visualize_cohort_retention()
    
    # 3. Create a comprehensive table visualization
    create_interactive_tables(analyzer, summary_table, retention_table, behavior_table, early_adopters_table)
    
    return analyzer, summary_table, retention_table, behavior_table, early_adopters_table

def create_consolidated_csv(summary_table, retention_table, behavior_table, early_adopters_table, filename='cohort_analysis_results.csv'):
    """
    Consolidate all cohort analysis tables into a single CSV file with multiple sheets.
    
    Args:
        summary_table: Cohort summary data
        retention_table: Cohort retention data  
        behavior_table: Cohort behavior data
        early_adopters_table: Early adopters data
        filename: Output CSV filename
    """
    import pandas as pd
    
    # Create a consolidated DataFrame with all tables
    consolidated_data = []
    
    # Add summary table data
    for _, row in summary_table.iterrows():
        consolidated_data.append({
            'Table_Type': 'Cohort_Summary',
            'Cohort_Period': row['Cohort_Period'],
            'Investor_Count': row['Investor_Count'],
            'Date_Range': row['Date_Range'],
            'Metric_1': None,
            'Metric_2': None,
            'Metric_3': None,
            'Metric_4': None,
            'Metric_5': None
        })
    
    # Add retention table data
    for _, row in retention_table.iterrows():
        consolidated_data.append({
            'Table_Type': 'Cohort_Retention',
            'Cohort_Period': row['Cohort'],
            'Investor_Count': None,
            'Date_Range': None,
            'Metric_1': row['7_Days'],
            'Metric_2': row['14_Days'],
            'Metric_3': row['30_Days'],
            'Metric_4': row['60_Days'],
            'Metric_5': row['90_Days']
        })
    
    # Add behavior table data (simplified)
    for _, row in behavior_table.iterrows():
        consolidated_data.append({
            'Table_Type': 'Cohort_Behavior',
            'Cohort_Period': row['Cohort'],
            'Investor_Count': row['Unique_Investors'],
            'Date_Range': None,
            'Metric_1': f"Total_Meetings: {row['Total_Meetings']}",
            'Metric_2': f"Avg_Per_Investor: {row['Avg_Meetings_Per_Investor']}",
            'Metric_3': f"Top_Company: {row.get('Top_Company_1', 'N/A')}",
            'Metric_4': f"Company_1_Count: {row.get('Company_1_Count', 'N/A')}",
            'Metric_5': f"Top_Company_2: {row.get('Top_Company_2', 'N/A')}"
        })
    
    # Add early adopters data (top 5 companies)
    for _, row in early_adopters_table.head(5).iterrows():
        consolidated_data.append({
            'Table_Type': 'Early_Adopters',
            'Cohort_Period': row['Company'],
            'Investor_Count': row['Total_Requests'],
            'Date_Range': None,
            'Metric_1': f"Early_Adopter_Requests: {row['Early_Adopter_Requests']}",
            'Metric_2': f"Early_Adoption_Rate: {row['Early_Adoption_Rate']}",
            'Metric_3': f"Cohort_2025-01_Early_Adopters: {row.get('Cohort_2025-01_Early_Adopters', 'N/A')}",
            'Metric_4': f"Cohort_2025-01_Avg_Days: {row.get('Cohort_2025-01_Avg_Days', 'N/A')}",
            'Metric_5': f"Cohort_2025-02_Early_Adopters: {row.get('Cohort_2025-02_Early_Adopters', 'N/A')}"
        })
    
    # Create consolidated DataFrame
    consolidated_df = pd.DataFrame(consolidated_data)
    
    # Save to CSV
    consolidated_df.to_csv(filename, index=False)
    
    return consolidated_df

def create_interactive_tables(analyzer, summary_table, retention_table, behavior_table, early_adopters_table):
    """Create interactive table visualizations using Plotly."""
    
    # Convert Period objects to strings for JSON serialization
    summary_table_display = summary_table.copy()
    summary_table_display['Cohort_Period'] = summary_table_display['Cohort_Period'].astype(str)
    
    retention_table_display = retention_table.copy()
    retention_table_display['Cohort'] = retention_table_display['Cohort'].astype(str)
    
    behavior_table_display = behavior_table.copy()
    behavior_table_display['Cohort'] = behavior_table_display['Cohort'].astype(str)
    
    # 1. Cohort Summary Table
    fig_summary = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_table_display.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[summary_table_display[col] for col in summary_table_display.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig_summary.update_layout(
        title='Cohort Summary Table',
        width=800,
        height=400
    )
    fig_summary.show()
    
    # 2. Retention Table
    fig_retention = go.Figure(data=[go.Table(
        header=dict(
            values=list(retention_table_display.columns),
            fill_color='lightblue',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[retention_table_display[col] for col in retention_table_display.columns],
            fill_color='lightcyan',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig_retention.update_layout(
        title='Cohort Retention Table',
        width=1000,
        height=400
    )
    fig_retention.show()
    
    # 3. Behavior Table (simplified for display)
    behavior_display = behavior_table_display[['Cohort', 'Total_Meetings', 'Unique_Investors', 'Avg_Meetings_Per_Investor']].copy()
    
    fig_behavior = go.Figure(data=[go.Table(
        header=dict(
            values=list(behavior_display.columns),
            fill_color='lightgreen',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[behavior_display[col] for col in behavior_display.columns],
            fill_color='lightyellow',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig_behavior.update_layout(
        title='Cohort Behavior Summary',
        width=800,
        height=400
    )
    fig_behavior.show()
    
    # 4. Early Adopters Table
    early_adopters_display = early_adopters_table[['Company', 'Total_Requests', 'Early_Adopter_Requests', 'Early_Adoption_Rate']].copy()
    
    fig_early = go.Figure(data=[go.Table(
        header=dict(
            values=list(early_adopters_display.columns),
            fill_color='lightcoral',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[early_adopters_display[col] for col in early_adopters_display.columns],
            fill_color='mistyrose',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig_early.update_layout(
        title='Early Adopters Analysis',
        width=800,
        height=400
    )
    fig_early.show()

def run_simple_test():
    """Run a simple test to verify the visualization works."""
    print("=== Simple Cohort Analysis Test ===")
    
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Initialize analyzer
        analyzer = CohortAnalyzer(df)
        
        # Define cohorts
        cohorts = analyzer.define_cohorts(cohort_period='M', min_requests=1)
        
        # Create basic tables
        summary_table = create_cohort_summary_table(analyzer)
        retention_table = create_retention_table(analyzer)
        
        print("\n✅ Test completed successfully!")
        print(f"Created {len(summary_table)} cohorts")
        print(f"Retention table has {len(retention_table)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run simple test first
    if run_simple_test():
        print("\n" + "="*50)
        print("Running full visualization dashboard...")
        print("="*50)
        
        # Run full dashboard
        analyzer, summary_table, retention_table, behavior_table, early_adopters_table = create_visualization_dashboard()
        
        print("\n🎉 Visualization dashboard completed!")
        print("Check the browser for interactive tables and charts.")
        
        # Save consolidated data to single CSV file
        consolidated_df = create_consolidated_csv(summary_table, retention_table, behavior_table, early_adopters_table)
        
        print("\n📁 Consolidated data saved to:")
        print("- cohort_analysis_results.csv")
        print(f"  Contains {len(consolidated_df)} rows across all analysis types")
    else:
        print("❌ Cannot proceed with full dashboard due to test failure.")

