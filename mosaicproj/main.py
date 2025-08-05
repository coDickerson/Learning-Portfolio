import numpy as np
import pandas as pd
from source_company import SourceCompany
from recommender_engine import recommend


#cleans excel spreadsheet into dataframe
company_df = pd.read_excel("Conf 2024 Request List Update.xlsx")
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

# creates the source company map containing id, requested, dates, and recommended
grouped = company_df.groupby('source_company')
source_company_map = {}
for source_id, group in grouped:
    requested = group['target_company'].tolist()
    dates = set(group['request_date'])        
    source_company_map[source_id] = SourceCompany(source_id, requested, dates)

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
    exit()

# Method selection
method_map = {'1': 'pairwise', '2': 'multivector'}
method = method_map.get(method_input, 'multivector')  # Default to multivector

print(f"\nUsing {method} method for recommendations...")

# Generate recommendations
try:
    recommendations = recommend(
        company_df, 
        source_id, 
        source_company_map, 
        method=method,
        threshold=0.4,  # For pairwise method
        top_n=10
    )
    
    if not recommendations.empty:
        print(f"\n=== FINAL RECOMMENDATIONS ===")
        print(recommendations.head(10))
    else:
        print("\nNo recommendations could be generated for this investor.")
        
except Exception as e:
    print(f"Error generating recommendations: {e}")
    print("Please check your input and try again.")
