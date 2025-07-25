from main import df, source_company_map
import pandas as pd

# recommend other correlated companies to meet based on previous behavior
# ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
#     do you want to go ask your client if they might be interested?

def recommend(source_company_id, threshold=.4):
    # binary interaction matrix
    # describes if a target company was requested by a source, 1=yes and 0=no
    interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])

    # pearson correlation
    company_corr = interaction_matrix.corr(method='pearson')

    # checks if company id exists
    if source_company_id not in interaction_matrix.index:
        print(f'Investor {source_company_id} not found.')
        return

    # source company row from map and list of requested companies
    source_company_row = source_company_map[source_company_id]
    requested_companies = source_company_row.requested
    
    # flattens correlation matrix into a workable table
    # filters out correlation below threshold and finds pairs with one company from requsted companies list
    company_corr.columns.name = None
    company_corr.index.name = None
    company_pairs = company_corr.stack().reset_index() # flattens square matrix
    company_pairs.columns = ['company_a', 'company_b', 'correlation']
    company_pairs = company_pairs[company_pairs['company_a'] < company_pairs['company_b']] # deals with duplicates and same company correlations
    significant_pairs = company_pairs[company_pairs['correlation'] > threshold] 
    filtered = significant_pairs [
    ((significant_pairs['company_a'].isin(requested_companies)) & (~company_pairs['company_b'].isin(requested_companies))) | 
    ((significant_pairs['company_b'].isin(requested_companies)) & (~company_pairs['company_a'].isin(requested_companies)))
    ]
    filtered = filtered.sort_values(by='correlation', ascending=False)

    # print(filtered)
    # print(f'\n {source_company_row.requested}')

    # adds other company and correlation to suggestions list
    suggestions = []
    for _, row in filtered.iterrows():
        if row['company_a'] in requested_companies and row['company_b'] not in requested_companies:
            suggestions.append((row['company_b'], row['correlation']))
        elif row['company_b'] in requested_companies and row['company_a'] not in requested_companies:
            suggestions.append((row['company_a'], row['correlation']))

    # adds suggestions to data frame to be printed
    recommendations_df = pd.DataFrame(suggestions, columns=['recommended company', 'correlation'])
    recommendations_df = recommendations_df.drop_duplicates(subset='recommended company')
    recommendations_df = recommendations_df.sort_values(by='correlation', ascending=False)

    print("\nHere are a list of similar companies that are based on your previous company requests:\n")
    print(recommendations_df.head(10))
    return (recommendations_df)
    
    