from main import df, source_company_map
import pandas as pd

# recommend other correlated companies to meet based on previous behavior
# ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
#     do you want to go ask your client if they might be interested?

def recommend(source_company_id, threshold=.9):
    # binary interaction matrix
    # describes if a target company was requested by a source, 1=yes and 0=no
    interaction_matrix = pd.crosstab(df['source_company'], df['target_company'])

    # pearson correlation
    company_corr = interaction_matrix.corr(method='pearson')

    if source_company_id not in interaction_matrix.index:
        print(f'Investor {source_company_id} not found.')
        return

    source_company_row = source_company_map[source_company_id]
    requested_companies = source_company_row.requested
    
    company_corr.columns.name = None
    company_corr.index.name = None
    company_pairs = company_corr.stack().reset_index() # flattens square matrix
    company_pairs.columns = ['company_a', 'company_b', 'correlation']
    company_pairs = company_pairs[company_pairs['company_a'] < company_pairs['company_b']]
    significant_pairs = company_pairs[company_pairs['correlation'] > threshold] # deals with duplicates and same company correlations
    filtered = significant_pairs [
    ((significant_pairs['company_a'].isin(requested_companies)) ) |
    ((significant_pairs['company_b'].isin(requested_companies)) )
    ]
    print(filtered)
    print(f'\n {source_company_row.requested}')





    suggestions = []
    for company in requested_companies:
        if company not in company_corr.columns:
            continue  # skip if no correlation data available

        correlated = company_corr[company]
        filtered = correlated[(correlated > threshold) & (~correlated.index.isin(requested_companies))]

        for other_company, corr_value in filtered.items():
            suggestions.append((other_company, corr_value))

    recommendations_df = pd.DataFrame(suggestions, columns=['recommended company', 'correlation'])
    recommendations_df = recommendations_df.drop_duplicates(subset='recommended company')
    recommendations_df = recommendations_df.sort_values(by='correlation', ascending=False)

    # print(recommendations_df)
    return 



    
    

    # relevant_pairs = company_pairs


    # # new table showing significant correlation between two companies 
    #  # filters for only significant company pairs; .5 arbitrary
    # significant_pairs = significant_pairs.sort_values(by='correlation', ascending=False)
    # # print(significant_pairs)



    
    