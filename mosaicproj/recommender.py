from main import significant_pairs, interaction_matrix

# recommend other correlated companies to meet based on previous behavior
# ie: 80% of the clients who requested meetings with Block also requested meetings with Dataiku…
#     do you want to go ask your client if they might be interested?

def recommend(source_company_id):
    requested_companies = interaction_matrix.loc[source_company_id]
    requested_companies = requested_companies[requested_companies == 1].index.tolist()
    print(requested_companies)
    print(significant_pairs)
    relevant_pairs = [
        (significant_pairs['company_a'].isin(requested_companies)) |
        (significant_pairs['company_b'].isin(requested_companies))
    ]
    print(relevant_pairs)
    
    