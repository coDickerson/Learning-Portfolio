import numpy as np
import pandas as pd
from classificationcluster import classify, visualize
from source_company import SourceCompany

#cleans excel spreadsheet into dataframe
df = pd.read_excel("Conf 2024 Request List Update.xlsx")
df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
df = df.rename(columns = 
            {
            'Target Company - who was requested to meet':'target company',
            'Source Company - who made the request':'source company', 
            'Request Date Created' : 'request date'
            }
        )
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
cols = df.columns.to_list()
cols[0], cols[1] = cols[1], cols[0]
df = df[cols] # flips source company and target company columns
df = df.drop_duplicates()

# creates the source company map containing id, requested, dates, and recommended
grouped = df.groupby('source_company')
source_company_map = {}
for source_id, group in grouped:
    requested = group['target_company'].tolist()
    dates = set(group['request_date']) 
    source_company_map[source_id] = SourceCompany(source_id, requested, dates)



