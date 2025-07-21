import numpy as np
import pandas as pd

#cleans excel spreadsheet into dataframe
df = pd.read_excel("Conf 2024 Request List Update.xlsx")
df = df.drop(columns=['Source Full Name', 'Source First', 'Source Last'])
df = df.rename(columns = {'Target Company - who was requested to meet':'target company','Source Company - who made the request':'source company', 'Request Date Created':'request date'})
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
df = df.drop_duplicates()

print(df.head())
