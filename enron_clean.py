import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("C:\\Users\\Acer\\Downloads\\enron_emails_1702 (2).csv", low_memory=False)

#Someplaces have random data instead of the folder structure suppoed to be present in the X-Folder data. So keep only records where the entry is the folder structure.
df_filtered = df[df['X-Folder'].str.contains('\\\\', regex=True, na=False)]
#Even after manually removing irrelevant columns, due to errors in dataset they were coming as unnamed columns with no data. So removed them
df_filtered = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

#Remove blank rows
df_filtered = df_filtered.replace(r'^\s*$', np.nan, regex=True)
df_filtered = df_filtered.dropna(how='all')

# Split the strings in 'X-Folder' by backslash '\' and take the last element. This gives the folder name. ie the target we will predict later.
df_filtered['X-Folder'] = df_filtered['X-Folder'].apply(lambda x: x.split('\\')[-1])

# Display the updated DataFrame
print(df_filtered)

df_filtered.to_csv("C:\\Users\\Acer\\Downloads\\enron_filtered3.csv")