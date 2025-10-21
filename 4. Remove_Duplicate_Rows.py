# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:15:07 2024

@author: navee
"""

# Remove repetition of text


import pandas as pd

# Read the input CSV file
input_file = "Communication_Data.csv"
output_file = "Communication_Data_Unique_Rows.csv"

df = pd.read_csv(input_file)

# Drop duplicate rows based on the "Message" column, keeping the first instance
df.drop_duplicates(subset="Message", keep="first", inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Duplicate rows removed. Output saved to {output_file}")

label_counts = df['Label'].value_counts()

print("Number of rows with Label=1:", label_counts.get(1, 0))
print("Number of rows with Label=0:", label_counts.get(0, 0))

total_rows = len(df)
print("Total number of rows:", total_rows)