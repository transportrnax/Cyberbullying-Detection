# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:11:05 2024

@author: navee
"""

import pandas as pd

# Read the Final_DataSet.csv file
final_df = pd.read_csv('Communication_Data_Unique_Rows.csv')

# Group by User1 ID and User2 ID, calculate total messages and aggressive count
grouped_df = final_df.groupby(['User1 ID', 'User2 ID']).agg({'Message': 'count', 'Label': lambda x: (x == 1).sum()}).reset_index()
grouped_df.rename(columns={'Message': 'Total_messages', 'Label': 'Aggressive_Count'}, inplace=True)

# Save the results to Repetition.csv
grouped_df.to_csv('Repetition.csv', index=False)

print("Repetition.csv has been created successfully.")
