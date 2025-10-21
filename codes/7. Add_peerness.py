# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:50:38 2024

@author: navee
"""

import pandas as pd

# Assuming 'Repetition_Intent.csv' and 'peerness_values.csv' are the names of your CSV files
repetition_intent_df = pd.read_csv('Repetition_Intent.csv')
peerness_values_df = pd.read_csv('peerness_values.csv')

# Merge dataframes based on User1 ID and User2 ID
merged_df = pd.merge(repetition_intent_df, peerness_values_df, left_on=['User1 ID', 'User2 ID'], right_on=['User 1', 'User 2'], how='left')

# Drop redundant columns User1 and User2
merged_df = merged_df.drop(['User 1', 'User 2'], axis=1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('Repetition_Intent_peerness.csv', index=False)
