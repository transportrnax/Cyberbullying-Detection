import pandas as pd

# Assuming 'Repetition.csv' is the name of your CSV file
df = pd.read_csv('Repetition.csv')

# Calculate A and B
df['A'] = df['Aggressive_Count'] / df['Total_messages']
df['B'] = df['Total_messages'] / df['Total_messages'].max()

# Calculate Intent to Harm
df['Intent_to_Harm'] = 0.5 * df['A'] + 0.5 * df['B']

# Drop intermediate columns 'A' and 'B'
df = df.drop(['A', 'B'], axis=1)

# Save the dataframe to a new CSV file
df.to_csv('Repetition_Intent.csv', index=False)
