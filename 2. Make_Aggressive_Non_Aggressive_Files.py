import os
import pandas as pd
import re
import random

# Function to clean text
def clean_text(text):
    # Check if text is a string; if not, return an empty string
    if isinstance(text, str):
        # Remove special characters, hashtags, retweets, and UTC
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return cleaned_text.strip()
    else:
        return ''

# Folder path containing CSV files
folder_path = "Messages"

# Initialize empty lists for aggressive and non-aggressive messages
aggressive_messages_list = []
non_aggressive_messages_list = []

# Loop through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Filter aggressive and non-aggressive messages
        aggressive_messages = df[df['Label'] == 1]
        non_aggressive_messages = df[df['Label'] == 0]

        # Clean text in both categories
        aggressive_messages['Text'] = aggressive_messages['Text'].apply(clean_text)
        non_aggressive_messages['Text'] = non_aggressive_messages['Text'].apply(clean_text)

        # Append to the main lists
        aggressive_messages_list.extend(aggressive_messages['Text'].tolist())
        non_aggressive_messages_list.extend(non_aggressive_messages['Text'].tolist())

# Shuffle the lists to randomize the order
random.shuffle(aggressive_messages_list)
random.shuffle(non_aggressive_messages_list)

# Select the desired number of messages from each category (minimum between aggressive and non-aggressive)
desired_messages_count = min(len(aggressive_messages_list), len(non_aggressive_messages_list))
aggressive_messages_list = aggressive_messages_list[:desired_messages_count]
non_aggressive_messages_list = non_aggressive_messages_list[:desired_messages_count]

# Create DataFrames from the lists
aggressive_df = pd.DataFrame({'No.': range(1, desired_messages_count + 1), 'Message': aggressive_messages_list})
non_aggressive_df = pd.DataFrame({'No.': range(1, desired_messages_count + 1), 'Message': non_aggressive_messages_list})

# Save the cleaned DataFrames to CSV files
aggressive_df.to_csv('Aggressive_All.csv', index=False)
non_aggressive_df.to_csv('Non_Aggressive_All.csv', index=False)
