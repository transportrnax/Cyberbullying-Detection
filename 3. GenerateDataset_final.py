import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate a random date between start_date and end_date
def random_date(start_date, end_date):
    return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

# Function to generate a random time (hours and minutes)
def random_time():
    return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

# Load aggressive and non-aggressive messages
agg_df = pd.read_csv('Aggressive_All.csv').dropna(subset=['Message'])
non_agg_df = pd.read_csv('Non_Aggressive_All.csv').dropna(subset=['Message'])

# Generate conversations
conversations = []
labels = []

num_of_users=100
# Ensure more instances where a user sends aggressive messages
agg_instances = random.sample(range(1, num_of_users+1), 40)

for user1 in range(1, num_of_users+1):
    for user2 in range(1, num_of_users+1):
        if user1 != user2:
            # Generate a random number of messages between all pairs
            num_messages = random.randint(1, 40)

            # Create a set to store unique messages
            unique_messages = set()

            for _ in range(num_messages):
                date = random_date(datetime(2021, 1, 1), datetime(2022, 3, 1))
                time = random_time()

                # Determine if the message should be aggressive or non-aggressive
                is_aggressive = random.choice([True, False])

                if is_aggressive:
                    messages_df = agg_df
                else:
                    messages_df = non_agg_df

                # Ensure that there are messages in the dataframe
                if not messages_df.empty:
                    try:
                        message = random.choice(messages_df['Message'])
                    except KeyError:
                        continue  # Skip if there is an issue with the index
                    
                    # Check for empty message and duplicates
                    if isinstance(message, str) and message.strip() and message not in unique_messages:
                        unique_messages.add(message)

                        # Append the conversation details to the lists
                        conversations.append([date.date(), time, user1, user2, message, int(is_aggressive)])
                        labels.append(int(is_aggressive))

# Create the final DataFrame
final_df = pd.DataFrame(conversations, columns=['Date', 'Time', 'User1 ID', 'User2 ID', 'Message', 'Label'])

# Write DataFrame to CSV
final_df.to_csv('Communication_Data.csv', index=False)
