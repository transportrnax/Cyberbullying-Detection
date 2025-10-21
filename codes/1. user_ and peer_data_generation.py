import csv
import random
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler


# This is the first file to run

# Function to generate random age between 8 and 18 (both inclusive)
def generate_random_age():
    return random.randint(8, 18)

# Function to generate random school name from a list of 15 schools
def generate_random_school():
    schools = [f"School{i}" for i in range(1, 16)]
    return random.choice(schools)

# Function to calculate grade based on age
def calculate_grade(age):
    return max(1, age - 6)

# Generating data for 500 users
users_data = []
num_of_users=100
for user_id in range(1, num_of_users+1):
    age = generate_random_age()
    gender = random.choice(["Male", "Female", "Others"])
    school_name = generate_random_school()
    grade = calculate_grade(age)

    user_data = [user_id, age, gender, school_name, grade]
    users_data.append(user_data)

# Writing data to CSV file
csv_file_path = "users_data.csv"
header = ["UserID", "Age", "Gender", "School Name", "Grade"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(users_data)

print(f"CSV file '{csv_file_path}' generated successfully.")

#Function to perform min-max normalization
def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Function to compute peerness values
def compute_peerness(user1, user2):
    min_age, max_age = 8, 18
    min_grade, max_grade = 2, 12

    # Normalizing age and grade values
    scaler = MinMaxScaler()
    normalized_age_user1 = min_max_normalize(user1['Age'], min_age, max_age)
    normalized_age_user2 = min_max_normalize(user2['Age'], min_age, max_age)
    normalized_grade_user1 = min_max_normalize(user1['Grade'], min_grade, max_grade)
    normalized_grade_user2 = min_max_normalize(user2['Grade'], min_grade, max_grade)

    # Computing Gradescore and AgeScore
    gradescore = (1 - abs(normalized_grade_user1 - normalized_grade_user2)) / 3
    agescore = (1 - abs(normalized_age_user1 - normalized_age_user2)) / 3

    # Computing SchoolScore
    schoolscore = 0.33 if user1['School Name'] == user2['School Name'] else 0.1

    # Computing Peerness
    peerness = gradescore + agescore + schoolscore

    return peerness

# Reading user data from CSV file
csv_file_path = "users_data.csv"
users_data = []

with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        users_data.append({
            'UserID': int(row['UserID']),
            'Age': int(row['Age']),
            'Gender': row['Gender'],
            'School Name': row['School Name'],
            'Grade': int(row['Grade'])
        })

# Generating peerness values and writing to CSV file
output_csv_file_path = "peerness_values.csv"
header = ["User 1", "User 2", "Peerness"]

with open(output_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Computing peerness for each distinct pair of users
    for user1, user2 in combinations(users_data, 2):
        peerness_value = compute_peerness(user1, user2)
        writer.writerow([user1['UserID'], user2['UserID'], peerness_value])

print(f"Peerness values written to '{output_csv_file_path}' successfully.")

