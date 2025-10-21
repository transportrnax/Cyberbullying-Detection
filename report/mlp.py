
# Load the preprocessed data from the CSV file
data = pd.read_csv("final_preprocessed_data.csv")

# Ensure all values in 'Message' are strings
data['Message'] = data['Message'].fillna('')  # Fill NaN with empty strings
data['Message'] = data['Message'].astype(str)  # Convert all to string

# Filter out 20,000 samples with label 0 and 20,000 samples with label 1
label_0_data = data[data['label'] == 0].head(20000)  # Filter for label = 0
label_1_data = data[data['label'] == 1].head(20000)  # Filter for label = 1

# Concatenate the two datasets to create the final training set
train_data = pd.concat([label_0_data, label_1_data])

# Print the shape of the final training data to ensure correct size
print(f"Final training data shape: {train_data.shape}")

# Extract the 'Message' column (assuming it's the text data column) and split each message into words
sentences = train_data['Message'].apply(lambda x: x.split()).tolist()  # Split each sentence into a list of words

# Import Word2Vec from Gensim for training the word embeddings
from gensim.models import Word2Vec

# Train the Word2Vec model using the processed sentences
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model for later use
model.save("word2vec_final.model")

# The model can be loaded later using the following line:

# model = Word2Vec.load"word2vec_final.model")
# Load the trained Word2Vec model
word2vec_model = Word2Vec.load("word2vec_final.model")

# Get the word vector for a specific word (e.g., 'hello')
vector = word2vec_model.wv['hi']

# Get the top 5 most similar words to 'hello'
similar_words = word2vec_model.wv.most_similar('hi', topn=5)
print(similar_words)
# Load the preprocessed data from the CSV file
data = pd.read_csv("data_preprocessed_val.csv")
# Ensure all values in 'Message' are strings
data['text'] = data['text'].fillna('')  # Fill NaN with empty strings
data['text'] = data['text'].astype(str)  # Convert all to string
from collections import defaultdict

def compute_class_word_frequencies(data, labels):
    class_0_words = Counter()
    class_1_words = Counter()
    total_class_0_words = 0
    total_class_1_words = 0
    
    for message, label in zip(data, labels):
        words = message.split()  # Split the message into words
        if label == 0:
            class_0_words.update(words)
            total_class_0_words += len(words)
        else:
            class_1_words.update(words)
            total_class_1_words += len(words)
    
    return class_0_words, class_1_words, total_class_0_words, total_class_1_words

# Now we can compute NBLCR weights
def compute_nblcr_weights(class_0_words, class_1_words, total_class_0_words, total_class_1_words):
    nblcr_weights = {}
    
    all_words = set(class_0_words.keys()).union(set(class_1_words.keys()))
    
    for word in all_words:
        p_w_class_0 = class_0_words.get(word, 0) / total_class_0_words
        p_w_class_1 = class_1_words.get(word, 0) / total_class_1_words
        
        # NBLCR weight: log ratio of probabilities
        if p_w_class_1 > 0 and p_w_class_0 > 0:
            nblcr_weight = np.log(p_w_class_1 / p_w_class_0)
        else:
            nblcr_weight = 0
        
        nblcr_weights[word] = nblcr_weight
    
    return nblcr_weights

# Extract messages and labels
messages = data['text']
labels = data['label']

# Compute word frequencies
class_0_words, class_1_words, total_class_0_words, total_class_1_words = compute_class_word_frequencies(messages, labels)

# Compute NBLCR weights for all words
nblcr_weights = compute_nblcr_weights(class_0_words, class_1_words, total_class_0_words, total_class_1_words)

def compute_sentence_vector_nblcr(sentence, word2vec_model, nblcr_weights):
    words = sentence.split()  # Split sentence into words
    sentence_vector = np.zeros(word2vec_model.vector_size)  # Initialize sentence vector with zeros
    total_weight = 0.0  # Initialize weight accumulator
    
    for word in words:
        if word in word2vec_model.wv:  # Check if the word exists in the Word2Vec model
            weight = nblcr_weights.get(word, 0)  # Get NBLCR weight for the word (default is 0)
            sentence_vector += weight * word2vec_model.wv[word]  # Add weighted word vector to sentence vector
            total_weight += abs(weight)  # Accumulate weight
    
    # Normalize the sentence vector by the total weight if there are any weights
    if total_weight > 0:
        sentence_vector /= total_weight
    
    return sentence_vector

# Now, we can compute sentence vectors for all messages
sentence_vectors = np.array([
    compute_sentence_vector_nblcr(text, word2vec_model, nblcr_weights) 
    for text in data['text']
])
import pickle

#Save the merged nblcr_weights
with open("nblcr_weights_combined.pkl", "wb") as f:
    pickle.dump(nblcr_weights, f)
    print("Combined NBLCR weights have been saved to 'nblcr_weights_combined.pkl'.")
from keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Get the labels
y = data['label']  # Assuming 'label' column contains the labels

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Define the neural network structure using Functional API
input_layer = Input(shape=(100,))  # Assuming 100 features after processing
x = Dense(256, activation='relu')(input_layer)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax')(x)  # 2 classes for binary classification

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(),  # Use Adam optimizer
              loss='sparse_categorical_crossentropy',  # Sparse categorical crossentropy for integer labels
              metrics=['accuracy'])
# Print model structure
model.summary()


# Function to evaluate model using KFold
def evaluate_model(model, X, y, kf):
    accuracy_scores = []
    classification_reports = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict the classes
        predictions = np.argmax(model.predict(X_test), axis=1)

        # Collect the metrics
        accuracy_scores.append(accuracy_score(y_test, predictions))
        classification_reports.append(classification_report(y_test, predictions))

    avg_accuracy = np.mean(accuracy_scores)
    avg_classification_report = classification_reports

    return avg_accuracy, avg_classification_report

# Evaluate the model using KFold cross-validation
accuracy, report = evaluate_model(model, sentence_vectors, y, kf)

# Print results
print(f"Average Accuracy: {accuracy}")
print("Classification Report (from each fold):")
for idx, fold_report in enumerate(report):
    print(f"\nFold {idx+1} Classification Report:")
    print(fold_report)
