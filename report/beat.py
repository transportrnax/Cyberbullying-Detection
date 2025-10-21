import pandas as pd

# Read the preprocessed CSV file
data = pd.read_csv("data_preprocessed_gen.csv")
texts = data["text"]
labels = data["label"]


# Check for missing values in 'text' column
print(data['text'].isnull().sum())

# Remove rows with NaN values in 'text' column
data = data.dropna(subset=['text'])

# Tokenizing the text and adding token count column
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data['token_count'] = data['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))

# Print token count statistics
print(data['token_count'].describe())

# Check for empty or whitespace-only texts
empty_or_whitespace = data[data['text'].str.strip() == '']
print(f"Number of empty or whitespace-only rows: {len(empty_or_whitespace)}")

# Remove empty or whitespace-only rows
data = data[data['text'].str.strip() != '']

# Check the character length of each text
data['char_length'] = data['text'].apply(len)
print(data['char_length'].describe())

# Remove texts that are longer than 512 characters
data = data[data['char_length'] <= 512]

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [str(text) if not pd.isnull(text) else '' for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            label = self.labels[idx]

            # Tokenizing the text
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise

# Ensure data has no missing or non-string values
data['text'] = data['text'].fillna('').astype(str)
texts = data['text'].tolist()
data['label'] = data['label'].fillna(0).astype(int)
labels = data['label'].tolist()

# Create Dataset
max_len = 64
dataset = TextDataset(texts, labels, tokenizer, max_len)

# Check a sample from the dataset
sample = dataset[0]
print("Input IDs shape:", sample['input_ids'].shape)
print("Attention Mask shape:", sample['attention_mask'].shape)
print("Label:", sample['label'])

from torch.utils.data import DataLoader, random_split

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for both training and validation
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Check the output of a batch from the DataLoader
for batch in train_loader:
    print("Batch input_ids shape:", batch['input_ids'].shape)  # (batch_size, max_len)
    print("Batch attention_mask shape:", batch['attention_mask'].shape)  # (batch_size, max_len)
    print("Batch labels shape:", batch['label'].shape)  # (batch_size,)
    break

from transformers import BertForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss

# Load the pre-trained BERT model with classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=3e-5)
loss_fn = CrossEntropyLoss()

from tqdm import tqdm

# Define the training loop
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

# Define the evaluation loop
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

# Train and evaluate the model for multiple epochs
epochs = 3
best_accuracy = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
    print(f"Train loss: {train_loss}, accuracy: {train_acc}")

    val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
    print(f"Val loss: {val_loss}, accuracy: {val_acc}")

    # Save the best model
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# Load the best model
from transformers import BertForSequenceClassification, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the optimal model weights
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prediction function
def predict(texts, model, tokenizer, device):
    inputs = tokenizer(
        texts,
        max_length=128,  # Adjust max length as needed
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1)

    return preds.cpu().numpy(), probabilities.cpu().numpy()

# Example test sentences
test_sentences = [
    "good morning!",
    "good idea!",
    "you are stupid",
    "fuck stupid",
    "i love you",
]

# Make predictions
predictions, probabilities = predict(test_sentences, model, tokenizer, device)

# Print results
for i, text in enumerate(test_sentences):
    label = "Offensive" if predictions[i] == 1 else "Not Offensive"
    print(f"Text: {text}")
    print(f"Prediction: {label} (Probability: {probabilities[i][1]:.4f})")
