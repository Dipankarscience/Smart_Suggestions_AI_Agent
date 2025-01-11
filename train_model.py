import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
import pickle
import re

# Function to load data
def load_data(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = [pd.read_csv(file) for file in files]
    data = pd.concat(dataframes, ignore_index=True)
    return data

# Tokenizer for XPath
def tokenize_xpath(xpath):
    token_pattern = re.compile(
        r'(/+)|'                           # Path separator (/, //)
        r'([a-zA-Z_][a-zAZ0-9_-]*)|'      # Node names
        r'\[@([a-zA-Z_][a-zA-Z0-9_-]*)='   # Attribute name
        r"'([^']*)'\]|"                    # Attribute value
        r'\[(\d+)\]'                       # Index predicate
    )

    tokens = []
    for match in token_pattern.finditer(xpath):
        if match.group(1):  # Path separator
            tokens.append({'type': 'separator', 'value': match.group(1)})
        elif match.group(2):  # Node name
            tokens.append({'type': 'node', 'value': match.group(2)})
        elif match.group(3) and match.group(4):  # Attribute
            tokens.append({'type': 'attribute', 'name': match.group(3), 'value': match.group(4)})
        elif match.group(5):  # Index predicate
            tokens.append({'type': 'index', 'value': int(match.group(5))})
    return tokens

# Preprocessing function
def preprocess_data(data):
    data['xpaths'] = data['xpath'].str.split('|')
    data = data.explode('xpaths')
    data = data[['webelement_id', 'xpaths']].dropna()
    data['tokenized_xpaths'] = data['xpaths'].apply(tokenize_xpath)
    return data

# Encoding function
def encode_data(data, tokenizer, max_length=128):
    def tokens_to_string(tokens):
        return " ".join(f"{token['type']}:{token['value']}" for token in tokens)

    tokenized_sequences = data['tokenized_xpaths'].apply(tokens_to_string).tolist()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['webelement_id'])
    encodings = tokenizer(
        tokenized_sequences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings, labels, label_encoder

# Training the model with improved logic
def train_model(encodings, labels, num_classes, epochs=5, batch_size=16):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    }, labels)).shuffle(len(labels)).batch(batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for batch, (features, batch_labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(features, training=True).logits
                loss = loss_fn(batch_labels, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            epoch_accuracy.update_state(batch_labels, logits)

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy.result().numpy():.4f}")

    return model

# Main function to train and save model
def main(data_dir):
    print("Loading data...")
    data = load_data(data_dir)

    print("Preprocessing data...")
    data = preprocess_data(data)

    print("Encoding data...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings, labels, label_encoder = encode_data(data, tokenizer)

    num_classes = len(set(labels))

    print("Training model...")
    model = train_model(encodings, labels, num_classes)

    print("Saving model...")
    model.save_pretrained("xpath_model")
    tokenizer.save_pretrained("xpath_model")
    with open("xpath_model/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Model trained and saved successfully!")

# Run the script
if __name__ == "__main__":
    data_dir = "C:\\Users\\DIPANKAR\\Desktop\\xpath model\\New folder\\dataset"  # Replace with your actual data directory
    main(data_dir)