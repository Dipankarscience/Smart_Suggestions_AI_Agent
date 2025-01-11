import streamlit as st
import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pickle
import re

# Function to tokenize XPath
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

# Function to preprocess input for prediction
def preprocess_input(xpath, tokenizer):
    tokenized_xpath = tokenize_xpath(xpath)
    tokenized_sequence = " ".join(f"{token['type']}:{token['value']}" for token in tokenized_xpath)
    encoding = tokenizer(
        tokenized_sequence,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )
    return encoding

# Function to predict WebElement ID
def predict_webelement_id(xpath, model, tokenizer, label_encoder):
    encoding = preprocess_input(xpath, tokenizer)
    logits = model(encoding).logits
    predicted_label = np.argmax(logits, axis=1).item()
    decoded_label = label_encoder.inverse_transform([predicted_label])[0]
    return decoded_label


# Add custom CSS for stylish sidebar and footer
st.markdown("""
    <style>
        .css-1kyxreq {
            background-color: #f1f1f1; 
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .css-1q6en9e {
            padding: 10px;
            font-size: 16px;
            color: #1a73e8;
            background-color: #e1f5fe;
            border-radius: 8px;
        }
        .css-1q6en9e:hover {
            background-color: #a2d1f7;
        }
        .css-15tx938 {
            padding: 20px;
            font-size: 14px;
        }
        
        /* Stylish footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #0a74da;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .footer a {
            color: white;
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Section 1: Select Model
st.sidebar.header("Select Model")
model_names = ["xpath_model", "model2", "model3"]
model_choice = st.sidebar.selectbox("Choose a pre-trained model:", model_names)

# Dynamically change the UI based on the selected model
if model_choice:
    # Load model-specific settings
    model_info = f"Model: {model_choice} selected"
    st.sidebar.write(model_info)

    # Section 2: Update prediction instructions or input UI based on the model
    if "xpath_model" in model_choice:
        # Streamlit UI
        st.title("XPath Model Predictor")
        st.header("XPath webelement identifier model")
        st.write("This model predicts WebElement ID for specific XPath expressions.")

        # XPath input for the prediction
        xpath_input = st.text_input("Enter an XPath for prediction:")

        if st.button("Predict"):
            try:
                model = TFBertForSequenceClassification.from_pretrained(model_choice)
                tokenizer = BertTokenizer.from_pretrained(model_choice)
                with open(f"{model_choice}/label_encoder.pkl", "rb") as f:
                    label_encoder = pickle.load(f)

                decoded_label = predict_webelement_id(xpath_input, model, tokenizer, label_encoder)
                st.success(f"Predicted WebElement ID: {decoded_label}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    elif "model2" in model_choice:
        # Streamlit UI
        st.title("Smart Suggestions")
        st.header("Smart Suggestions")
        st.write("This model suggests method based on given prompt present in framework")
    else:
        # Streamlit UI
        st.title("Smart Suggestions")
        st.header("Predict WebElement ID for Other Models")
        st.write("This is a generic model for WebElement prediction.")

# Footer
st.markdown("""
    <div class="footer">
        <p>Made by PLM QA Team | <a href="https://micron.com">Visit our website</a></p>
    </div>
""", unsafe_allow_html=True)