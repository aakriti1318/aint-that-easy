import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import time
import requests
import json
import os
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Comparison: PyTorch vs. GPT-4o-mini",
    page_icon="🤖",
    layout="wide"
)

# Constants
MAX_SEQUENCE_LENGTH = 200
BATCH_SIZE = 32
EMBEDDING_DIM = 200  
HIDDEN_DIM = 256     
EPOCHS = 10      
NUM_CLASSES = 3  # Negative, Neutral, Positive
LEARNING_RATE = 0.001

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# LSTM Model for sentiment analysis
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# PyTorch pipeline
class PyTorchSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.tokenizer(clean_text(text))
    
    # Modify the clean_text function to handle non-string inputs
    def clean_text(text):
        # Handle non-string inputs (like NaN or float)
        if not isinstance(text, str):
            if pd.isna(text):  # Check if it's NaN
                return ""
            text = str(text)  # Convert other types to string
        
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces, replace punctuation with space
        text = text.lower()  # Convert to lowercase
        
        # Don't remove stopwords as they might contain important sentiment information
        # Instead, keep key negation words that are often in stopwords list
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'never', 'none', 'isn', 'aren', 'wasn', 'weren', 'don', 'didn', 'doesn', 'haven', 'hadn', 'hasn', 'won', 'wouldn', 'should', 'shouldn', 'couldn', 'can', 'cannot'}
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        return text

    # Modify the train method to show class distribution and handle errors
    def train(self, texts, labels):
        # Show class distribution
        label_counts = pd.Series(labels).value_counts().sort_index()
        st.write("Label distribution in training data:")
        st.write(label_counts)
        
        # Convert all texts to strings and handle NaN
        texts = [str(text) if not pd.isna(text) else "" for text in texts]
        
        # Continue with the original training code...
        # Build vocabulary
        self.vocab = build_vocab_from_iterator(self.yield_tokens(texts), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        # Text pipeline
        text_pipeline = lambda x: [self.vocab[token] for token in self.tokenizer(clean_text(x))]
        
        # Create tensors for training
        X = []
        for text in texts:
            tokens = text_pipeline(text)
            if len(tokens) < MAX_SEQUENCE_LENGTH:
                tokens = tokens + [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
            else:
                tokens = tokens[:MAX_SEQUENCE_LENGTH]
            X.append(tokens)
        
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        
        # Create the model
        self.model = SentimentLSTM(len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
        self.model.to(self.device)
        
        # Use class weights to handle imbalanced data
        # Calculate class weights (inverse of frequency)
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()  # Normalize
        class_weights = class_weights.to(self.device)
        
        # Use weighted CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Train the model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        self.model.train()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(EPOCHS):
            running_loss = 0.0
            
            # Create batches
            for i in range(0, len(X), BATCH_SIZE):
                batch_X = X[i:i+BATCH_SIZE].to(self.device)
                batch_y = y[i:i+BATCH_SIZE].to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            status_text.text(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")
            progress_bar.progress((epoch + 1) / EPOCHS)
            
        status_text.text("Training completed!")
        progress_bar.empty()

    # Modify the predict method to handle non-string inputs
    def predict(self, text):
        if self.model is None or self.vocab is None:
            return "Model not trained"
        
        # Handle non-string input
        if not isinstance(text, str):
            if pd.isna(text):
                text = ""
            else:
                text = str(text)
        
        # Text pipeline
        text_pipeline = lambda x: [self.vocab[token] for token in self.tokenizer(clean_text(x))]
        
        # Preprocess the text
        tokens = text_pipeline(text)
        if len(tokens) < MAX_SEQUENCE_LENGTH:
            tokens = tokens + [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
        else:
            tokens = tokens[:MAX_SEQUENCE_LENGTH]
        
        # Convert to tensor
        X = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
        
        predicted_class = predicted.item()
        sentiment = self.label_to_sentiment[predicted_class]
        
        return sentiment, probabilities

# Gemini API handler
class GeminiHandler:
    def __init__(self, api_key):
        # Configure Gemini API
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Set the model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_sentiment(self, text):
        try:
            # Prepare prompt for sentiment analysis
            prompt = f"""
            Analyze the sentiment of the following customer review. 
            Classify it as Positive, Neutral, or Negative. 
            Also provide a brief explanation of your reasoning.
            
            Review: "{text}"
            
            Return your response in JSON format with the following structure:
            {{
                "sentiment": "Positive/Neutral/Negative",
                "confidence": 0.XX,
                "explanation": "Your reasoning here"
            }}
            
            Ensure your response is properly formatted as a JSON object.
            """
            
            # Make the API request
            start_time = time.time()
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time
            
            content = response.text
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response if it's not pure JSON
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    result = json.loads(json_content)
                else:
                    raise json.JSONDecodeError("No JSON found", content, 0)
                    
            except json.JSONDecodeError:
                # Handle case where response is not valid JSON
                sentiment_match = re.search(r'"sentiment":\s*"([^"]+)"', content)
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', content)
                explanation_match = re.search(r'"explanation":\s*"([^"]+)"', content)
                
                sentiment = sentiment_match.group(1) if sentiment_match else "Unknown"
                confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                explanation = explanation_match.group(1) if explanation_match else "No explanation provided"
                
                result = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "explanation": explanation
                }
            
            result["response_time"] = response_time
            return result
            
        except Exception as e:
            return {
                "sentiment": "Error",
                "confidence": 0.0,
                "explanation": f"Exception: {str(e)}",
                "response_time": 0
            }

# Main Streamlit app
def main():
    st.title("📊 Sentiment Analysis Comparison: PyTorch vs. LLM (gemini)")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    # Initialize session state
    if "pytorch_analyzer" not in st.session_state:
        st.session_state.pytorch_analyzer = PyTorchSentimentAnalyzer()
    
    if "gemini_handler" not in st.session_state and api_key:
        st.session_state.gemini_handler = GeminiHandler(api_key)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Training", "Analysis", "Benchmark"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Using Google's Gemini LLM for sentiment analysis comparison.")
    
    # Tab 1: Training the PyTorch model
    with tab1:
        st.header("Train PyTorch Model")
        st.write("Upload a CSV file with review texts and sentiment labels to train the model.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Display dataframe preview
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Column selection
            text_column = st.selectbox("Select text column", df.columns.tolist())
            label_column = st.selectbox("Select label column (0: Negative, 1: Neutral, 2: Positive)", df.columns.tolist())
            
            if st.button("Train Model"):
                try:
                    # Prepare data
                    texts = df[text_column].tolist()
                    labels = df[label_column].astype(int).tolist()
                    
                    # Train the model
                    with st.spinner("Training the PyTorch model..."):
                        st.session_state.pytorch_analyzer.train(texts, labels)
                    
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        
        # Option to use sample data
        st.write("---")
        st.subheader("Or use sample data")
        
        if st.button("Train with sample data"):
            # Create sample data
            sample_texts = [
                "This product is amazing. I love it!",
                "Not bad, but could be better.",
                "Terrible experience. Never buying again.",
                "The quality is good, but the price is high.",
                "Absolutely fantastic service and product.",
                "Disappointed with the quality.",
                "Average product, nothing special.",
                "So happy with my purchase!",
                "Waste of money. Don't recommend.",
                "It's okay, I guess."
            ]
            
            sample_labels = [2, 1, 0, 1, 2, 0, 1, 2, 0, 1]  # 0: Negative, 1: Neutral, 2: Positive
            
            # Train the model
            with st.spinner("Training the PyTorch model with sample data..."):
                st.session_state.pytorch_analyzer.train(sample_texts, sample_labels)
            
            st.success("Model trained successfully with sample data!")
    
    # Tab 2: Analyzing reviews
    with tab2:
        st.header("Analyze Reviews")
        
        # Text input
        review_text = st.text_area("Enter a customer review:", "This product exceeded my expectations. The quality is excellent and customer service was top-notch!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PyTorch Analysis")
            if st.button("Analyze with PyTorch"):
                if "pytorch_analyzer" in st.session_state and st.session_state.pytorch_analyzer.model is not None:
                    with st.spinner("Analyzing with PyTorch..."):
                        start_time = time.time()
                        sentiment, probabilities = st.session_state.pytorch_analyzer.predict(review_text)
                        pytorch_time = time.time() - start_time
                    
                    st.write(f"**Sentiment:** {sentiment}")
                    st.write(f"**Response Time:** {pytorch_time:.3f} seconds")
                    
                    # Display probabilities
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sentiments = ["Negative", "Neutral", "Positive"]
                    ax.bar(sentiments, probabilities)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability")
                    ax.set_title("Sentiment Probabilities")
                    st.pyplot(fig)
                else:
                    st.warning("Please train the PyTorch model first!")
        
        with col2:
            st.subheader("Gemini Analysis")
            if st.button("Analyze with Gemini"):
                if api_key:
                    if "gemini_handler" not in st.session_state:
                        st.session_state.gemini_handler = GeminiHandler(api_key)
                    
                    with st.spinner("Analyzing with Gemini..."):
                        result = st.session_state.gemini_handler.analyze_sentiment(review_text)
                    
                    st.write(f"**Sentiment:** {result['sentiment']}")
                    st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")
                    st.write(f"**Response Time:** {result.get('response_time', 0):.3f} seconds")
                    st.write("**Explanation:**")
                    st.write(result.get('explanation', 'No explanation provided'))
                else:
                    st.warning("Please enter your Gemini API key in the sidebar!")
    
    # Tab 3: Benchmark
    with tab3:
        st.header("Benchmark Comparison")
        st.write("Compare the performance of PyTorch and GPT-4o-mini on multiple reviews.")
        
        # Sample reviews for benchmarking
        sample_reviews = [
            "I'm extremely disappointed with this product. It broke after just two days of use.",
            "The service was okay. Nothing special but no major issues either.",
            "This is the best purchase I've made all year! Highly recommend to everyone.",
            "While the product functions as advertised, the price point seems a bit high for what you get.",
            "I'm returning this immediately. Terrible quality and the customer service was unhelpful."
        ]
        
        # Custom reviews input
        st.subheader("Custom benchmark reviews")
        custom_reviews = st.text_area("Enter multiple reviews (one per line):", "")
        
        if custom_reviews:
            benchmark_reviews = custom_reviews.split("\n")
        else:
            benchmark_reviews = sample_reviews
        
        if st.button("Run Benchmark"):
            if "pytorch_analyzer" in st.session_state and st.session_state.pytorch_analyzer.model is not None and api_key:
                if "gemini_handler" not in st.session_state:
                    st.session_state.gemini_handler = GeminiHandler(api_key)
                
                results = []
                
                with st.spinner("Running benchmark..."):
                    for i, review in enumerate(benchmark_reviews):
                        if not review.strip():
                            continue
                        
                        # PyTorch prediction
                        start_time = time.time()
                        sentiment, _ = st.session_state.pytorch_analyzer.predict(review)
                        pytorch_time = time.time() - start_time
                        
                        # Gemini prediction
                        gemini_result = st.session_state.gemini_handler.analyze_sentiment(review)
                        
                        results.append({
                            "Review": review,
                            "PyTorch Sentiment": sentiment,
                            "PyTorch Time (s)": pytorch_time,
                            "Gemini Sentiment": gemini_result.get("sentiment", "Error"),
                            "Gemini Time (s)": gemini_result.get("response_time", 0),
                            "Gemini Explanation": gemini_result.get("explanation", "N/A")
                        })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("Benchmark Results")
                st.dataframe(results_df[["Review", "PyTorch Sentiment", "PyTorch Time (s)", "Gemini Sentiment", "Gemini Time (s)"]])
                
                # Compare timing
                avg_pytorch_time = results_df["PyTorch Time (s)"].mean()
                avg_gemini_time = results_df["Gemini Time (s)"].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Average Response Time")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.bar(["PyTorch", "Gemini"], [avg_pytorch_time, avg_gemini_time])
                    ax.set_ylabel("Time (seconds)")
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Key Differences")
                    st.write("**PyTorch:**")
                    st.write("- Faster inference")
                    st.write("- Simple classification only")
                    st.write("- Local computation")
                    
                    st.write("**Gemini:**")
                    st.write("- Includes explanation")
                    st.write("- Understands nuance")
                    st.write("- No training needed")
                
                # Detailed explanations
                st.subheader("Gemini Explanations")
                for i, row in enumerate(results):
                    with st.expander(f"Review {i+1}: {row['Review'][:50]}..."):
                        st.write(f"**PyTorch:** {row['PyTorch Sentiment']}")
                        st.write(f"**Gemini:** {row['Gemini Sentiment']}")
                        st.write(f"**Explanation:** {row['Gemini Explanation']}")
            else:
                if "pytorch_analyzer" not in st.session_state or st.session_state.pytorch_analyzer.model is None:
                    st.warning("Please train the PyTorch model first!")
                if not api_key:
                    st.warning("Please enter your Gemini API key in the sidebar!")
    
    # Footer
    st.write("---")
    st.write("Sentiment Analysis Comparison | PyTorch vs. Gemini")

if __name__ == "__main__":
    main()