# Sentiment Analysis Comparison: PyTorch vs. Gemini

This Streamlit application demonstrates the differences between traditional machine learning approaches (PyTorch) and large language models (Google's Gemini) for sentiment analysis on customer reviews.

## Features

- **Train a PyTorch LSTM model** on your own dataset or use sample data
- **Analyze reviews** using both PyTorch and Google's Gemini
- **Benchmark** both approaches on multiple reviews
- **Compare results** including sentiment classifications, response times, and explanations

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the application**

```bash
streamlit run app.py
```

3. **Generate sample data** (optional)

```bash
python generate_sample_data.py
```

## How to Use

### 1. Training the PyTorch Model

- Upload a CSV file with review texts and sentiment labels (0: Negative, 1: Neutral, 2: Positive)
- Select the appropriate columns for text and labels
- Click "Train Model"
- Alternatively, use the sample data by clicking "Train with sample data"

### 2. Analyzing Reviews

- Enter a customer review in the text area
- Click "Analyze with PyTorch" to see the PyTorch model's prediction
- Enter your Gemini API key in the sidebar
- Click "Analyze with Gemini" to see the LLM's prediction and explanation

### 3. Benchmarking

- Use the provided sample reviews or enter your own (one per line)
- Click "Run Benchmark" to compare both approaches on multiple reviews
- View average response times and detailed explanations

## Key Differences Highlighted

### PyTorch Approach

- Requires training data
- Faster inference (typically milliseconds)
- Simple classification without explanations
- Runs locally without API calls
- Limited to exactly what it was trained on

### Gemini Approach

- No training needed (zero-shot capability)
- Provides explanations for decisions
- Understands nuance and context
- Typically slower (API call latency)
- More flexible and adaptable to different phrasing

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Gemini API Key

You'll need a Google Gemini API key to use the Gemini functionality. Enter it in the sidebar when prompted. You can get a Gemini API key by signing up at the [Google AI Studio](https://makersuite.google.com/).
