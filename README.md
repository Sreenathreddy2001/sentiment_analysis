---
title: Sentiment Analysis
emoji: 🐨
colorFrom: indigo
colorTo: indigo
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
short_description: 'summarizes the news of the stocks  '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# Company News Analyzer with Text-to-Speech

This application extracts news articles about a company from Yahoo Finance, performs sentiment analysis, compares the sentiment across articles, and provides a text-to-speech summary in multiple languages, including Hindi.

## Features

- **News Extraction**: Fetches news articles related to a company using web scraping with BeautifulSoup
- **Sentiment Analysis**: Analyzes the sentiment of each article (positive, negative, neutral) using NLTK's VADER
- **Topic Extraction**: Identifies key topics in each article using TF-IDF
- **Comparative Analysis**: Compares sentiment across articles and identifies common topics
- **Multi-language Text-to-Speech**: Converts the analysis summary to speech in multiple languages
- **User Interface**: Simple Streamlit web interface for easy interaction

## Architecture

The application is built with the following components:

1. **Streamlit Frontend**: User-friendly interface for inputting company ticker symbols and displaying results
2. **API Layer**: Core functionality for fetching and processing news articles
3. **News Scraper**: Extracts news articles from Yahoo Finance using BeautifulSoup
4. **Sentiment Analyzer**: Uses NLTK's VADER for sentiment analysis
5. **Topic Extractor**: Implements TF-IDF to identify key topics
6. **Text-to-Speech Service**: Converts text to speech in multiple languages using gTTS

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sreenath2001/sentiment_analysis/company-news-analyzer.git
   cd company-news-analyzer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK resources (first time only):
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Usage

### Running Locally

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Enter a company ticker symbol (e.g., AAPL for Apple Inc.) and click "Generate Report" to analyze news articles.

4. Select your preferred language for the text-to-speech output.

### Using the API Directly

You can also use the API functions directly in your Python code:

```python
import asyncio
from api import get_news_and_generate_speech

# Fetch news for Apple Inc. with Hindi speech output
results = asyncio.run(get_news_and_generate_speech("AAPL", output_language='hi', output_file="apple_news.mp3"))

# Print the articles
for i, article in enumerate(results['articles']):
    print(f"\nArticle {i+1}:")
    for key, value in article.items():
        print(f"{key}: {value}")
    print("-" * 50)

# Print summary and audio path
print(f"Summary: {results['summary_total']}")
print(f"Audio saved as: {results['audio_path']}")
```

## Deployment

The application can be deployed on Hugging Face Spaces for easy access.

### Deployment Steps

1. Create a Hugging Face account if you don't have one
2. Create a new Space on Hugging Face Spaces
3. Select Streamlit as the SDK
4. Upload your code to the Space
5. The Space will automatically build and deploy your application

## Models and Libraries Used

- **Sentiment Analysis**: NLTK's VADER (Valence Aware Dictionary and Sentiment Reasoner)
- **Topic Extraction**: Scikit-learn's TF-IDF Vectorizer
- **Text Summarization**: Hugging Face's Transformers with facebook/bart-large-cnn model
- **Translation**: Google Translate API via googletrans library
- **Text-to-Speech**: Google Text-to-Speech (gTTS) for multi-language speech synthesis

## Limitations and Assumptions

- The application scrapes Yahoo Finance for news, which may change its HTML structure over time
- Sentiment analysis is performed using VADER, which is optimized for social media text but works reasonably well for news
- The application assumes internet connectivity to access Yahoo Finance and use the text-to-speech service
- Translation quality depends on the Google Translate API
- The summarization model has token limitations, so very long articles may be truncated
- The application currently supports a limited set of languages for text-to-speech

## Future Improvements

- Add support for additional news sources
- Implement more sophisticated sentiment analysis with fine-tuned models
- Add historical sentiment tracking over time
- Integrate with finance APIs to correlate news sentiment with stock price movements
- Improve error handling and fallback mechanisms
- Add user accounts and the ability to save analysis results

## API Reference

### `get_news_and_generate_speech(company, output_language='en', output_file="news_summary.mp3")`

Fetches and analyzes news for a company, and generates speech output.

Parameters:
- `company` (str): The company ticker symbol (e.g., "AAPL" for Apple)
- `output_language` (str): Language code for text-to-speech (default: 'en')
- `output_file` (str): Filename for the generated audio file

Returns:
- dict: Results including articles, summary, and audio path

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.