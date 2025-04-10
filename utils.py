import streamlit as st
import pandas as pd
import nltk
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Download required NLTK packages if they don't exist"""
    try:
        # Check if we have the VADER lexicon
        nltk.data.find('sentiment/vader_lexicon.zip')
        logger.info("NLTK VADER lexicon already downloaded")
    except LookupError:
        logger.info("Downloading NLTK VADER lexicon")
        nltk.download('vader_lexicon')
        
    try:
        # Check if we have the punkt tokenizer
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer already downloaded")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')
         
def save_to_report(company, results, output_format='csv'):
    """
    Saves analysis results to a file
    
    Args:
        company (str): Company ticker
        results (dict): Analysis results
        output_format (str): 'csv' or 'json'
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        df = pd.DataFrame(results['articles'])
        
        if output_format.lower() == 'csv':
            output_path = f"reports/{company}_news_analysis.csv"
            df.to_csv(output_path, index=False)
        else:
            output_path = f"reports/{company}_news_analysis.json"
            df.to_json(output_path, orient='records')
            
        return output_path
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return None

def display_sentiment_chart(sentiment_distribution):
    """
    Displays a pie chart of sentiment distribution using Streamlit
    
    Args:
        sentiment_distribution (dict): Dictionary containing sentiment counts
    """
    try:
        import plotly.express as px
        
        # Convert sentiment distribution to DataFrame
        df = pd.DataFrame({
            'Sentiment': list(sentiment_distribution.keys()),
            'Count': list(sentiment_distribution.values())
        })
        
        # Create pie chart
        fig = px.pie(df, values='Count', names='Sentiment', 
                    title='Sentiment Distribution',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': 'green',
                        'Negative': 'red',
                        'Neutral': 'gray'
                    })
        
        # Display the chart
        st.plotly_chart(fig)
        
    except Exception as e:
        logger.error(f"Error displaying sentiment chart: {str(e)}")
        st.error("Failed to display sentiment chart")
