import streamlit as st
import matplotlib.pyplot as plt
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

def display_sentiment_chart(sentiment_distribution):
    """
    Displays a pie chart of sentiment distribution
    
    Args:
        sentiment_distribution (dict): A dictionary of sentiment counts
    """
    if not sentiment_distribution:
        st.warning("No sentiment data available.")
        return
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Sentiment': list(sentiment_distribution.keys()),
        'Count': list(sentiment_distribution.values())
    })
    
    # Prepare colors for sentiment
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    color_map = [colors.get(s, 'blue') for s in df['Sentiment']]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        df['Count'], 
        labels=df['Sentiment'],
        autopct='%1.1f%%',
        startangle=90,
        colors=color_map
    )
    
    # Customize text appearance
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
    
    ax.set_title('Sentiment Distribution', fontsize=14)
    st.pyplot(fig)

def create_wordcloud(common_topics):
    """
    Creates a word cloud visualization from common topics
    
    Args:
        common_topics (list): List of common topics
    """
    if not common_topics:
        st.warning("No common topics available for visualization.")
        return
    
    try:
        from wordcloud import WordCloud
        
        # Create a frequency dictionary
        topic_freq = {}
        for topic in common_topics:
            if topic in topic_freq:
                topic_freq[topic] += 1
            else:
                topic_freq[topic] = 1
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_freq)
        
        # Display the wordcloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ImportError:
        st.warning("WordCloud package not installed. Install with 'pip install wordcloud'")
        # Fallback to simple bar chart
        topic_counts = pd.Series(common_topics).value_counts()
        st.bar_chart(topic_counts)

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