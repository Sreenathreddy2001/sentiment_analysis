import streamlit as st
import asyncio
import nest_asyncio
import time
import os
from api import get_news_and_generate_speech
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .success-text {
        color: #4CAF50;
    }
    .warning-text {
        color: #FF9800;
    }
    .error-text {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>News Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Analyze news articles for any company and get sentiment analysis, key topics, and audio summaries.</p>", unsafe_allow_html=True)

# User inputs section without container display
st.markdown("<h2 class='sub-header'>stock</h2>", unsafe_allow_html=True)

# Create columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    # Company ticker input
    company_ticker = st.text_input("Company Ticker Symbol", value="AAPL")

with col2:
    # Language selection
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Chinese": "zh"
    }
    selected_language = st.selectbox("Output Language", options=list(language_options.keys()))
    output_language = language_options[selected_language]

with col3:
    # Output file name
    output_file = st.text_input("Output Audio File", value=f"{company_ticker.lower()}_news.mp3")

# Analyze button
analyze_button = st.button("Analyze News", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    with st.spinner("Analyzing news articles... This may take a moment."):
        start_time = time.time()
        
        # Run the analysis
        results = asyncio.run(get_news_and_generate_speech(
            company=company_ticker, 
            output_language=output_language, 
            output_file=output_file
        ))
        
        if not results:
            st.error("Failed to get results. Please check your internet connection and try again.")
        else:
            # Display execution time
            execution_time = time.time() - start_time
            st.success(f"Analysis completed in {execution_time:.2f} seconds")
            
            # Display articles
            st.markdown("<h2 class='sub-header'>News Articles</h2>", unsafe_allow_html=True)
            
            list_articles = results['articles']
            st.info(f"Found {len(list_articles)} articles")
            
            # Create a DataFrame for better display
            articles_df = pd.DataFrame(list_articles)
            
            # Display articles in an expandable section
            with st.expander("View Articles", expanded=True):
                for i, article in enumerate(list_articles):
                    st.markdown(f"**Article {i+1}:**")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Title:** {article['Title']}")
                        st.markdown(f"**Summary:** {article['summary']}")
                    
                    with col2:
                        # Color-coded sentiment
                        sentiment = article['sentiment']
                        if sentiment == 'positive':
                            st.markdown(f"<p class='success-text'><strong>Sentiment:</strong> {sentiment.capitalize()}</p>", unsafe_allow_html=True)
                        elif sentiment == 'negative':
                            st.markdown(f"<p class='error-text'><strong>Sentiment:</strong> {sentiment.capitalize()}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='warning-text'><strong>Sentiment:</strong> {sentiment.capitalize()}</p>", unsafe_allow_html=True)
                        
                        st.markdown("**Key Topics:**")
                        for topic in article['Key topics']:
                            st.markdown(f"- {topic}")
                    
                    st.markdown("---")
            
            # Display summary
            st.markdown("<h2 class='sub-header'>Summary</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='info-text'>{results['summary_total']}</p>", unsafe_allow_html=True)
            
            # Display audio player if audio was generated
            if results['audio_path'] and os.path.exists(results['audio_path']):
                st.markdown("<h2 class='sub-header'>Audio Summary</h2>", unsafe_allow_html=True)
                st.audio(results['audio_path'])
                st.markdown(f"<p class='info-text'>Audio saved as: {results['audio_path']}</p>", unsafe_allow_html=True)
            
            # Display sentiment distribution
            st.markdown("<h2 class='sub-header'>Sentiment Distribution</h2>", unsafe_allow_html=True)
            
            sentiment_dist = results['report']['sentiment_distribution']
            
            # Create a pie chart for sentiment distribution
            if sentiment_dist:
                sentiment_df = pd.DataFrame({
                    'Sentiment': list(sentiment_dist.keys()),
                    'Count': list(sentiment_dist.values())
                })
                
                fig = px.pie(
                    sentiment_df, 
                    values='Count', 
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={
                        'positive': '#4CAF50',
                        'negative': '#F44336',
                        'neutral': '#FF9800'
                    },
                    title='Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display common topics
            st.markdown("<h2 class='sub-header'>Common Topics</h2>", unsafe_allow_html=True)
            
            common_topics = results['report']['common_topics']
            if common_topics:
                st.markdown("<p class='info-text'>Common topics across articles:</p>", unsafe_allow_html=True)
                for topic in common_topics:
                    st.markdown(f"- {topic}")
            else:
                st.info("No common topics found across articles.")

# Footer
st.markdown("---")
st.markdown("<p class='info-text' style='text-align: center;'>Created with Streamlit | News Sentiment Analysis</p>", unsafe_allow_html=True) 
