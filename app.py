import streamlit as st
import asyncio
from api import fetch_company_news
import utils
import base64
from io import BytesIO

st.set_page_config(page_title="Financial News Summarizer", layout="wide")

def get_audio_player(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return st.audio(audio_bytes, format="audio/mp3")

def main():
    st.title("Financial News Summarizer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        This app fetches the latest financial news for a company, performs sentiment analysis, 
        extracts key topics, and provides a summarized version in your chosen language.
        """)
    
    with col2:
        st.image("https://via.placeholder.com/150", caption="News AI")
    
    ticker = st.text_input("Enter company ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    language_options = {
        "English": "en", 
        "Hindi": "hi"
    }
    
    selected_language = st.selectbox("Select output language", list(language_options.keys()))
    
    if st.button("Get News"):
        if not ticker:
            st.warning("Please enter a ticker symbol")
            return
            
        with st.spinner("Fetching and analyzing news..."):
            try:
                results = asyncio.run(fetch_company_news(
                    ticker, 
                    output_language=language_options[selected_language],
                    output_file=f"{ticker}_news.mp3"
                ))
                
                if not results or 'articles' not in results:
                    st.error("Failed to fetch news or no news found.")
                    return
                
                # Display sentiment distribution
                st.subheader("Sentiment Distribution")
                utils.display_sentiment_chart(results['report']['sentiment_distribution'])
                
                # Display common topics
                st.subheader("Common Topics")
                st.write(", ".join(results['report']['common_topics']))
                
                # Display news summary
                st.subheader("News Summary")
                st.write(results['summary_total'])
                
                # Audio playback
                if results['audio_path']:
                    st.subheader(f"Audio Summary in {selected_language}")
                    get_audio_player(results['audio_path'])
                    
                    # Download button for audio
                    with open(results['audio_path'], "rb") as file:
                        btn = st.download_button(
                            label=f"Download {selected_language} Audio Summary",
                            data=file,
                            file_name=f"{ticker}_news.mp3",
                            mime="audio/mp3"
                        )
                
                # Display individual articles
                st.subheader("Individual News Articles")
                for i, article in enumerate(results['articles']):
                    with st.expander(f"Article {i+1}: {article['Title']}"):
                        st.write(f"**Summary:** {article['summary']}")
                        st.write(f"**Sentiment:** {article['sentiment']}")
                        st.write(f"**Key topics:** {', '.join(article['Key topics'])}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()