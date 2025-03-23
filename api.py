import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from transformers import pipeline
from gtts import gTTS
from googletrans import Translator
import asyncio
import nest_asyncio
import os

# Apply nest_asyncio to allow nested event loops in environments like Jupyter/Streamlit
nest_asyncio.apply()

async def fetch_company_news(company, output_language='en', output_file="news_summary.mp3"):
    """
    Fetches news about a company, performs analysis, and generates a speech summary.
    
    Args:
        company (str): The company ticker symbol (e.g., 'AAPL')
        output_language (str): Language code for the output speech (default: 'en')
        output_file (str): Filename for the output audio file
        
    Returns:
        dict: Contains analysis results, articles, summary and audio path
    """
    try:
        # Fetch news articles
        url = f"https://finance.yahoo.com/quote/{company}/news/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch news articles. Status code: {response.status_code}")
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        sentiment_analyser = SentimentIntensityAnalyzer()

        # Extract titles, summaries, and paragraphs
        paragraphs = []
        titles = []
        summaries = []
        article = []

        for news in soup.find_all("div", class_="holder yf-1napat3"):
            title_all = news.find_all('h3', class_="clamp yf-82qtw3")
            summary_all = news.find_all('p', class_="clamp yf-82qtw3")
            for title, summary in zip(title_all, summary_all):
                title_text = title.get_text()
                summary_text = summary.get_text()
                paragraph = title_text + ' ' + summary_text
                titles.append(title_text)
                summaries.append(summary_text)
                paragraphs.append(paragraph)

        if not paragraphs:
            print("No news articles found.")
            return {}

        # Perform batch TF-IDF analysis
        vectoriser = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = vectoriser.fit_transform(paragraphs)
        keywords_list = vectoriser.get_feature_names_out()

        # Analyze sentiment and prepare the output
        for i, paragraph in enumerate(paragraphs):
            scores = sentiment_analyser.polarity_scores(paragraph)['compound']
            if scores >= 0.05:
                sentiment = 'positive'
            elif scores <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            # Collect keywords for the paragraph
            tfidf_scores = tfidf_matrix[i].toarray().flatten()
            keywords = [keywords_list[j] for j in tfidf_scores.argsort()[-5:][::-1]]  # Top 5 keywords

            article.append({
                "Title": titles[i],
                "summary": summaries[i],
                "sentiment": sentiment,
                "Key topics": keywords
            })

        # Create a DataFrame after all articles are processed
        df = pd.DataFrame(article)

        # Sentiment distribution
        sentiment_dist = df['sentiment'].value_counts().to_dict()

        # Compare articles for common topics
        report = {
            "sentiment_distribution": sentiment_dist,
            "coverage_differences": [],
            "common_topics": []
        }

        for i, row in df.iterrows():
            topics = set(row['Key topics'])
            for j, comp_row in df.iterrows():
                if i != j:
                    comp_topics = set(comp_row['Key topics'])
                    overlap = topics.intersection(comp_topics)
                    if overlap:
                        report["common_topics"].extend(list(overlap))

        common_topics_flat = pd.Series(report["common_topics"])
        common_topics_count = common_topics_flat.value_counts()
        report["common_topics"] = common_topics_count.head(5).index.tolist() if not common_topics_count.empty else []
        
        # Summarize the articles
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Join the title and summary into a single string for summarization, limiting to the first 5 articles
        text_to_summarize = " ".join([d['Title'] + " " + d['summary'] for d in article[:5]])
        
        try:
            summary_final = summarizer(text_to_summarize, min_length=50, do_sample=False, truncation=True)[0]['summary_text']
        except Exception as e:
            # If there's an error, try a shorter version
            text_to_summarize = text_to_summarize[:1024]  # Limit to 1024 tokens
            summary_final = summarizer(text_to_summarize, min_length=50, do_sample=False, truncation=True)[0]['summary_text']
        
        # Text-to-speech with language translation if needed
        audio_path = None
        
        try:
            if output_language != 'en':
                # Translate the summary to the target language
                translator = Translator()
                translated = translator.translate(summary_final, src='en', dest=output_language)
                translated_text = translated.text
                
                # Generate speech in the translated language
                tts = gTTS(text=translated_text, lang=output_language)
                tts.save(output_file)
                audio_path = output_file
            else:
                # For English, use the original text with gTTS
                tts = gTTS(text=summary_final, lang='en')
                tts.save(output_file)
                audio_path = output_file
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            audio_path = None
        
        return {
            "articles": article[:5], 
            "summary_total": summary_final,
            "audio_path": audio_path,
            "report": report
        }
    except Exception as e:
        print(f"Error in fetch_company_news: {str(e)}")
        return {}