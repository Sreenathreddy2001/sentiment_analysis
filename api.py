import asyncio
import nest_asyncio
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gtts import gTTS
from googletrans import Translator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def get_news_and_generate_speech(company, output_language='en', output_file="news_summary.mp3"):
    try:
        logger.info(f"Fetching news for {company}")
        # Fetch news articles
        url = f"https://finance.yahoo.com/quote/{company}/news/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logger.error(f"Failed to fetch news articles. Status code: {response.status_code}")
            return {}

        logger.info("Parsing HTML content")
        soup = BeautifulSoup(response.content, 'html.parser')
        sentiment_analyser = SentimentIntensityAnalyzer()

        # Extract titles, summaries, and paragraphs
        paragraphs = []
        titles = []
        summaries = []
        articles = []  # Reset the articles list for each call

        logger.info("Extracting news articles")
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
            logger.warning("No articles found")
            return {}

        logger.info(f"Found {len(paragraphs)} articles")

        # Perform batch TF-IDF analysis
        logger.info("Performing TF-IDF analysis")
        vectoriser = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = vectoriser.fit_transform(paragraphs)
        keywords_list = vectoriser.get_feature_names_out()

        # Analyze sentiment and prepare the output
        logger.info("Analyzing sentiment")
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

            articles.append({
                "Title": titles[i],
                "summary": summaries[i],
                "sentiment": sentiment,
                "Key topics": keywords
            })

        # Create a DataFrame after all articles are processed
        df = pd.DataFrame(articles)

        # Sentiment distribution
        sentiment_dist = df['sentiment'].value_counts().to_dict()

        # Compare articles for common topics
        logger.info("Analyzing common topics")
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

        # Use the first article's summary as the overall summary
        summary_final = articles[0]['summary'] if articles else "No summary available"

        # Text-to-speech with language translation if needed
        audio_path = None

        try:
            logger.info(f"Generating audio in {output_language}")
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
            logger.error(f"Error in text-to-speech: {str(e)}")
            audio_path = None

        logger.info("Analysis completed successfully")
        return {
            "articles": articles[:5],
            "summary_total": summary_final,
            "audio_path": audio_path,
            "report": report
        }

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return {}

# Run the function and display the results
async def main():
    print("Fetching news for AAPL and generating Hindi speech...")
    results = await get_news_and_generate_speech("AAPL", output_language='hi', output_file="apple_news.mp3")
    
    if not results:
        print("Failed to get results. Please check your internet connection and try again.")
        return
    
    list_articles = results.get('articles', [])
    
    for i, article in enumerate(list_articles):
        print(f"\nArticle {i+1}:")
        for key, value in article.items():
            print(f"{key}: {value}")
        print("-" * 50)
    
    print(f"Summary: {results.get('summary_total', 'No summary available')}")
    print(f"Audio saved as: {results.get('audio_path', 'No audio file created')}")
    print("\nCommon Topics across articles:", ", ".join(results.get('report', {}).get('common_topics', [])))
    print("Sentiment Distribution:", results.get('report', {}).get('sentiment_distribution', {}))

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
