from datetime import datetime, timedelta
import os
import time
import logging
import requests

from dotenv import load_dotenv
import pandas as pd

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class CoindeskArticleFetcher:
    
    def __init__(self):
        self.base_url = "https://data-api.coindesk.com/news/v1/article/list"
        self.api_key = os.getenv("COINDESK_API_KEY")
        self.dataset_path = "crypto_sentiment_dataset.csv"
        
    def check_api_key(self):
        return self.api_key is not None
    
    def fetch_articles(self, days_back=30, limit=100, offset=0):
        if not self.check_api_key():
            return []
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logging.info(f"Fetching articles from {start_date_str} to {end_date_str}")
        
        try:
            response = requests.get(
                self.base_url,
                params={
                    "startDate": start_date_str,
                    "endDate": end_date_str,
                    "limit": limit,
                    "offset": offset,
                    "includeSentiment": "true",
                    "format": "json"
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("Data", [])
                logging.info(f"Successfully fetched {len(articles)} articles")
                return articles
            else:
                logging.error(f"API request failed with status code {response.status_code}")
                logging.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logging.error(f"Error fetching articles: {str(e)}")
            return []
    
    def fetch_all_articles(self, days_back=30, batch_size=100, max_articles=1000):
        all_articles = []
        offset = 0
        
        while len(all_articles) < max_articles:
            batch = self.fetch_articles(days_back, batch_size, offset)
            if not batch:
                break
                
            all_articles.extend(batch)
            offset += batch_size
            
            time.sleep(1) # avoid hitting rate limits
            logging.info(f"Fetched {len(all_articles)} articles so far")
            
            # fewer articles than requested == reached the end
            if len(batch) < batch_size:
                break
        
        return all_articles[:max_articles]
    
    def process_article_data(self, articles):
        processed_data = []
        
        for article in articles:
            try:
                # Convert UNIX timestamp to readable date format and
                # extract relevant fields according to the API schema
                published_date = datetime.fromtimestamp(article.get("PUBLISHED_ON", 0))
                processed_article = {
                    "id": article.get("ID"),
                    "guid": article.get("GUID"),
                    "title": article.get("TITLE"),
                    "subtitle": article.get("SUBTITLE"),
                    "content": article.get("BODY", ""),
                    "published_date": published_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "published_timestamp": article.get("PUBLISHED_ON"),
                    "url": article.get("URL"),
                    "image_url": article.get("IMAGE_URL"),
                    "authors": article.get("AUTHORS", ""),
                    "source_id": article.get("SOURCE_ID"),
                    "keywords": article.get("KEYWORDS", ""),
                    "language": article.get("LANG", "EN"),
                    "upvotes": article.get("UPVOTES", 0),
                    "downvotes": article.get("DOWNVOTES", 0),
                    "score": article.get("SCORE", 0),
                    "sentiment": article.get("SENTIMENT", "NEUTRAL"),
                    "status": article.get("STATUS", "ACTIVE"),
                }
                processed_data.append(processed_article)
            except Exception as e:
                logging.error(f"Error processing article: {str(e)}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def save_dataset(self, df, output_path=None):
        if output_path is None:
            output_path = self.dataset_path
            
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset saved to {output_path}")

    
    def create_dataset(self, days_back=90, max_articles=5000):
        articles = self.fetch_all_articles(days_back, max_articles=max_articles)
        
        if not articles:
            logging.error("No articles fetched, cannot create dataset")
            return None
            
        df = self.process_article_data(articles)
        self.save_dataset(df)
        
        sentiment_counts = df["sentiment"].value_counts()
        logging.info(f"Dataset sentiment distribution: {sentiment_counts.to_dict()}")
        
        return df


def main():
    logging.info("Starting CoinDesk article fetcher for sentiment analysis dataset...")
    
    fetcher = CoindeskArticleFetcher()
    if not fetcher.check_api_key():
        logging.error("API key is required to fetch articles. Please set the COINDESK_API_KEY environment variable.")
        return
    
    dataset = fetcher.create_dataset(days_back=7, max_articles=10)
    
    if dataset is not None:
        logging.info(f"Successfully created dataset with {len(dataset)} articles")
        logging.info(f"Dataset saved to {fetcher.dataset_path}")
        
        sentiment_counts = dataset["sentiment"].value_counts()
        logging.info(f"Sentiment distribution in dataset:")
        for sentiment, count in sentiment_counts.items():
            logging.info(f"  {sentiment}: {count} articles ({count/len(dataset)*100:.1f}%)")
    else:
        logging.error("Failed to create dataset")


if __name__ == "__main__":
    main() 