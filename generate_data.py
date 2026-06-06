import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass
import trafilatura
import requests
import json
from datetime import datetime, timedelta
import time
import os
import re
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient, UpdateOne
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()


def get_past_month(reference_date=None):
    """Return (year, month) for the calendar month before reference_date."""
    reference_date = reference_date or datetime.utcnow()
    first_of_current = datetime(reference_date.year, reference_date.month, 1)
    last_of_previous = first_of_current - timedelta(days=1)
    return last_of_previous.year, last_of_previous.month


def month_date_range(year, month):
    """Return inclusive start/end datetimes for a calendar month."""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    end_date = end_date.replace(hour=23, minute=59, second=59)
    return start_date, end_date


CATEGORY_ALIASES = {
    "technology": {"technology", "tech", "gadgets", "science", "ai", "artificial intelligence", "startup", "cyber", "software", "app"},
    "sports": {"sports", "sport", "cricket", "football", "tennis", "ipl", "match", "wicket", "goal", "team", "player"},
    "entertainment": {"entertainment", "movies", "movie", "film", "films", "bollywood", "hollywood", "music", "actor", "actress", "ott"},
    "business": {"business", "markets", "market", "economy", "finance", "stocks", "stock", "money", "bank", "rbi", "profit", "shares"},
    "health": {"health", "wellness", "medical", "medicine", "fitness", "doctor", "hospital", "disease", "covid", "drug"},
    "india": {"india", "national", "domestic", "delhi", "mumbai", "bengaluru", "state", "court", "police"},
    "politics": {"politics", "political", "elections", "election", "government", "bjp", "congress", "minister", "cm", "pm", "party"},
}


DOMAIN_CATEGORY_HINTS = {
    "economictimes": "business",
    "financialexpress": "business",
    "business-standard": "business",
    "livemint": "business",
    "espncricinfo": "sports",
    "sports.ndtv": "sports",
    "cricbuzz": "sports",
    "bollywood": "entertainment",
}


def normalize_category(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if not value or value in {"none", "null", "nan", "uncategorized"}:
        return None
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    for canonical, aliases in CATEGORY_ALIASES.items():
        if value == canonical or value in aliases:
            return canonical
        if any(alias in value.split() for alias in aliases):
            return canonical
    return None


def infer_article_category(article):
    category = normalize_category(article.get("category"))
    if category:
        return category

    domain_url = " ".join(
        str(article.get(field) or "").lower()
        for field in ("domain_name", "url")
    )
    for hint, canonical in DOMAIN_CATEGORY_HINTS.items():
        if hint in domain_url:
            return canonical

    title = str(article.get("title") or "").lower()
    haystack = " ".join(
        str(article.get(field) or "")
        for field in ("title", "description", "domain_name", "url", "content")
    ).lower()
    keyword_scores = {
        canonical: sum(
            2 if alias in title else 1
            for alias in aliases
            if re.search(rf"\b{re.escape(alias)}\b", haystack)
        )
        for canonical, aliases in CATEGORY_ALIASES.items()
    }
    best_category, best_score = max(keyword_scores.items(), key=lambda item: item[1])
    return best_category if best_score > 0 else "india"


class IndiaCompleteCorpusBuilder:
    """
    Build a complete corpus of ALL Indian news for a given month.
    Stores in MongoDB with full content scraping.
    """

    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="news_db"):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        #setup mongodb
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db['final_dataset']
        
        #create indexes
        self.collection.create_index("url", unique=True)
        self.collection.create_index([("published", -1)])
        self.collection.create_index("domain_name")
        self.collection.create_index("country")
        
        print(f"[MongoDB] Connected to {db_name}.raw_articles")

    def build_past_month_corpus(self, output_dir="india_corpus", reference_date=None):
        """Build corpus for the previous calendar month relative to reference_date."""
        year, month = get_past_month(reference_date)
        return self.build_monthly_corpus(year, month, output_dir)

    def build_monthly_corpus(self, year, month, output_dir="india_corpus"):

        start_date, end_date = month_date_range(year, month)

        print("=" * 70)
        print(f"Building Complete Indian News Corpus: {year}-{month:02d}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print("=" * 70 + "\n")

        all_articles = []

        print("STEP 1: Collecting from RSS Feeds...")
        print("-" * 70)
        rss_articles = self._collect_all_rss(start_date, end_date)
        all_articles.extend(rss_articles)
        print(f"[OK] RSS Collection: {len(rss_articles)} articles in range\n")

        print("STEP 2: Collecting from GDELT (Historical)...")
        print("-" * 70)

        gdelt_articles = self._collect_gdelt_month(start_date, end_date)
        all_articles.extend(gdelt_articles)
        print(f"[OK] GDELT Collection: {len(gdelt_articles)} articles\n")

        print("STEP 3: Deduplicating and filtering by date...")
        print("-" * 70)
        unique_articles = self._deduplicate(all_articles)
        unique_articles = self._filter_by_date_range(unique_articles, start_date, end_date)
        print(f"[OK] Unique articles in range: {len(unique_articles)}\n")

        print("STEP 4: Scraping content from articles...")
        print("-" * 70)
        articles_with_content = self._scrape_content_batch(unique_articles)
        print(f"[OK] Content scraped for {len(articles_with_content)} articles\n")

        print("STEP 5: Saving to MongoDB...")
        print("-" * 70)
        inserted, updated = self._save_to_mongodb(articles_with_content)
        print(f"[OK] MongoDB: {inserted} inserted, {updated} updated\n")

        print("STEP 6: Saving backup files...")
        print("-" * 70)
        filename = f"india_complete_{year}_{month:02d}"
        self._save_corpus(articles_with_content, filename, output_dir)

        print("\n" + "=" * 70)
        print("CORPUS COMPLETE!")
        print(f"Total articles: {len(articles_with_content)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"MongoDB: {self.db.name}.{self.collection.name}")
        print(f"Backup: {output_dir}/")
        print("=" * 70)

        return articles_with_content

    def _normalize_datetime(self, dt):
        if dt is None:
            return None
        if getattr(dt, "tzinfo", None) is not None:
            return dt.replace(tzinfo=None)
        return dt

    def _filter_by_date_range(self, articles, start_date, end_date):
        filtered = []
        for article in articles:
            published = self._normalize_datetime(article.get("published"))
            if published and start_date <= published <= end_date:
                article["published"] = published
                filtered.append(article)
        return filtered

    def _collect_all_rss(self, start_date=None, end_date=None):
        """Collect from all Indian RSS feeds, optionally limited to a date range."""

        feeds = {
            'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
            'https://indianexpress.com/feed/',
            'https://www.thehindu.com/news/national/feeder/default.rss',
            'https://feeds.feedburner.com/ndtvnews-india-news',
            'https://www.indiatoday.in/rss/home',
            'https://www.news18.com/rss/india.xml',
            'https://zeenews.india.com/rss/india-national-news.xml',
            'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'https://www.livemint.com/rss/homepage',
            'https://www.business-standard.com/rss/home_page_top_stories.rss',
            'https://www.moneycontrol.com/rss/latestnews.xml',
            'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms',
            'https://www.bgr.in/feed/',
            'https://tech.hindustantimes.com/rss/tech/rssfeed.xml',
            'https://indianexpress.com/section/political-pulse/feed/',
            'https://timesofindia.indiatimes.com/rssfeeds/4719148.cms',
            'https://www.hindustantimes.com/feeds/rss/cricket/rssfeed.xml',
            'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms',
            'https://www.thehindu.com/news/cities/Delhi/feeder/default.rss',
            'https://www.thehindu.com/news/cities/mumbai/feeder/default.rss',
            'https://www.thehindu.com/news/cities/bangalore/feeder/default.rss',
        }

        articles = []

        def fetch_feed(feed_url):
            try:
                feed = feedparser.parse(feed_url)
                feed_articles = []

                for entry in feed.entries:
                    url = entry.get('link', 'N/A')
                    domain = urlparse(url).netloc if url != 'N/A' else 'Unknown'
                    
                    #extract category if available
                    category = None
                    if hasattr(entry, 'tags') and entry.tags:
                        category = entry.tags[0].get('term', None)
                    
                    published = self._normalize_datetime(
                        self._parse_date(entry.get('published', entry.get('updated', '')))
                    )

                    if start_date and end_date:
                        if not published or published < start_date or published > end_date:
                            continue

                    article = {
                        'title': entry.get('title', 'N/A'),
                        'url': url,
                        'domain_name': domain,
                        'published': published,
                        'content': '',
                        'lang': 'en',
                        'country': 'IN',
                        'category': category,
                        'collection_method': 'RSS',
                        'description': entry.get('summary', entry.get('description', ''))
                    }
                    feed_articles.append(article)

                print(f"[OK] {feed.feed.get('title', 'Unknown')}: {len(feed_articles)} articles")
                return feed_articles

            except Exception as e:
                print(f"[ERR] RSS error in {feed_url}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_feed, feed) for feed in feeds]
            for future in as_completed(futures):
                articles.extend(future.result())

        return articles

    def _collect_gdelt_month(self, start_date, end_date):
        """Collect GDELT articles for the entire month."""
        all_articles = []
        current_date = start_date

        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)

            params = {
                'query': 'sourcecountry:IN',
                'mode': 'artlist',
                'maxrecords': 250,
                'startdatetime': current_date.strftime('%Y%m%d%H%M%S'),
                'enddatetime': next_date.strftime('%Y%m%d%H%M%S'),
                'format': 'json'
            }

            retries = 0
            max_retries = 3
            day_success = False

            while retries < max_retries:
                try:
                    response = self.session.get(
                        "https://api.gdeltproject.org/api/v2/doc/doc",
                        params=params,
                        timeout=20
                    )

                    if not response.text.strip():
                        raise ValueError("Empty response from GDELT")

                    data = response.json()

                    if "articles" not in data:
                        raise ValueError("Invalid JSON returned")

                    articles = data["articles"]

                    for article in articles:
                        domain = article.get("domain", "").lower()
                        if self._is_indian_source(domain):
                            url = article.get('url', 'N/A')
                            
                            seen_date = article.get('seendate', '')
                            parsed_date = self._parse_gdelt_date(seen_date)
                            
                            all_articles.append({
                                'title': article.get('title', 'N/A'),
                                'url': url,
                                'domain_name': domain,
                                'published': parsed_date,
                                'content': '',  # To be scraped
                                'lang': article.get('language', 'en'),
                                'country': 'IN',
                                'category': None,
                                'collection_method': 'GDELT',
                                'description': ''
                            })

                    print(f"[DAY] {current_date.date()}: +{len([a for a in articles if self._is_indian_source(a.get('domain', '').lower())])} Indian articles (Total: {len(all_articles)})")
                    day_success = True
                    break

                except Exception as e:
                    retries += 1
                    print(f"[ERR] GDELT error on {current_date.date()} (attempt {retries}): {e}")
                    time.sleep(2)

            if not day_success:
                print(f"[SKIP] Skipping {current_date.date()} after {max_retries} failures.")

            current_date = next_date
            time.sleep(1) 

        return all_articles

    def _parse_gdelt_date(self, date_string):
        """
        Robust parser for GDELT's seendate formats.
        Handles:
            20251118T123000Z
            20251118123000
            2025-11-18T12:30:00Z
            2025-11-18T12:30:00
            2025-11-18 12:30:00
        """
        if not date_string or date_string == 'N/A':
            return datetime.utcnow()

        s = date_string.strip()

        try:
            if 'T' in s:
                clean = s.replace('T', '').replace('Z', '')
                if len(clean) == 14 and clean.isdigit():
                    return datetime.strptime(clean, '%Y%m%d%H%M%S')
        except:
            pass

        try:
            if len(s) == 14 and s.isdigit():
                return datetime.strptime(s, '%Y%m%d%H%M%S')
        except:
            pass

        fmt_list = [
            '%Y-%m-%dT%H:%M:%S%z',   
            '%Y-%m-%dT%H:%M:%S',     
            '%Y-%m-%d %H:%M:%S',     
        ]

        for fmt in fmt_list:
            try:
                return datetime.strptime(s, fmt)
            except:
                continue

        print(f"[WARN] Could not parse GDELT date '{date_string}'. Using UTC now.")
        return datetime.utcnow()


    def _scrape_content_batch(self, articles, max_workers=5):
        """Scrape content from article URLs in parallel."""
        
        def scrape_single(article):
            if article['url'] in (None, 'N/A', ''):
                return article
            
            try:
                downloaded = trafilatura.fetch_url(article['url'])
                content = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=True
                )
                if content:
                    article['content'] = content[:50000] if content else ''  
                    article['scraped_at'] = datetime.utcnow()
                
            except Exception as e:
                article['scrape_error'] = str(e)
                article['content'] = ''
            
            return article
        
        print(f"Scraping content from {len(articles)} articles...")
        scraped_articles = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(scrape_single, article) for article in articles]
            for i, future in enumerate(as_completed(futures), 1):
                scraped_articles.append(future.result())
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(articles)} articles scraped")
        
        return scraped_articles

    def _save_to_mongodb(self, articles):
        """Save articles to MongoDB using bulk upsert."""
        
        operations = []
        for article in articles:
            category = infer_article_category(article)
            doc = {
                'title': article.get('title'),
                'url': article.get('url'),
                'domain_name': article.get('domain_name'),
                'published': article.get('published'),
                'content': article.get('content', ''),
                'lang': article.get('lang', 'en'),
                'country': article.get('country', 'IN'),
                'category': category,
                'source_category': article.get('category'),
                'category_source': 'source' if normalize_category(article.get('category')) else 'inferred',
                'collection_method': article.get('collection_method'),
                'description': article.get('description', ''),
                'scraped_at': article.get('scraped_at'),
                'updated_at': datetime.utcnow()
            }
            
            operations.append(
                UpdateOne(
                    {'url': article['url']},
                    {'$set': doc, '$setOnInsert': {'created_at': datetime.utcnow()}},
                    upsert=True
                )
            )
        
        if operations:
            result = self.collection.bulk_write(operations, ordered=False)
            return result.upserted_count, result.modified_count
        
        return 0, 0

    def _is_indian_source(self, domain):
        """Check if domain is from an Indian news source."""
        indian_domains = [
            'timesofindia.', 'hindustantimes.', 'indianexpress.',
            'thehindu.', 'ndtv.', 'news18.', 'indiatoday.',
            'business-standard.', 'livemint.', 'economictimes.',
            'thequint.', 'scroll.in', 'thewire.in', 'newslaundry.',
            'moneycontrol.', 'zeenews.', 'aajtak.', 'abplive.',
            'oneindia.', 'india.com', 'firstpost.', 'swarajyamag.',
            'deccanherald.', 'tribuneindia.', 'mid-day.', 'mumbaimirror.',
            'indianarrative.', 'opindia.', 'newsable.', 'newsx.',
            'outlookindia.', 'jagran.', 'amarujala.', 'dainikbhaskar.',
            'financialexpress.', 'indianewsnetwork.'
        ]
        return any(d in domain.lower() for d in indian_domains)

    def _parse_date(self, date_string):
        """Parse various RSS date formats to datetime."""
        if not date_string or date_string == 'N/A':
            return datetime.utcnow()
        
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',  
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',       
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                date_str = date_string[:35]
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        print(f"[WARN] Could not parse date: {date_string}")
        return datetime.utcnow()

    def _deduplicate(self, articles):
        """Remove duplicate articles based on URL."""
        seen = set()
        unique = []
        for article in articles:
            url = article.get('url')
            if url not in seen and url not in (None, 'N/A', ''):
                seen.add(url)
                unique.append(article)
        return unique

    def _save_corpus(self, articles, filename, output_dir):
        """Save corpus to multiple formats."""
        os.makedirs(output_dir, exist_ok=True)

        json_path = f"{output_dir}/{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False, default=str)
        print(f"[OK] Saved JSON: {json_path}")

        txt_path = f"{output_dir}/{filename}_corpus.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, article in enumerate(articles):
                f.write(f"--- DOCUMENT {i} ---\n")
                f.write(f"Title: {article.get('title')}\n")
                f.write(f"Domain: {article.get('domain_name')}\n")
                f.write(f"Published: {article.get('published')}\n")
                f.write(f"URL: {article.get('url')}\n")
                f.write(f"Language: {article.get('lang')}\n")
                f.write(f"Country: {article.get('country')}\n")
                f.write(f"Category: {article.get('category')}\n\n")
                f.write(f"{article.get('content', '')}\n\n")
        print(f"[OK] Saved Text: {txt_path}")

        csv_path = f"{output_dir}/{filename}_metadata.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,title,domain_name,published,url,lang,country,category\n")
            for i, article in enumerate(articles):
                t = str(article.get('title', '')).replace('"', '""')
                d = str(article.get('domain_name', '')).replace('"', '""')
                u = str(article.get('url', ''))
                p = str(article.get('published', ''))
                l = str(article.get('lang', 'en'))
                c = str(article.get('country', 'IN'))
                cat = str(article.get('category', '')).replace('"', '""')
                f.write(f'{i},"{t}","{d}","{p}","{u}","{l}","{c}","{cat}"\n')
        print(f"[OK] Saved CSV: {csv_path}")


if __name__ == "__main__":

    builder = IndiaCompleteCorpusBuilder(
        mongo_uri=os.getenv("MONGO_URI"),
        db_name=os.getenv("MONGO_DB")
    )
    
    builder.build_past_month_corpus("india_corpus")
