import os
import trafilatura
import sys
import re
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from gemini_client import generate_text

load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[DB]
raw_collection = db["raw_articles"]
final_collection = db["final_dataset"]
collection = raw_collection if raw_collection.estimated_document_count() else final_collection
print(f"Using source collection: {collection.name}")

nlp = None


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


def load_nlp():
    global nlp
    if nlp is not None:
        return nlp
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Warning: spaCy unavailable; using regex sentence splitter: {e}")
        nlp = False
    return nlp


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


def infer_article_category(article, content=""):
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

    haystack = " ".join(
        str(article.get(field) or "")
        for field in ("title", "description", "domain_name", "url")
    ).lower()
    if content:
        haystack = f"{haystack} {content[:3000].lower()}"

    keyword_scores = {
        canonical: sum(
            2 if alias in str(article.get("title") or "").lower() else 1
            for alias in aliases
            if re.search(rf"\b{re.escape(alias)}\b", haystack)
        )
        for canonical, aliases in CATEGORY_ALIASES.items()
    }
    best_category, best_score = max(keyword_scores.items(), key=lambda item: item[1])
    return best_category if best_score > 0 else None


def save_inferred_category(article_id, category):
    if not article_id or not category:
        return
    collection.update_one(
        {"_id": article_id},
        {
            "$set": {
                "category": category,
                "inferred_category": category,
                "category_source": "trivia_inference",
            }
        },
    )

def get_sentences(text):
    nlp_model = load_nlp()
    if nlp_model:
        doc = nlp_model(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", text) if sent.strip()]

def compute_sentence_scores(sentences):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1)
    return [score.item() for score in scores]

def get_top_sentences(text, top_n=3):
    sentences = get_sentences(text)
    if not sentences:
        return []
    scores = compute_sentence_scores(sentences)
    ranked = list(zip(sentences, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]


def parse_questions_block(text):
    lines = text.strip().split("\n")
    questions = []
    block = []

    for line in lines:
        if re.match(r"^\d+\.", line.strip()):
            if block:
                questions.append("\n".join(block))
                block = []
        block.append(line.strip())

    if block:
        questions.append("\n".join(block))

    return questions[:4]   


def generate_questions_from_text(title, text):
    prompt = f"""
You are a trivia question generator.

Using ONLY the title and article content below, create exactly **4** trivia questions.

Allowed formats:
2 ques of Multiple-choice (MCQ)
2 ques of Fill-in-the-blank

Rules:
- Every question must be factual and short.
- Every MCQ must contain 4 options: a), b), c), d)
- Each question must include an "ans:" line.
- 1. and 2. should be MCQ and 3. and 4. must be fill in the blanks.
- No explanations. No extra text.

STRICT OUTPUT FORMAT:

1. <question?>
   a) <option>
   b) <option>
   c) <option>
   d) <option>
   ans: <a/b/c/d>
2. <question?>
   a) <option>
   b) <option>
   c) <option>
   d) <option>
   ans: <a/b/c/d>
3. <fill-in-the-blank sentence ___ >
   ans: <correct word/phrase>
4. <fill-in-the-blank sentence ___ >
   ans: <correct word/phrase>

TITLE:
{title}

CONTENT:
{text}
"""

    text = generate_text(prompt)
    if not text:
        return []

    return parse_questions_block(text)

categories = {"technology", "sports", "entertainment", "business", "health", "india", "politics"}

cursor = collection.find(
    {},
    {"title": 1, "url": 1, "domain_name": 1, "description": 1, "content": 1, "category": 1}
)

for article in cursor:
    title = article.get("title")
    url = article.get("url")

    if not title or not url:
        continue

    content = (article.get("content") or "").strip()

    if len(content) < 50:
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True
        ) or ""

    if not content or len(content) < 50:
        continue

    cat = infer_article_category(article, content)
    if cat not in categories:
        print(f"Skipping uncategorized article: {title}")
        continue

    save_inferred_category(article.get("_id"), cat)

    trivia_collection = db[f"trivia_{cat}"]
    if trivia_collection.count_documents({"source_url": url}, limit=1):
        print(f"Skipping existing trivia for: {title}")
        continue

    full_text = f"{title}. {content}"

    top_sentences = get_top_sentences(full_text, top_n=3)
    combined_text = " ".join(top_sentences)

    questions = generate_questions_from_text(title, combined_text)

    inserted = 0
    for q in questions:
        ans_index = q.lower().rfind("ans:")
        if ans_index == -1:
            continue

        question_text = q[:ans_index].strip()
        answer = q[ans_index + 4:].strip()

        if "a)" in q and "b)" in q:
            qtype = "mcq"
        else:
            qtype = "fill"

        trivia_collection.insert_one({
            "title": title,
            "category": cat,
            "question": question_text,
            "answer": answer,
            "type": qtype,
            "source_url": url
        })
        inserted += 1

    print(f"{cat}: inserted {inserted} question(s) for {title}")
    print("=" * 90)
    time.sleep(7)
