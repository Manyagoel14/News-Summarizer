# News Summarizer

A Streamlit-based news intelligence app that builds a MongoDB news corpus, retrieves relevant articles with a custom IR pipeline, and uses Gemini for grounded summaries, follow-up QA, trivia generation, and fake-vs-real headline games.

## Tech Stack

- Python 3.11+
- Streamlit frontend
- MongoDB document store
- Custom LNC / TF-IDF retrieval in `search.py`
- Gemini API through `google-genai`
- Trafilatura, Feedparser, GDELT/RSS ingestion

## Repository Layout

```text
.
+-- frontend/
|   +-- app.py                    # Streamlit landing page
|   +-- pages/
|       +-- summary.py            # Search, summary, follow-up QA
|       +-- play_trivia.py        # Trivia UI
|       +-- realorfake.py         # Fake-vs-real news game UI
+-- generate_data.py              # Builds final_dataset corpus
+-- preprocess_new.py             # Tokenization, lemmatization, vector storage
+-- search.py                     # Hybrid retrieval/ranking engine
+-- summarizer.py                 # Gemini-backed summary and follow-up QA
+-- generate_trivia_ques.py       # Generates trivia_<category> collections
+-- backfill_categories.py        # Infers category for existing DB rows
+-- gemini_client.py              # Gemini client, retries, model config
+-- requirements.txt
```

## Environment Variables

Copy the sample env file and fill in your values:

```powershell
copy .env.example .env
```

Required:

```env
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=news_summarizer
GEMINI_API_KEY=your_gemini_api_key
```

Optional:

```env
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.0-flash
GROQ_API_KEY=your_groq_api_key
```

`GEMINI_FALLBACK_MODEL` is useful when Gemini returns temporary `503 UNAVAILABLE` high-demand errors.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Make sure MongoDB is running locally or set `MONGO_URI` to your Atlas connection string.

## Run The App

Start the Streamlit UI:

```powershell
streamlit run frontend/app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Data Pipeline

### 1. Build Or Refresh The News Corpus

```powershell
python generate_data.py
```

This collects Indian news from RSS/GDELT sources, scrapes article content, infers categories, and upserts documents into:

```text
final_dataset
```

### 2. Preprocess Documents For Search

```powershell
python preprocess_new.py
```

This computes cleaned tokens, LNC vectors, and title bigram metadata used by `search.py`.

### 3. Backfill Categories For Existing Data

Run this if older documents have `category: null`:

```powershell
python backfill_categories.py
```

It updates `final_dataset` and `raw_articles` with inferred categories such as:

```text
business, sports, entertainment, technology, health, india, politics
```

### 4. Generate Trivia Collections

```powershell
python generate_trivia_ques.py
```

This reads from `raw_articles` if available; otherwise it falls back to `final_dataset`. It creates collections named:

```text
trivia_business
trivia_sports
trivia_entertainment
trivia_technology
trivia_health
trivia_india
trivia_politics
```

The script calls Gemini, so it may hit API quota or rate limits.

## MongoDB Collections

| Collection | Purpose |
| --- | --- |
| `final_dataset` | Main searchable article corpus |
| `trivia_<category>` | Generated trivia questions per category |
| game/score collections | Used by fake-vs-real and CLI game scripts |

## Core Search Flow

1. User enters a query in Streamlit.
2. `search.py` preprocesses the query.
3. Hybrid ranking combines content vector score and title bigram score.
4. Top documents are sent to `summarizer.py`.
5. Gemini generates a grounded answer using only retrieved documents.
6. Follow-up questions reuse cached documents and recent follow-up history.

## Gemini Configuration

The Gemini wrapper is in `gemini_client.py`.

Supported env variables:

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.0-flash
```

## Common Commands

```powershell
# Run app
streamlit run frontend/app.py

# Build corpus
python generate_data.py

# Build search vectors
python preprocess_new.py

# Fix null categories
python backfill_categories.py

# Generate trivia
python generate_trivia_ques.py

# CLI search/summarizer test
python summarizer.py
```
