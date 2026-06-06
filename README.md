# News Summarizer using Retrieval-Augmented Generation (RAG)

An interactive IR + NLP–based system that retrieves the most relevant news articles, ranks them using a hybrid unigram–bigram IR model, and generates concise, factual summaries using Retrieval-Augmented Generation.
The system also supports follow-up question answering, top-headline extraction, trivia games, and fake-vs-real news detection.

## Features

### Query-Based Search
- Accepts user queries.
- Retrieves top-matching news articles from a custom corpus.
- Uses hybrid unigram + bigram IR scoring.
- Dynamic pruning removes irrelevant documents.

### RAG-Based Summarization
- Summaries are generated *only from retrieved evidence*.
- Grounded prompting ensures no hallucinations.
- Produces short, factual, query-focused summaries.

### Follow-Up Question Answering
- Extracts relevant sentences using TF-IDF + cosine similarity.
- Responds with short factual answers derived from matched evidence.

### Headlines Retrieval
- Fetches top headlines from an API.
- Displays title, URL, and instant summary.

### Trivia Game
- Extracts important sentences per article.
- Generates MCQs and fill-in-the-blanks using Gemini API.
- Helps users learn and retain news information.

### Fake vs Real News Module
- Generates fake variations of real headlines.
- Users must identify the authentic one.
- Designed to improve media literacy.


---

## Module Overview

### 1. Dataset Building
- Scrapes news articles from verified Indian sources.
- Uses GDELT API + Trafilatura.
- Builds a corpus for a chosen time period.

### 2. Document Preprocessing
- Removes metadata, URLs, timezones, punctuation, etc.
- Lemmatization and stopword removal using spaCy.
- Title-level bigram extraction.
- LNC vector formation for content:
  
  ```
  LNC(t) = 1 + log(tf_t)
  ```

 
### 3. Hybrid Search & Ranking Engine

#### Query Processing
- Cleaned and tokenized queries.
- Synonym expansion:
  - ai → artificial intelligence  
  - us → united states  
  - pm → prime minister  

#### Relevance Scoring

  ```
  Relevance = α * content_score + (1 − α) * bigram_score
  ```


Default weights:
- Content score: **0.65**
- Bigram score: **0.35**

#### Dynamic Pruning
- Removes documents with insufficient term overlap.
- Gap-based cutoff removes documents with sudden score drop.

---

##  Summarization & Follow-Up QA

### Summarization (RAG)
- Retrieves top-K documents.
- Builds an “evidence block” from matched text.
- Grounded prompting ensures summarization is factual.

### Follow-Up Question Answering
- Splits retrieved documents into sentences.
- Builds TF-IDF vectors.
- Selects top relevant sentences using cosine similarity.
- Generates short factual answers using grounded prompting.

---

##  Experiments & Results

The evaluation includes:

1. **Relevant document retrieval**  
2. **Generated query-focused summary**  
3. **Follow-up QA responses**

Screenshots (search page, summary page, QA page) demonstrate:
- Accurate IR retrieval  
- Extractive, grounded summaries  
- Reliable follow-up answers  

---

##  Conclusions

- Bigram scoring at the headline level improves query precision.
- The hybrid unigram–bigram approach outperforms unigram-only models.
- RAG provides hallucination-free summarization.
- The system is effective for fast, reliable, multi-source news summarization.

---

##  Limitations

- Corpus limited to one month due to API constraints.
- Free LLM APIs restrict number of requests/minute.
- No real-time automated scraping pipeline.
- Corpus generation is manual and local.

---

## Contributers

- Manya Goel
- Dhanya Girdhar
- Raashi Sharma









