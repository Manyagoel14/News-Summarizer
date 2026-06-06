import os
import math
from datetime import datetime

from pymongo import MongoClient
from dotenv import load_dotenv

from preprocess_new import preprocess_text_to_tokens

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["final_dataset"]

SYNONYM_MAP = {
    "us": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "pm": "prime minister",
    "pres": "president",
    "u.n": "united nations",
    "un": "united nations",
    "ai": "artificial intelligence",
}

def expand_synonyms_query(tokens):
    expanded = []
    for t in tokens:
        if t.lower() in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[t.lower()].split())
        else:
            expanded.append(t)
    return expanded


def tf_weight(tf):
    return 1.0 + math.log10(tf) if tf > 0 else 0.0

def make_query_bigrams(tokens):
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]

def query_to_ltn_vector(query_text, idf_map, vocab=None):
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=True, min_lemma_len=1)
    tokens = expand_synonyms_query(tokens)

    if not tokens:
        return {}

    tf_q = {}
    for t in tokens:
        tf_q[t] = tf_q.get(t, 0) + 1

    vec_q = {}
    for t, f in tf_q.items():
        if idf_map and t in idf_map:   
            vec_q[t] = tf_weight(f) * idf_map[t]

    if not vec_q:
        return {}

    norm = sum(v * v for v in vec_q.values()) ** 0.5
    if norm == 0:
        return vec_q

    for k in vec_q:
        vec_q[k] /= norm

    return vec_q


def query_to_bigram_vector(query_text, idf_map=None, vocab=None):
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=True, min_lemma_len=1)
    print("DEBUG TOKENS FOR BIGRAM:", tokens)
    tokens = expand_synonyms_query(tokens)
    bigrams = make_query_bigrams(tokens)

    if not bigrams:
        return {}

    tf_b = {}
    for b in bigrams:
        tf_b[b] = tf_b.get(b, 0) + 1

    vec_b = {}
    for b, f in tf_b.items():
        idf_b = (
            idf_map.get(b)
            if idf_map and b in idf_map else
            idf_map.get(b.replace("_", " ")) if idf_map and b.replace("_", " ") in idf_map else
            idf_map.get(b.replace("_", "-")) if idf_map and b.replace("_", "-") in idf_map else
            None
        )

        if idf_b is None:
            continue

        vec_b[b] = tf_weight(f) * idf_b

    if not vec_b:
        return {}

    norm = sum(v * v for v in vec_b.values()) ** 0.5
    if norm == 0:
        return vec_b

    for k in vec_b:
        vec_b[k] /= norm

    return vec_b


def canonicalize_bigram_token(b):
    if not isinstance(b, str):
        return None
    b = b.strip()
    if " " in b:
        return b.replace(" ", "_")
    if "-" in b:
        return b.replace("-", "_")
    return b  

def get_or_build_index_from_collection(collection,
                                       use_sampling=False,
                                       sample_size=1000,
                                       bigram_field_names=None,
                                       debug=False):
    bigram_field_names = bigram_field_names or ["title_bigram_lnc", "title_bigram_weights"]

    try:
        meta = collection.find_one({"_id": "_index_meta_"})
        if meta and isinstance(meta.get("N"), int) and isinstance(meta.get("df_map"), dict) and isinstance(meta.get("idf_map"), dict):
            if debug:
                print("Loaded index metadata from collection meta doc")
            return meta["N"], meta["df_map"], meta["idf_map"]
    except Exception as e:
        if debug:
            print("Meta load error:", e)

    df_map = {}
    N = 0

    if use_sampling:
        cursor = collection.aggregate([
            {"$sample": {"size": sample_size}},
            {"$project": {"content_clean": 1, "title_bigrams": 1, **{f: 1 for f in bigram_field_names}}}
        ])
    else:
        cursor = collection.find({}, {"content_clean": 1, "title_bigrams": 1, **{f: 1 for f in bigram_field_names}})

    for doc in cursor:
        N += 1
        content = doc.get("content_clean") or ""
        if isinstance(content, str) and content.strip():
            try:
                unique_terms = set(content.split())
            except Exception:
                unique_terms = set()
            for t in unique_terms:
                if not t:
                    continue
                df_map[t] = df_map.get(t, 0) + 1

        tb_arr = doc.get("title_bigrams")
        if isinstance(tb_arr, list):
            for raw_b in set(tb_arr):
                b = canonicalize_bigram_token(raw_b)
                if b:
                    df_map[b] = df_map.get(b, 0) + 1

        for bf in bigram_field_names:
            bmap = doc.get(bf)
            if isinstance(bmap, dict):
                for raw_b in bmap.keys():
                    b = canonicalize_bigram_token(raw_b)
                    if b:
                        df_map[b] = df_map.get(b, 0) + 1

    if N == 0:
        return 0, {}, {}

    idf_map = {}
    for term, df in df_map.items():
        try:
            idf_map[term] = math.log10((N) / (1.0 + df)) if df > 0 else 0.0
        except Exception:
            idf_map[term] = 0.0

  
    try:
        collection.update_one(
            {"_id": "_index_meta_"},
            {"$set": {"N": N, "df_map": df_map, "idf_map": idf_map, "built_at": datetime.utcnow()}},
            upsert=True
        )
    except Exception:
        if debug:
            print("Warning: failed to persist index meta")

    return N, df_map, idf_map

def score_docs_for_query(vec_q, vec_q_bigram=None, top_k=10, content_weight=0.65, bigram_weight=0.35):
    if (not vec_q or len(vec_q) == 0) and (not vec_q_bigram or len(vec_q_bigram) == 0):
        return []

    vec_q = vec_q or {}
    vec_q_bigram = vec_q_bigram or {}

    def l2_norm_vec(v):
        if not v:
            return 0.0
        return math.sqrt(sum(val * val for val in v.values()))

    max_q_weight = l2_norm_vec(vec_q)
    max_q_bigram_weight = l2_norm_vec(vec_q_bigram)

    cursor = coll.find(
        {"vector": {"$exists": True}},
        {"title": 1, "url": 1, "vector": 1, "published": 1, "content_clean": 1,
         "title_bigram_lnc": 1, "title_bigram_weights": 1, "title_bigrams": 1}
    )

    results = []

    for doc in cursor:
        doc_vec = doc.get("vector", {}) or {}

        content_score_raw = 0.0
        if vec_q and doc_vec:
            content_score_raw = sum(vec_q.get(t, 0.0) * doc_vec.get(t, 0.0) for t in vec_q.keys())

        bigram_score_raw = 0.0
        if vec_q_bigram:
            doc_bigram_map = None
            if isinstance(doc.get("title_bigram_lnc"), dict) and doc.get("title_bigram_lnc"):
                doc_bigram_map = { canonicalize_bigram_token(k): v for k, v in doc.get("title_bigram_lnc").items() if canonicalize_bigram_token(k) }
            elif isinstance(doc.get("title_bigram_weights"), dict) and doc.get("title_bigram_weights"):
                doc_bigram_map = { canonicalize_bigram_token(k): v for k, v in doc.get("title_bigram_weights").items() if canonicalize_bigram_token(k) }

            if doc_bigram_map:
                doc_bigram_norm = math.sqrt(sum(v * v for v in doc_bigram_map.values())) if doc_bigram_map else 0.0
                if doc_bigram_norm > 0:
                    for b, qw in vec_q_bigram.items():
                        cb = canonicalize_bigram_token(b)
                        if not cb:
                            continue
                        doc_w = doc_bigram_map.get(cb, 0.0)
                        if doc_w:
                            bigram_score_raw += qw * (doc_w / doc_bigram_norm)
                else:
                    bigram_score_raw = sum(vec_q_bigram.get(b, 0.0) * doc_bigram_map.get(canonicalize_bigram_token(b), 0.0) for b in vec_q_bigram.keys())
            else:
                tb_arr = doc.get("title_bigrams") or []
                if tb_arr:
                    tb_set = set(canonicalize_bigram_token(x) for x in tb_arr if x)
                    for b, qw in vec_q_bigram.items():
                        cb = canonicalize_bigram_token(b)
                        if cb in tb_set:
                            bigram_score_raw += qw

        content_score_norm = content_score_raw / max_q_weight if max_q_weight > 0 else content_score_raw
        bigram_score_norm = bigram_score_raw / max_q_bigram_weight if max_q_bigram_weight > 0 else bigram_score_raw

        final_relevance = (content_weight * content_score_norm) + (bigram_weight * bigram_score_norm)

        if final_relevance > 0:
            results.append({
                "doc": doc,
                "relevance": final_relevance,
                "content_score": content_score_norm,
                "bigram_score": bigram_score_norm
            })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]


def hybrid_prune_dynamic(hits, query_text, idf_map, alpha=0.25, gap_ratio_threshold=2.0, min_docs=1, debug=True):
    if not hits:
        return []

    hits = sorted(hits, key=lambda x: x['relevance'], reverse=True)
    vec_q = query_to_ltn_vector(query_text, idf_map)
    if not vec_q:
        return hits[:min_docs]

    max_q_weight = sum(vec_q.values())
    kept = []
    debug_rows = []
    for h in hits:
        doc_terms = h['doc'].get("content_clean", "").split()
        overlap_weight = sum(vec_q.get(t, 0) for t in doc_terms)
        passes = overlap_weight >= alpha * max_q_weight
        if passes:
            kept.append(h)
        if debug:
            debug_rows.append({"title": h['doc'].get("title"), "overlap_weight": overlap_weight, "passes": passes})

    if debug:
        print("\nPRUNE DEBUGGING")
        for r in debug_rows[:20]:
            print(f"  {r['title'][:50]:50s} | overlap={r['overlap_weight']:.4f} | passes={r['passes']}")

    if len(kept) > 1:
        scores = [h['relevance'] for h in kept]
        ratios = [(scores[i] / scores[i+1]) if scores[i+1] != 0 else float('inf') for i in range(len(scores)-1)]
        max_ratio = max(ratios)
        if max_ratio >= gap_ratio_threshold:
            idx = ratios.index(max_ratio)
            kept = kept[:idx+1]

    if len(kept) < min_docs:
        return hits[:min_docs]
    return kept

def get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50):
    if not content_clean or not query_terms:
        return content_clean[:snippet_length] if content_clean else ""
    words = content_clean.split()
    query_set = set(query_terms)
    if len(words) <= window_size:
        snippet = " ".join(words)
        return snippet[:snippet_length]
    best_score = 0
    best_start = 0
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        score = sum(1 for w in window if w in query_set)
        if score > best_score:
            best_score = score
            best_start = i
    snippet_words = words[best_start:best_start + window_size]
    snippet = " ".join(snippet_words)
    if len(snippet) > snippet_length:
        snippet = snippet[:snippet_length].rsplit(' ', 1)[0] + "..."
    return snippet

def search(query_text, k=10, prune=True, gap_ratio_threshold=3.0, alpha=0.3, debug=False,
           content_weight=0.65, bigram_weight=0.35, use_sampling_index=False, min_relevance=0.067):

    N, df_map, idf_map = get_or_build_index_from_collection(
        coll,
        use_sampling=use_sampling_index,
        sample_size=1000,
        bigram_field_names=["title_bigram_lnc", "title_bigram_weights"],
        debug=debug
    )
    vocab = set(df_map.keys()) if df_map else set()

    #query vectors
    vec_q = query_to_ltn_vector(query_text, idf_map, vocab=None)
    vec_q_bigram = query_to_bigram_vector(query_text, idf_map=idf_map, vocab=None)

    if debug or True: 
      print(f"\nQUERY VECTOR DEBUG: ")
      print(f"Unigram vector: {vec_q}")
      print(f"Bigram vector: {vec_q_bigram}")
      print(f"Unigram weight sum: {sum(vec_q.values()) if vec_q else 0}")
      print(f"Bigram weight sum: {sum(vec_q_bigram.values()) if vec_q_bigram else 0}")

    query_tokens_raw = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    query_terms = expand_synonyms_query(query_tokens_raw)

    if not vec_q and not vec_q_bigram:
        if debug:
            print("⚠ Query produced no valid terms.")
        return []

    hits = score_docs_for_query(vec_q, vec_q_bigram=vec_q_bigram, top_k=100,
                                content_weight=content_weight, bigram_weight=bigram_weight)
    if not hits:
        if debug:
            print("⚠ No matching documents found.")
        return []

    if prune:
        pruned_hits = hybrid_prune_dynamic(
            hits, query_text, idf_map,
            alpha=alpha,
            gap_ratio_threshold=gap_ratio_threshold,
            min_docs=1,
            debug=debug
        )
    else:
        pruned_hits = hits

    if min_relevance is not None:
        before_count = len(pruned_hits)
        pruned_hits = [h for h in pruned_hits if h.get("relevance", 0.0) >= float(min_relevance)]
        after_count = len(pruned_hits)
        if debug:
            dropped = before_count - after_count
            print(f"[debug] min_relevance={min_relevance} removed {dropped} docs (kept {after_count})") 
            
    if k is not None:
        pruned_hits = pruned_hits[:k]

    results = []
    for h in pruned_hits:
        doc = h["doc"]
        content_clean = doc.get("content_clean", "")
        snippet = get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50)
        results.append({
            "title": doc.get("title", "Untitled"),
            "url": doc.get("url", ""),
            "relevance": h["relevance"],
            "snippet": snippet,
            "published": doc.get("published"),
            "content_score": h.get("content_score", 0.0),
            "bigram_score": h.get("bigram_score", 0.0)
        })
    return results

def interactive_search():
    print("\n" + "=" * 60)
    print("SEARCH ENGINE - Interactive Mode")
    print("=" * 60)
    print("Type your query and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nSee ya later! Alligator!")
                break
            print(f"\nSearching for: '{query}'")
            hits = search(query, k=10, prune=True, debug=False)
            if not hits:
                print("No results found.\n")
                continue
            print(f"\nFound {len(hits)} results:\n")
            for i, h in enumerate(hits, 1):
                print(f"{i}. [total={h['relevance']:.4f}] (content={h['content_score']:.4f}, bigram={h['bigram_score']:.4f})")
                print(f"   {h['title']}")
                print(f"    {h['url']}")
                print(f"   {h['snippet'][:150]}...")
                if h.get('published'):
                    print(f"    {h['published']}")
                print()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    interactive_search()