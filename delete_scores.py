from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["final_dataset"]

result = coll.update_many({}, {"$unset": {"content_clean": ""}})
print(f"Removed 'content_clean' from {result.modified_count} documents.")
res = coll.update_many({"vector": {"$exists": True}}, {"$unset": {"vector": ""}})
print(f"Removed 'vectors' from {res.modified_count} documents.")
bigram = coll.update_many({"title_bigrams": {"$exists": True}}, {"$unset": {"title_bigrams": ""}})
print(f"Removed 'title_bigrams' from {bigram.modified_count} documents.")
a = coll.update_many({"term_idf": {"$exists": True}}, {"$unset": {"term_idf": ""}})
print(f"Removed 'title_bigrams' from {a.modified_count} documents.")
b = coll.update_many({"title_bigram_idf": {"$exists": True}}, {"$unset": {"title_bigram_idf": ""}})
print(f"Removed 'title_bigrams_idf' from {b.modified_count} documents.")
c= coll.update_many({"title_bigram_weights": {"$exists": True}}, {"$unset": {"title_bigram_weights": ""}})
print(f"Removed 'title_bigram_weights' from {c.modified_count} documents.")
d= coll.update_many({"term_lnc": {"$exists": True}}, {"$unset": {"term_lnc": ""}})
print(f"Removed 'title_bigrams' from {c.modified_count} documents.")
e= coll.update_many({"title_bigram_lnc": {"$exists": True}}, {"$unset": {"title_bigram_lnc": ""}})
print(f"Removed 'title_bigram_lnc' from {c.modified_count} documents.")
