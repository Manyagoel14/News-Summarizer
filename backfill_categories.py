import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from generate_data import infer_article_category, normalize_category


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017/"
DB_NAME = os.getenv("MONGO_DB") or "news_db"
COLLECTIONS = ("final_dataset", "raw_articles")


def needs_category(doc):
    return normalize_category(doc.get("category")) is None


def backfill_collection(db, collection_name, batch_size=500):
    collection = db[collection_name]
    if collection.estimated_document_count() == 0:
        print(f"{collection_name}: empty, skipping")
        return 0

    cursor = collection.find(
        {},
        {"title": 1, "url": 1, "domain_name": 1, "description": 1, "content": 1, "category": 1},
    )

    updates = []
    updated = 0

    for doc in cursor:
        if not needs_category(doc):
            continue

        category = infer_article_category(doc)
        if not category:
            continue

        updates.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "category": category,
                        "inferred_category": category,
                        "category_source": "backfill",
                        "category_backfilled_at": datetime.utcnow(),
                    }
                },
            )
        )

        if len(updates) >= batch_size:
            result = collection.bulk_write(updates, ordered=False)
            updated += result.modified_count
            updates = []

    if updates:
        result = collection.bulk_write(updates, ordered=False)
        updated += result.modified_count

    print(f"{collection_name}: updated {updated} article(s)")
    return updated


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    total = 0

    for collection_name in COLLECTIONS:
        total += backfill_collection(db, collection_name)

    print(f"Done. Updated {total} article(s).")


if __name__ == "__main__":
    main()
