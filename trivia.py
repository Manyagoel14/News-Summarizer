import os
import spacy
import trafilatura
import sys
import re
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import random

load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")

model = GenerativeModel("gemini-1.5-flash") 

client = MongoClient(MONGO_URI)
db = client[DB]

print("TRIVIA TIME")
print("1. Sports\n2. Entertainment\n3. Politics\n4. Health\n5. Technology\n6. India\n7. Business")

choice = int(input("Choose any one: "))

d = {
    1: "sports",
    2: "entertainment",
    3: "politics",
    4: "health",
    5: "technology",
    6: "india",
    7: "business"
}

collection = db[f"trivia_{d[choice]}"]

data = list(collection.find())

random.shuffle(data)

print("\nYOUR TRIVIA QUIZ\n")

for i in range(5):
    print(f"Q{i+1}. {data[i]['question'][3:]}")
    print(f"Answer: {data[i]['answer']}\n")
