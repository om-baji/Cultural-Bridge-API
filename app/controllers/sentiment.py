from fastapi import HTTPException
from typing import Any, List, Dict
import ollama
import asyncio
from dotenv import load_dotenv
import datetime


load_dotenv()
PATH="./models/sentiment_model.h5"
TOKENIZER_PATH="./models/tfidf_vectorizer.h5"