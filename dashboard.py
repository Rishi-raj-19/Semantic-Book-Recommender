import pandas as pd
import chromadb
import numpy as np
import os
import requests
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    INITIAL_TOP_K = 50
    FINAL_TOP_K = 16
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GORQ_API_KEY = "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"
    GORQ_ENDPOINT = "https://api.gorq.dev/embeddings"
    DEFAULT_COVER = "cover-not-found.jpg"


# === Helper Functions ===
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def format_authors(authors_str):
    if not authors_str or pd.isna(authors_str):
        return "Unknown Author"

    authors_split = authors_str.split(";")
    if len(authors_split) == 1:
        return authors_split[0]
    elif len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    else:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"


# === Step 1: Enhanced Embeddings with Fallback ===
class HybridEmbeddings(Embeddings):
    def __init__(self):
        self.gorq_api_key = Config.GORQ_API_KEY
        self.gorq_endpoint = Config.GORQ_ENDPOINT
        self.local_embeddings = HuggingFaceEmbeddings(model_name=Config.LOCAL_EMBEDDING_MODEL)
        self.use_local = False

        # Test Gorq connection
        try:
            socket.create_connection(("api.gorq.dev", 443), timeout=3).close()
        except (socket.gaierror, socket.timeout) as e:
            logger.warning(f"Gorq API unavailable: {str(e)}, falling back to local embeddings")
            self.use_local = True

    def embed_documents(self, texts):
        if self.use_local:
            return self.local_embeddings.embed_documents(texts)

        try:
            headers = {
                "Authorization": f"Bearer {self.gorq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "default-model",
                "input": texts
            }
            response = requests.post(self.gorq_endpoint, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            logger.error(f"Gorq API failed: {str(e)}, switching to local embeddings")
            self.use_local = True
            return self.local_embeddings.embed_documents(texts)

    def embed_query(self, text):
        if self.use_local:
            return self.local_embeddings.embed_query(text)

        try:
            headers = {
                "Authorization": f"Bearer {self.gorq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "default-model",
                "input": text
            }
            response = requests.post(self.gorq_endpoint, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Gorq API failed: {str(e)}, switching to local embeddings")
            self.use_local = True
            return self.local_embeddings.embed_query(text)


# === Step 2: Load data with error handling ===
try:
    if not os.path.exists("books_with_emotions.csv"):
        raise FileNotFoundError("Could not find books_with_emotions.csv file")

    books = pd.read_csv("books_with_emotions.csv")

    # Handle thumbnail URLs
    books["large_thumbnail"] = books["thumbnail"].apply(
        lambda x: (x + "&fife=w800" if is_valid_url(x) else Config.DEFAULT_COVER)
        if pd.notna(x) else Config.DEFAULT_COVER
    )

    # Validate required columns
    required_columns = ["isbn13", "title", "authors", "description", "simple_categories"]
    for col in required_columns:
        if col not in books.columns:
            raise ValueError(f"Missing required column: {col}")

except Exception as e:
    logger.error(f"Data loading error: {str(e)}")
    raise

# === Step 3: Load and embed documents ===
try:
    if not os.path.exists("tagged_description.txt"):
        raise FileNotFoundError("Could not find tagged_description.txt file")

    documents = []
    with open("tagged_description.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) > 1:  # Ensure line has content beyond just ISBN
                documents.append(Document(
                    page_content=line,
                    metadata={"source": "tagged_description.txt"}
                ))

    if not documents:
        raise ValueError("No valid documents found in tagged_description.txt")

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # Initialize embeddings with automatic fallback
    embeddings = HybridEmbeddings()
    db_books = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=chromadb.config.Settings(anonymized_telemetry=False)
    )

except Exception as e:
    logger.error(f"Document processing error: {str(e)}")
    raise


# === Step 4: Recommendation logic ===
def retrieve_semantic_recommendations(query, category=None, tone=None):
    try:
        if not isinstance(query, str) or not query.strip():
            return pd.DataFrame()

        recs = db_books.similarity_search(query, k=Config.INITIAL_TOP_K)
        books_list = []
        for rec in recs:
            try:
                isbn_str = rec.page_content.strip('"').split()[0]
                books_list.append(int(isbn_str))
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to process ISBN from document: {str(e)}")
                continue

        if not books_list:
            return pd.DataFrame()

        book_recs = books[books["isbn13"].isin(books_list)].copy()

        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category]

        if tone and tone != "All":
            tone_map = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            if tone in tone_map:
                book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

        return book_recs.head(Config.FINAL_TOP_K)
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()


def recommend_books(query, category, tone):
    if not query or len(query.strip()) < 3:  # Minimum length check
        return [(Config.DEFAULT_COVER, "Please enter at least 3 characters")]

    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            try:
                description = row.get("description", "No description available")
                truncated_desc = " ".join(
                    description.split()[:30]) + "..." if description else "No description available"

                authors_str = format_authors(row.get("authors"))
                title = row.get("title", "Untitled")
                thumbnail = row.get("large_thumbnail", Config.DEFAULT_COVER)

                caption = f"{title} by {authors_str}: {truncated_desc}"
                results.append((thumbnail, caption))
            except Exception as e:
                logger.warning(f"Error processing book {row.get('isbn13', 'unknown')}: {str(e)}")

        return results if results else [(Config.DEFAULT_COVER, "No recommendations found. Try a different query.")]
    except Exception as e:
        logger.error(f"Error in recommendation system: {str(e)}")
        return [(Config.DEFAULT_COVER, "An error occurred. Please try again.")]


# === Step 5: Gradio interface ===
# === Step 5: Gradio interface ===
try:
    categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
except Exception as e:
    logger.error(f"Failed to prepare UI options: {str(e)}")
    raise

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
            max_length=500
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books",
        columns=4,
        rows=4,
        object_fit="contain",
        height="auto"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
