# Semantic-Book-Recommender
A smart, emotion-aware book recommendation system that uses semantic similarity, vector search, and natural language understanding to match users with books based on the meaning behind their input — not just keywords.
This project is ideal for readers who want personalized and emotionally aligned book suggestions, based on how they feel or what kind of story they're looking for.

**Features**
Accepts free-text user input (e.g., “a bittersweet love story with a happy ending”)
Filters by genre and emotional tone (e.g., joy, sadness, fear)
Uses HuggingFace sentence-transformers or Gorq API for embeddings
Performs vector similarity search using ChromaDB
Displays book results with cover images, titles, authors, and summaries
Clean, interactive Gradio web interface

**Tech Stack**
Python 3.8+
Gradio
LangChain
HuggingFace Transformers
ChromaDB
dotenv, requests, pandas

**Project Structure**
├── dashboard.py               # Main app
├── tagged_description.txt     # ISBN-tagged book descriptions
├── books_with_emotions.csv    # Book metadata with emotional scores
├── cover-not-found.jpg        # Fallback image
├── .env                       # API key storage
