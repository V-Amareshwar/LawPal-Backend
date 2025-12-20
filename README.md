# LawPal Backend - AI Legal Assistant

AI-powered legal assistant using RAG (Retrieval-Augmented Generation) for Indian legal queries.

## Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd LawPal-backend
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
- Copy `.env.example` to `.env`
- Add your Groq API keys

5. **Build database (one-time)**
```bash
python build_db.py
```

6. **Run the server**
```bash
python app.py
```

## API Usage

**POST** `/ask`
```json
{
  "query": "What is Section 498A IPC?"
}
```

## Tech Stack
- Flask
- ChromaDB (Vector Database)
- BM25 (Keyword Search)
- Groq (LLM)
- Sentence Transformers (Embeddings)