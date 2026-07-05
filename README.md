# Hospital AI Service

AI-powered microservice for the Hospital Management System built with Python, Flask, LangChain, and Groq.

## Features

### 1. Symptom Checker
- Patient describes symptoms
- AI analyzes and recommends doctor specialization
- Provides urgency level and basic advice

### 2. Patient Q&A (RAG)
- Retrieval Augmented Generation using FAISS vector search
- Answers medical questions from knowledge base
- Uses Sentence Transformers for embeddings

### 3. Discharge Summary Generator
- Doctor inputs patient diagnosis and notes
- AI generates professional medical discharge summary
- Automatically saved to medical record

## Tech Stack
- **Python** — Flask REST API
- **LangChain** — AI pipeline and prompt templates
- **FAISS** — Vector database for RAG
- **Sentence Transformers** — Text embeddings
- **Groq API** — LLaMA 3.3 70B model

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ai/symptom-checker` | Check symptoms |
| POST | `/ai/patient-qa` | Patient Q&A |
| POST | `/ai/discharge-summary` | Generate discharge summary |

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable:
```bash
GROQ_API_KEY=your_groq_api_key
```

3. Run:
```bash
python app.py
```

## Part of
This service is part of the [AI-Powered Hospital Management System](https://github.com/rajdev18/Hospital-management-system)
