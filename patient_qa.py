
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# load knowledge
with open("knowledge.txt", "r") as f:
    content = f.read()

sentences = [s.strip() for s in content.split("\n") if s.strip() != ""]

# setup vector search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(sentences)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

client = Groq(api_key="GROQ_API_KEY")

def search(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [sentences[i] for i in indices[0]]

def answer_patient_question(question):
    relevant_context = search(question)
    context_text = "\n".join(relevant_context)

    prompt = f"""You are a helpful medical assistant in a hospital.
Answer the patient's question in a clear, friendly and professional way.
Use the context provided and your general medical knowledge to give helpful answers.
Always recommend consulting a doctor for serious concerns.

Context:
{context_text}

Patient question: {question}

Answer helpfully and clearly."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"