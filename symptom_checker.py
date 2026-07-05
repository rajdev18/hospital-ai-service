
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

def check_symptoms(symptoms):
    relevant_context = search(symptoms)
    context_text = "\n".join(relevant_context)

    prompt = f"""You are an experienced medical assistant in a hospital.
A patient has described their symptoms. Carefully analyze the symptoms and 
suggest the most appropriate doctor specialization.

Important rules:
- For fever with common symptoms → recommend General Physician first
- For chest pain → recommend Cardiologist
- For persistent neurological symptoms without fever → recommend Neurologist
- For skin issues → recommend Dermatologist
- Always consider the combination of symptoms together
- Never recommend a specialist directly if a General Physician visit is more appropriate first

Context from knowledge base:
{context_text}

Patient symptoms: {symptoms}

Provide:
1. Recommended doctor specialization
2. Reason for recommendation
3. Urgency level (Normal/Urgent/Emergency)
4. Basic advice while waiting for appointment

Answer in a clear, friendly and professional way."""

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