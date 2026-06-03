import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Gemini Client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

st.title("📄 PDF Chatbot")

# Load PDF chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(chunks)

st.success("PDF Loaded Successfully ✅")

# User question
query = st.text_input("Ask a question from the PDF")

if query:

    # Convert query into vector
    q_vec = vectorizer.transform([query])

    # Calculate similarity
    scores = cosine_similarity(q_vec, vectors)[0]

    # Get top 3 relevant chunks
    top_idx = scores.argsort()[-3:][::-1]

    context = ""

    for idx in top_idx:
        if scores[idx] > 0:
            start = max(0, idx - 1)
            end = min(len(chunks), idx + 2)

            context += " ".join(chunks[start:end]) + "\n"

    if context.strip() == "":
        st.warning("No relevant information found in PDF.")
    else:

        prompt = f"""
You are a PDF assistant.

Answer ONLY using the information provided in the context below.

If the context contains partial information,
provide the best possible answer from that information.

Only reply:
'Information not available in document'
if no relevant information exists at all.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("Answer")
            st.write(response.text)

            with st.expander("View Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error: {e}")

            st.subheader("Retrieved PDF Context")
            st.write(context)