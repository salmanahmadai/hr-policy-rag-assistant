
import re
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("HR Policy Assistant")
st.write("Ask questions about your company's HR policies.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

uploaded_pdf = st.file_uploader(
    "Upload HR Employee Handbook",
    type=["pdf"]
)

api_key = st.text_input(
    "Groq API Key",
    type="password"
)

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)

def split_into_chunks(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    headings = [
        "Annual Leave",
        "Sick Leave",
        "Work From Home",
        "Working Hours",
        "Late Arrival",
        "Travel Reimbursement",
        "Meal Allowance",
        "Internet Reimbursement",
        "Learning Budget",
        "Password Policy",
        "Phishing",
        "Conflict of Interest",
        "Gifts",
        "Performance Review",
        "Internal Mobility"
    ]

    pattern = (
        r"(?im)^\s*(?:HR\s*\d+\s*[-:.)]?\s*)?"
        r"(?=" + "|".join(re.escape(h) for h in headings) + r")"
    )

    parts = re.split(pattern, text)
    chunks = [part.strip() for part in parts if part.strip()]

    return chunks

if uploaded_pdf is not None:
    if st.button("Process HR Handbook"):
        if not api_key.strip():
            st.error("Please enter your Groq API key.")
        else:
            with st.spinner("Processing handbook..."):
                try:
                    raw_text = extract_text(uploaded_pdf)
                    chunks = split_into_chunks(raw_text)

                    if not raw_text.strip():
                        st.error(
                            "No readable text found. "
                            "The PDF may be scanned."
                        )
                    elif not chunks:
                        st.error("No text chunks could be created.")
                    else:
                        client = chromadb.Client()

                        try:
                            client.delete_collection("hr_policies")
                        except Exception:
                            pass

                        collection = client.create_collection(
                            name="hr_policies"
                        )

                        embeddings = embedding_model.encode(
                            chunks
                        ).tolist()

                        collection.add(
                            ids=[
                                f"policy_{i}"
                                for i in range(len(chunks))
                            ],
                            documents=chunks,
                            embeddings=embeddings
                        )

                        st.session_state["processed"] = True
                        st.session_state["chunks"] = chunks
                        st.session_state["api_key"] = api_key
                        st.session_state["client"] = client

                        st.success(
                            f"Handbook processed: {len(chunks)} chunks"
                        )

                except Exception as e:
                    st.error(f"Processing error: {e}")

if st.session_state.get("processed", False):
    st.divider()
    st.subheader("Ask an HR Policy Question")

    question = st.text_input(
        "Enter your question",
        key="hr_question"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching policies and generating answer..."):
                try:
                    collection = st.session_state[
                        "client"
                    ].get_collection("hr_policies")

                    query_embedding = embedding_model.encode(
                        [question]
                    ).tolist()

                    results = collection.query(
                        query_embeddings=query_embedding,
                        n_results=min(3, collection.count())
                    )

                    retrieved_docs = results["documents"][0]
                    context = "\n\n".join(retrieved_docs)

                    groq_client = Groq(
                        api_key=st.session_state["api_key"]
                    )

                    prompt = f"""
You are an HR policy assistant.

Answer the user's question using only the policy context below.
Do not invent rules, numbers, dates, or benefits.
If the context does not contain the answer, say:
"I could not find this information in the HR handbook."

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, concise answer.
"""

                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You answer HR questions using "
                                    "only the provided evidence."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0
                    )

                    answer = response.choices[0].message.content

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Retrieved Sources")

                    for i, doc in enumerate(
                        retrieved_docs, start=1
                    ):
                        with st.expander(f"Source {i}"):
                            st.write(doc)

                except Exception as e:
                    st.error(f"Answer generation error: {e}")
