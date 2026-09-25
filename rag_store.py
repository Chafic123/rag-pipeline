# rag_store.py
# Full RAG, refactored so the pieces are reusable and safe to import.
#
# THE FLOW when answering one question (answer()):
#   1. Question comes in.
#   2. Embed the question into a vector.
#   3. Retrieve the most similar chunks  -> this is the CONTEXT.
#   4. Augment: prompt = instructions + context + question.
#   5. Generate: send the prompt to the LLM.
#   6. Return the grounded answer.
#
# build_index() runs the ingestion pipeline ONCE: load -> chunk -> embed -> store.

import os

from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingest import chunk_text

load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
qdrant = QdrantClient(path="qdrant_data")   # local vector DB, shared by both functions

EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "documents"


def embed(texts):
    """Turn a list of texts into a list of vectors (one vector per text)."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index(path="docs/sample.txt"):
    """Ingestion pipeline: load a document -> chunk -> embed -> store in the vector DB."""
    # load
    with open(path, encoding="utf-8") as f:
        document = f.read()
    # chunk
    chunks = chunk_text(document)
    # embed
    vectors = embed(chunks)
    # store: (re)create the collection, then write the points
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=len(vectors[0]),          # must match the embedding size (1536)
            distance=Distance.COSINE,      # rank by cosine similarity
        ),
    )
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)                     # how many chunks were stored


def answer(question, k=2):
    """Full RAG for one question: Retrieve -> Augment -> Generate."""
    # STEP 2 - embed the question into a vector
    query_vector = embed([question])[0]

    # STEP 3 - retrieve the k chunks most similar to the question
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points
    context = "\n\n".join(point.payload["text"] for point in results)

    # STEP 4 - augment: build the prompt with the retrieved context
    prompt = (
        "Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    # STEP 5 - generate the answer from the augmented prompt
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    # STEP 6 - return the grounded answer
    return response.choices[0].message.content


# This block runs ONLY when you run `python rag_store.py` directly.
# When another file imports this module (like the API will), it does NOT run,
# so importing will not rebuild the index or print anything.
if __name__ == "__main__":
    count = build_index()
    print(f"Stored {count} chunks in the '{COLLECTION}' collection.\n")

    question = "How does retrieval augmented generation work?"
    print(f"Question: {question}\n")
    print(answer(question))
