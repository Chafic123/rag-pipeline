# rag_store.py
# Full RAG in one file.
#
# ---------------------------------------------------------------------------
# THE FLOW when answering one question (this is what answer() does):
#   1. Question comes in (hardcoded now, will be the USER's input via the API).
#   2. Embed the question into a vector.
#   3. Retrieve the most similar chunks from the vector store  -> this is the CONTEXT.
#   4. Augment: build a prompt = instructions + retrieved context + question.
#   5. Generate: send that prompt to the LLM as a message.
#   6. Return the model's answer, grounded in the retrieved context.
#
# BUILD-THE-INDEX steps (run once, in the numbered section below):
#   load document -> chunk -> embed -> store in the vector database.
# ---------------------------------------------------------------------------

import os

from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingest import chunk_text          # reuse the chunker from step 1

load_dotenv()                          # load OPENAI_API_KEY from the .env file
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small" # embedding model (outputs 1536 numbers)
COLLECTION = "documents"               # name of our "table" of vectors


def embed(texts):
    """Turn a list of texts into a list of vectors (one vector per text)."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def answer(question, k=2):
    """Full RAG for one question: Retrieve -> Augment -> Generate."""
    # STEP 2 - embed the question into a vector
    query_vector = embed([question])[0]

    # STEP 3 - retrieve: find the k chunks most similar to the question
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,                                        # how many chunks to pull
        with_payload=True,
    ).points
    # join the retrieved chunk texts into one context string
    context = "\n\n".join(point.payload["text"] for point in results)

    # STEP 4 - augment: build the prompt = instructions + context + question.
    # The first sentence keeps the model grounded so it does not make things up.
    prompt = (
        "Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    # STEP 5 - generate: the LLM answers using that augmented prompt
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # STEP 6 - return the grounded answer
    return response.choices[0].message.content


# === BUILD THE INDEX (run once) ===

# 1. Load the document and split it into chunks
with open("docs/sample.txt", encoding="utf-8") as f:
    document = f.read()
chunks = chunk_text(document)

# 2. Turn every chunk into an embedding vector (one API call for all of them)
vectors = embed(chunks)

# 3. Create the vector store (a "collection" is a table just for vectors)
qdrant = QdrantClient(path="qdrant_data")      # local database saved on disk
if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)       # wipe any old version so re-runs are clean
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=len(vectors[0]),                  # must match the embedding size (1536)
        distance=Distance.COSINE,              # rank by cosine similarity
    ),
)

# 4. Store each chunk as a "point": an id, its vector, and its original text
points = [
    PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
    for i in range(len(chunks))
]
qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"Stored {len(points)} chunks in the '{COLLECTION}' collection.\n")

# === ASK A QUESTION (STEP 1) ===
question = "How does retrieval augmented generation work?"
print(f"Question: {question}\n")
print(answer(question))
