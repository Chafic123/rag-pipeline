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
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, SearchParams

from pathlib import Path

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


def build_index(docs_dir="docs"):
    """Ingestion pipeline: load EVERY document in docs_dir -> chunk -> embed -> store.
    Each chunk remembers which file it came from (payload 'source'), so evaluation
    can later check whether retrieval found the right document.
    Files whose name starts with '_' are ignored."""
    # load + chunk every document, remembering the source file for each chunk
    chunks, sources = [], []
    paths = sorted(list(Path(docs_dir).glob("*.txt")) + list(Path(docs_dir).glob("*.md")))
    for path in paths:
        if path.name.startswith("_"):      # underscore-prefixed files are skipped
            continue
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(path.name)

    # embed all chunks in a single API call
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
        # HNSW is the graph index Qdrant uses for fast approximate nearest-neighbour
        # search instead of comparing the query against every vector (O(n)).
        # These are the BUILD-TIME knobs (they shape the graph, once, at indexing):
        hnsw_config=HnswConfigDiff(
            m=16,             # edges per node: more edges = better recall, more memory
            ef_construct=100, # effort while BUILDING the graph: higher = better graph, slower to build
        ),
    )
    # each point stores the chunk text AND its source filename
    points = [
        PointStruct(id=i, vector=vectors[i],
                    payload={"text": chunks[i], "source": sources[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)                     # how many chunks were stored


def retrieve(question, k=3):
    """STAGE 1 retrieval: embed the question and return the k most similar chunks.
    Each returned point carries payload['text'] and payload['source'], plus a score.
    The reranker (stage 2) will be layered on top of this later."""
    query_vector = embed([question])[0]
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True,
        # QUERY-TIME knob: how many candidates to explore while walking the graph.
        # Higher hnsw_ef = better recall but slower. This is the recall-vs-speed dial.
        search_params=SearchParams(hnsw_ef=64),
    ).points
    return results


def answer(question, k=3):
    """Full RAG for one question: Retrieve -> Augment -> Generate."""
    # STEP 2 + 3 - retrieve the k chunks most similar to the question
    results = retrieve(question, k=k)
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
