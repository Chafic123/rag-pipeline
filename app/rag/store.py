"""Vector store: embeddings, indexing and retrieval with Qdrant (local mode).

    build_index()   load -> chunk -> embed -> store, once at startup
    retrieve()      embed the question -> nearest chunks (HNSW search)
"""
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from app.config import settings
from app.llm import get_openai
from app.rag.ingest import load_chunks


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Open the local Qdrant store on first use.
    Local mode locks the folder to ONE process, so stop the API before running evals."""
    return QdrantClient(path=str(settings.qdrant_path))


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of vectors, one vector per text."""
    response = get_openai().embeddings.create(model=settings.embed_model, input=texts)
    return [item.embedding for item in response.data]


def build_index() -> int:
    """Ingestion pipeline: load every document -> chunk -> embed -> store.
    Each point keeps its text and source filename in the payload, so evaluation
    can check which document a retrieved chunk came from. Returns the chunk count."""
    chunks, sources = load_chunks(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No documents found in {settings.docs_dir}")
    vectors = embed(chunks)

    qdrant = get_qdrant()
    if qdrant.collection_exists(settings.collection):
        qdrant.delete_collection(settings.collection)
    qdrant.create_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(
            size=len(vectors[0]),          # must match the embedding size (1536)
            distance=Distance.COSINE,      # rank by cosine similarity
        ),
        # HNSW: the graph index for fast approximate nearest-neighbour search,
        # instead of comparing the query against every vector (O(n)).
        # BUILD-TIME knobs, they shape the graph once at indexing:
        hnsw_config=HnswConfigDiff(
            m=16,             # edges per node: more edges = better recall, more memory
            ef_construct=100, # effort while building: higher = better graph, slower build
        ),
    )
    qdrant.upsert(
        collection_name=settings.collection,
        points=[
            PointStruct(id=i, vector=vector, payload={"text": chunk, "source": source})
            for i, (vector, chunk, source) in enumerate(zip(vectors, chunks, sources))
        ],
    )
    return len(chunks)


def retrieve(question: str, k: int | None = None):
    """Embed the question and return the k most similar chunks, best first.
    Each point carries payload['text'], payload['source'] and a similarity score.
    A reranker (stage 2) will be layered on top of this later."""
    query_vector = embed([question])[0]
    return get_qdrant().query_points(
        collection_name=settings.collection,
        query=query_vector,
        limit=k or settings.top_k,
        with_payload=True,
        # QUERY-TIME knob: candidates explored while walking the graph.
        # Higher hnsw_ef = better recall but slower: the recall-vs-speed dial.
        search_params=SearchParams(hnsw_ef=64),
    ).points
