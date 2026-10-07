# ingest.py
# RAG step 1: take a document and split it into small, overlapping pieces (chunks).


def chunk_text(text, chunk_size=500, overlap=100):
    # chunk_size: how many characters each chunk holds.
    # overlap:    how many characters each chunk repeats from the previous one,
    #             so an idea sitting on a boundary is not cut in half.
    chunks = []
    start = 0                          # left edge of the sliding window
    while start < len(text):
        end = start + chunk_size       # right edge = left edge + window size
        chunks.append(text[start:end]) # grab the slice text[start:end] and keep it
        if end >= len(text):           # reached the end of the document -> stop
            break
        start = end - overlap          # move the window forward, but step back by
                                       # `overlap` so the next chunk repeats the tail
    return chunks


# --- demo: runs ONLY when you execute `python ingest.py` directly, ---
# --- NOT when another module does `from ingest import chunk_text`.   ---
if __name__ == "__main__":
    with open("docs/_sample.txt", encoding="utf-8") as f:
        document = f.read()            # read the whole file into one string

    chunks = chunk_text(document)      # split it into chunks

    print(f"Document length: {len(document)} characters")
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks): # print a short preview of each chunk
        print(f"--- chunk {i} ({len(chunk)} chars) ---")
        print(chunk.strip()[:200], "...")
        print()
