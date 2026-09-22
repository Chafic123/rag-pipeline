import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Create OpenAI client using API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


sentences = [
    "The bank approved my loan application.",
    "I need to transfer money to my savings account.",
    "My credit card payment is due next week.",
    "The API returned a 500 server error.",
    "I fixed the bug in the Python function.",
    "We deployed the new microservice to production.",
    "This pizza recipe uses fresh basil.",
    "I love a good cup of coffee in the morning.",
    "It might rain heavily this afternoon.",
    "The sunset over the mountains was beautiful.",
]

# Convert all 10 sentences into embedding vectors
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences,
)

# Extract the vectors from the API response
vectors = [item.embedding for item in response.data]


def cosine(a, b):
    # Convert lists into NumPy arrays
    a, b = np.array(a), np.array(b)

    # Calculate cosine similarity:
    # dot product / (magnitude of A × magnitude of B)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Use the first sentence as our query
query_index = 0

print(f"Query: {sentences[query_index]}")
print(f"(each embedding has {len(vectors[0])} numbers)\n")


# Store similarity scores for each comparison
scores = []

# “Use sentence 0 as my query, and find which of the other sentences are most similar to it.”
for i, sentence in enumerate(sentences):

    # Don't compare the query with itself
    if i == query_index:
        continue

    # Compare the query vector with the current sentence vector
    score = cosine(vectors[query_index], vectors[i])

    # Store both the score and the sentence
    scores.append((score, sentence))


# Sort by similarity: highest score first
for score, sentence in sorted(scores, reverse=True):

    # Print similarity score and matching sentence
    print(f"{score:.3f}  {sentence}")