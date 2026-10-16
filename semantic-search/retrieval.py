# imports
from qdrant_client import QdrantClient
from constants import COLLECTION_NAME, DB_PATH, voyageApiKey, VOYAGE_EMBEDDING_MODEL, VOYAGE_EMBEDDINGS_URL
import json
import urllib.request
import urllib.error

# open the local vector db
def initializeQdrantClient():
    return QdrantClient(path=DB_PATH)

# embed the search query with voyage-4-lite (input_type="query")
def embedQuery(query):
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    payload = {
        "input": [query],
        "model": VOYAGE_EMBEDDING_MODEL,
        "input_type": "query",
    }

    request = urllib.request.Request(
        VOYAGE_EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {voyageApiKey}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"Voyage API error {error.code}: {detail}") from error

    # single query in -> single embedding out
    return result["data"][0]["embedding"]

# fetch the products most similar to the query
# scoreThreshold: keep only results with score >= threshold (None = no filtering).
# Using COSINE distance, so scores are in [-1, 1] and higher means more similar.
def search(query, limit=5, scoreThreshold=None, client=None, collectionName=COLLECTION_NAME):
    ownsClient = client is None
    if ownsClient:
        client = initializeQdrantClient()

    try:
        queryVector = embedQuery(query)

        response = client.query_points(
            collection_name=collectionName,
            query=queryVector,
            limit=limit,
            score_threshold=scoreThreshold,
            with_payload=True,
        )

        return [
            {"score": point.score, "product": point.payload}
            for point in response.points
        ]
    finally:
        # only close the client we opened ourselves
        if ownsClient:
            client.close()

# pretty print the results
def printResults(query, results):
    print(f"\nQuery: {query}")
    print("-" * 70)

    if not results:
        print("No matches found.")
        return

    for rank, result in enumerate(results, start=1):
        product = result["product"]
        print(f"{rank}. {product.get('title')}  (score: {result['score']:.4f})")
        print(
            f"   Brand: {product.get('brand')} | "
            f"Category: {product.get('category')} | "
            f"Price: {product.get('price')} | "
            f"Rating: {product.get('rating')}"
        )
        print(f"   {product.get('description')}")
        print()

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "beauty products"
    limit = 5

    client = initializeQdrantClient()
    try:
        results = search(query,scoreThreshold=0.5, limit=limit, client=client)
        printResults(query, results)
    finally:
        client.close()
