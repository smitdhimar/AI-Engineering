# imports
from qdrant_client import QdrantClient, models;
from constants import COLLECTION_NAME, VECTOR_SIZE, DB_PATHkeysToRemove, voyageApiKey, VOYAGE_EMBEDDING_MODEL, VOYAGE_EMBEDDINGS_URL
from uuid import uuid5, NAMESPACE_URL
import json
import urllib.request
import urllib.error

# drop unncessory data
def dropDetailsOfProducts(products):
    
    # List of keys you want to remove
    for product in products:
        for key in keysToRemove:
            # .pop(key, None) deletes the key safely if it exists
            product.pop(key, None) 
            
    return products

# initialize quadrant local vector db
def initializeQudrantClient():
    return QdrantClient(path=DB_PATH);

# read data from products.json
def getJsonData():
    with open("./semantic-search/products.json", "r") as f:
        data = json.load(f);

    return data;

# get description from data
def getDescriptionOfData(product):
    productDesc = (
        f" {product['title']} \n"
        f" Description: {product['description']} \n" if product.get("description") else ""
        f" Category: {product['category']} \n" if product.get("category") else ""
        f" Brand: {product['brand']} \n" if product.get("brand") else ""
        f" Tags: {', '.join(product.get('tags') or [])}\n" if product.get("tags") else ""
    ) if product.get("title") else ""
    return productDesc;

# get metadata object
def getMetadata(product):
    metaData = {
        "id": product.get("id"),
        "rating": product.get("rating"),
        "availibilityStatus": product.get("availibilityStatus"),
        "price": product.get("price"),
    }

    filtered_data = {k: v for k, v in metaData.items() if v is not None}
    return filtered_data

# call voyage-4-lite to generate embeddings
def httpCall(texts, inputType="document"):
    # allow a single string or a list of strings
    if isinstance(texts, str):
        texts = [texts];

    payload = {
        "input": texts,
        "model": VOYAGE_EMBEDDING_MODEL,
        "input_type": inputType,
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
            result = json.loads(response.read().decode("utf-8"));
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8");
        raise RuntimeError(f"Voyage API error {error.code}: {detail}") from error;

    # response.data is a list of {"object", "embedding", "index"}
    embeddings = [item["embedding"] for item in result["data"]];
    totalTokens = result.get("usage", {}).get("total_tokens");
    print(f"Voyage tokens used: {totalTokens}");

    return embeddings;


# Store embeddings with product metadata
def storeEmbeddings(client, products, embeddings):
    if len(products) != len(embeddings):
        raise ValueError(
            "Number of products must match number of embeddings"
        )

    if not embeddings:
        return

    vector_size = len(embeddings[0])

    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    points = []

    for product, embedding in zip(products, embeddings):
        product_id = product.get("id")

        if product_id is None:
            raise ValueError("Every product must have an id")

        # Qdrant point IDs must be unsigned integers or UUIDs.
        # Generate a deterministic UUID from the product ID.
        point_id = str(
            uuid5(NAMESPACE_URL, f"products/{product_id}")
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    **getMetadata(product),
                    "title": product.get("title"),
                    "description": product.get("description"),
                    "category": product.get("category"),
                    "brand": product.get("brand"),
                    "tags": product.get("tags", []),
                },
            )
        )

        # Insert or update points
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Stored {len(points)} product embeddings in Qdrant.")


data = getJsonData();
processedData = dropDetailsOfProducts(data);

print(len(processedData))

metadataColumns = getMetadata(processedData[0]);
print(metadataColumns)

description = getDescriptionOfData(processedData[0]);
print(description)

# generate embeddings for every product description via voyage-4-lite
descriptions = [getDescriptionOfData(product) for product in processedData];
embeddings = httpCall(descriptions);
print(f"Generated {len(embeddings)} embeddings (dimension {len(embeddings[0])})");

client = initializeQudrantClient();

storeEmbeddings(
    client=client,
    products=processedData,
    embeddings=embeddings,
)

client.close()