keysToRemove = [
        "discountPercentage", "stock", "sku", "weight", "dimensions",
        "reviews", "images", "thumbnail", "returnPolicy", 
        "minimumOrderQuantity", "meta", "warrantyInformation", "shippingInformation"
    ]

# voyage ai embedding config
VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings";
VOYAGE_EMBEDDING_MODEL = "voyage-4-lite";

# vector db configs
COLLECTION_NAME = "products"
VECTOR_SIZE = 1024
DB_PATH = "./semantic-search/vector_local_db"