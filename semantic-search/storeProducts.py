# imports
from qdrant_client import QdrantClient;
from constants import keysToRemove
import json

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
    client = QdrantClient(path="./semantic-search/vector_local_db");

# read data from products.json
def getJsonData():
    with open("./semantic-search/products.json", "r") as f:
        data = json.load(f);

    return data;

def getDescriptionOfData(product):
    productDesc = (
        f" {product['title']} \n"
        f" Description: {product['description']} \n"
        f" Category: {product['category']} \n"
        f" Brand: {product['brand']} \n"
        f" Tags: {', '.join(product.get('tags') or [])}\n"
    ) if product.get("title") else ""
    return productDesc;

def getMetadata(product):
    metaData = {
        "id": product.get("id"),
        "rating": product.get("rating"),
        "availibilityStatus": product.get("availibilityStatus"),
        "price": product.get("price"),
    }

    filtered_data = {k: v for k, v in metaData.items() if v is not None}
    return filtered_data

data = getJsonData();
processedData = dropDetailsOfProducts(data);

metadataColumns = getMetadata(processedData[0]);
print(metadataColumns)

description = getDescriptionOfData(processedData[0]);
print(description)
