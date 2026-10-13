# imports
from qdrant_client import QdrantClient;
import json

def dropDetailsOfProducts(products):
    # List of keys you want to remove
    keysToRemove = [
        "discountPercentage", "stock", "sku", "weight", "dimensions",
        "reviews", "images", "thumbnail", "returnPolicy", 
        "minimumOrderQuantity", "meta", "warrantyInformation", "shippingInformation"
    ]
    
    for product in products:
        for key in keysToRemove:
            # .pop(key, None) deletes the key safely if it exists
            product.pop(key, None) 
            
    return products

def initializeQudrantClient():
    # initialize quadrant local vector db
    client = QdrantClient(path="./semantic-search/vector_local_db");

def getJsonData():
    # read data from products.json
    with open("./semantic-search/products.json", "r") as f:
        data = json.load(f);

    return data;

def getDescriptionOfData(product):
    return product;

def getMetadata(product):
    return product

data = getJsonData();
processedData = dropDetailsOfProducts(data);


temp = getDescriptionOfData(processedData[0]);
print(temp)

