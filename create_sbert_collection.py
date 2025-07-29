from pymongo import MongoClient
from pymongo.operations import IndexModel

def create_sbert_embed_collection():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_recognition"]
    
    # Collection configuration
    collection_name = "sbert_embed"
    
    # Delete existing collection if it exists (optional)
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        print(f"Dropped existing '{collection_name}' collection")
    
    # Create new collection with schema validation
    db.create_collection(
        collection_name,
        validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["image_path", "upload_time", "caption", "sbert_embedding"],
                "properties": {
                    "image_path": {"bsonType": "string"},
                    "upload_time": {"bsonType": "date"},
                    "caption": {"bsonType": "string"},
                    "sbert_embedding": {
                        "bsonType": "array",
                        "items": {"bsonType": "double"}
                    },
                    "is_personalized": {"bsonType": "bool"},
                    "recognized_names": {
                        "bsonType": "array",
                        "items": {"bsonType": "string"}
                    }
                }
            }
        }
    )
    
    # Create indexes
    indexes = [
        IndexModel([("caption", "text")], name="caption_text_index"),
        IndexModel([("upload_time", -1)], name="upload_time_desc_index"),
        IndexModel([("recognized_names", 1)], name="recognized_names_index"),
        IndexModel([("is_personalized", 1)], name="is_personalized_index")
    ]
    
    db[collection_name].create_indexes(indexes)
    print(f"Created '{collection_name}' collection with schema validation and indexes")

if __name__ == "__main__":
    create_sbert_embed_collection()