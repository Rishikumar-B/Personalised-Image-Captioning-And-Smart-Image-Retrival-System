from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import requests
import shutil
from scipy.spatial.distance import cosine
import tempfile
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
CAPTIONED_IMAGES_FOLDER = "static/captioned_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAPTIONED_IMAGES_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CAPTIONED_IMAGES_FOLDER"] = CAPTIONED_IMAGES_FOLDER

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
sbert_collection = db["sbert_embed"]

# Initialize SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# URL of your testapp.py server
TESTAPP_URL = "http://localhost:5000"

@app.route("/")
def home():
    return render_template("captionsearch.html")

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {filepath}: {str(e)}")
        return False

@app.route("/get_images")
def get_images():
    try:
        images = []
        for doc in sbert_collection.find({}):
            images.append({
                "filename": doc["image_path"],
                "caption": doc["caption"],
                "upload_time": doc["upload_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "is_personalized": doc.get("is_personalized", False),
                "recognized_names": doc.get("recognized_names", [])
            })
        return jsonify({"images": images})
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/delete_images", methods=["POST"])
def delete_images():
    try:
        data = request.get_json()
        if not data or "images" not in data:
            return jsonify({"error": "No images specified"}), 400
        
        deleted_count = 0
        for filename in data["images"]:
            try:
                # Delete from filesystem
                filepath = os.path.join(app.config["CAPTIONED_IMAGES_FOLDER"], filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Delete from database
                result = sbert_collection.delete_one({"image_path": filename})
                if result.deleted_count > 0:
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {filename}: {str(e)}")
                continue
        
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "total_requested": len(data["images"])
        })
    except Exception as e:
        logger.error(f"Error in delete_images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload_images", methods=["POST"])
def upload_images():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "No selected files"}), 400
    
    results = []
    for file in files:
        if file.filename == "":
            continue
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        
        try:
            # Save the file temporarily
            file.save(temp_path)
            
            # Validate image before processing
            if not is_valid_image(temp_path):
                raise Exception("Invalid image file")
            
            # Call testapp.py's generate_caption endpoint
            with open(temp_path, 'rb') as f:
                response = requests.post(
                    f"{TESTAPP_URL}/generate_caption",
                    files={"image": f},
                    timeout=60  # Increased timeout
                )
            
            if response.status_code != 200:
                error_msg = response.text if response.status_code != 200 else "Caption generation failed"
                raise Exception(error_msg)
            
            caption_data = response.json()
            personalized_caption = caption_data.get("personalized_caption", "")
            
            # Generate SBERT embedding
            sbert_embedding = sbert_model.encode(personalized_caption).tolist()
            
            # Save image to permanent storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{secure_filename(file.filename)}"
            new_filepath = os.path.join(app.config["CAPTIONED_IMAGES_FOLDER"], new_filename)
            
            shutil.move(temp_path, new_filepath)
            
            # Store in MongoDB
            doc = {
                "image_path": new_filename,
                "upload_time": datetime.now(),
                "caption": personalized_caption,
                "sbert_embedding": sbert_embedding,
                "is_personalized": caption_data.get("is_personalized", False),
                "recognized_names": caption_data.get("recognized_names", [])
            }
            sbert_collection.insert_one(doc)
            
            results.append({
                "filename": new_filename,
                "caption": personalized_caption,
                "is_personalized": caption_data.get("is_personalized", False),
                "recognized_names": caption_data.get("recognized_names", []),
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
        finally:
            # Clean up
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    # Return results with success/failure status
    success_count = sum(1 for r in results if r.get("status") == "success")
    if success_count == 0:
        return jsonify({
            "error": "No files were processed successfully",
            "details": results
        }), 400
    
    return jsonify({
        "message": f"Processed {success_count}/{len(files)} files",
        "results": results
    }), 200

@app.route("/search_captions", methods=["GET", "POST"])
def search_captions():
    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        if not query_text:
            return render_template("search_results.html", 
                                error="Query text is required",
                                results=[])
        
        try:
            # Generate embedding for query
            query_embedding = sbert_model.encode(query_text)
            
            # Find similar captions
            results = []
            for doc in sbert_collection.find({}):
                try:
                    stored_embedding = doc["sbert_embedding"]
                    similarity = 1 - cosine(query_embedding, stored_embedding)
                    if similarity > 0.4:  # Lowered threshold to get more results
                        results.append({
                            "image_path": doc["image_path"],
                            "caption": doc["caption"],
                            "similarity": float(similarity),
                            "upload_time": doc["upload_time"].strftime("%Y-%m-%d %H:%M:%S"),
                            "is_personalized": doc.get("is_personalized", False)
                        })
                except KeyError:
                    continue
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return render_template("search_results.html", 
                                query=query_text, 
                                results=results,
                                error=None)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return render_template("search_results.html",
                                query=query_text,
                                results=[],
                                error=str(e))
    
    return render_template("search_captions.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, threaded=True)