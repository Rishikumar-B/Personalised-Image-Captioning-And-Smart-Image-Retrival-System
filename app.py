from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from deepface import DeepFace
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoFeatureExtractor
import torch
from PIL import Image

app = Flask(__name__)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Image Captioning Model
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = AutoModelForImageTextToText.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning") 

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/")
def index():
    return render_template("index.html")

# ----------------- FACE REGISTRATION -----------------
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Image and name are required"}), 400

    file = request.files["image"]
    name = request.form["name"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Extract face embeddings
        embedding_result = DeepFace.represent(img_path=filepath, model_name="Facenet", enforce_detection=True)

        if len(embedding_result) == 0:
            return jsonify({"error": "No face detected"}), 400

        embedding = embedding_result[0]["embedding"]

        # Store in MongoDB
        collection.insert_one({
            "name": name,
            "embedding": embedding
        })

        return jsonify({"message": f"Face embedding for {name} saved successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- FACE SEARCH -----------------
@app.route("/search", methods=["POST"])
def search_face():
    if "image" not in request.files:
        return jsonify({"error": "Image is required"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Extract embeddings of the uploaded image
        query_embedding_result = DeepFace.represent(img_path=filepath, model_name="Facenet", enforce_detection=True)

        if len(query_embedding_result) == 0:
            return jsonify({"error": "No face detected"}), 400

        query_embedding = query_embedding_result[0]["embedding"]

        # Retrieve all stored embeddings from MongoDB
        stored_faces = list(collection.find({}, {"_id": 0, "name": 1, "embedding": 1}))

        if not stored_faces:
            return jsonify({"error": "No faces stored in database"}), 400

        # Find the closest match using cosine similarity
        best_match = None
        best_score = float("inf")  # Cosine similarity (lower is better)

        for face in stored_faces:
            stored_embedding = np.array(face["embedding"])
            similarity_score = cosine(query_embedding, stored_embedding)

            if similarity_score < best_score:
                best_score = similarity_score
                best_match = face["name"]

        if best_match:
            return jsonify({"message": f"Best match found: {best_match} with similarity {1 - best_score:.2f}"}), 200
        else:
            return jsonify({"message": "No match found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- IMAGE CAPTIONING -----------------
@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "Image is required"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Open image
        image = Image.open(filepath).convert("RGB")
        
        # Process image for model
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)

        # Generate caption
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return jsonify({"caption": caption}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
