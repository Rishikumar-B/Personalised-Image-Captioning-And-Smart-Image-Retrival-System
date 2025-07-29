from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from deepface import DeepFace
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoFeatureExtractor
import torch
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Image Captioning Model with attention mask fix
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", pad_token='[PAD]')
model = AutoModelForImageTextToText.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def recognize_faces_in_image(filepath):
    """Recognize faces in an image using the reliable search method"""
    try:
        embedding_results = DeepFace.represent(
            img_path=filepath,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        recognized_names = []
        for result in embedding_results:
            if "embedding" in result:  # Check if face was detected
                name = get_face_name(result["embedding"])
                if name:
                    recognized_names.append(name)
        
        # Remove duplicates while preserving order
        seen = set()
        return [name for name in recognized_names if not (name in seen or seen.add(name))]
    except Exception as e:
        logger.error(f"Face recognition error: {str(e)}")
        return []

def get_face_name(face_embedding):
    stored_faces = list(collection.find({}))
    if not stored_faces:
        return None
    
    best_match = None
    best_score = float("inf")
    
    for face in stored_faces:
        stored_embedding = np.array(face["embedding"])
        similarity_score = cosine(face_embedding, stored_embedding)
        if similarity_score < best_score and similarity_score < 0.5:
            best_score = similarity_score
            best_match = face["name"]
    
    return best_match

def personalize_caption(caption, recognized_names):
    """Personalize caption by replacing pronouns and generic terms with recognized names"""
    if not recognized_names:
        return caption, False
    
    is_personalized = False
    personalized = caption
    
    # Handle two-person patterns first
    two_person_patterns = [
        r'a man and a woman',
        r'two men',
        r'two women',
        r'two people',
    ]
    for pattern in two_person_patterns:
        if re.search(pattern, personalized, re.IGNORECASE):
            if len(recognized_names) >= 2:
                replacement = f"{recognized_names[0]} and {recognized_names[1]}"
            elif len(recognized_names) == 1:
                replacement = f"{recognized_names[0]} and someone"
            else:
                replacement = None  # No replacement if no names
            if replacement:
                personalized = re.sub(pattern, replacement, personalized, flags=re.IGNORECASE)
                is_personalized = True
                break  # Only process the first matching pattern

    # Handle single-person patterns if not yet personalized
    if not is_personalized and len(recognized_names) >= 1:
        single_person_patterns = [
            r'a man',
            r'a woman',
            r'a person',
            r'a boy',
            r'a girl',
        ]
        for pattern in single_person_patterns:
            if re.search(pattern, personalized, re.IGNORECASE):
                personalized = re.sub(pattern, recognized_names[0], personalized, flags=re.IGNORECASE)
                is_personalized = True
                break  # Replace first occurrence

    # More sophisticated processing if simple replacements didn't work
    if not is_personalized:
        tokens = word_tokenize(personalized)
        tagged = pos_tag(tokens)
        
        new_tokens = []
        name_index = 0
        num_names = len(recognized_names)
        
        i = 0
        while i < len(tagged):
            word, tag = tagged[i]
            lower_word = word.lower()
            
            # Handle quantity + noun patterns
            if i + 1 < len(tagged) and tagged[i+1][1] in ['NN', 'NNS']:
                next_word = tagged[i+1][0].lower()
                phrase = f"{lower_word} {next_word}"
                
                if name_index < num_names and phrase in ['a man', 'a woman', 'a boy', 'a girl', 'a person']:
                    new_tokens.append(recognized_names[name_index])
                    name_index += 1
                    is_personalized = True
                    i += 2
                    continue
                elif phrase in ['two men', 'two women', 'two boys', 'two girls', 'two people']:
                    if name_index + 1 < num_names:
                        new_tokens.append(f"{recognized_names[name_index]} and {recognized_names[name_index+1]}")
                        name_index += 2
                    else:
                        new_tokens.append(f"{recognized_names[name_index]} and someone")
                        name_index += 1
                    is_personalized = True
                    i += 2
                    continue
            
            # Replace pronouns
            if tag == 'PRP' and lower_word in ['he', 'him', 'his', 'she', 'her'] and name_index < num_names:
                new_tokens.append(recognized_names[name_index])
                name_index += 1
                is_personalized = True
                i += 1
                continue
            elif tag == 'PRP' and lower_word in ['they', 'them', 'their'] and num_names - name_index >= 1:
                if num_names - name_index == 1:
                    new_tokens.append(recognized_names[name_index])
                    name_index += 1
                else:
                    new_tokens.append(' and '.join(recognized_names[name_index:]))
                    name_index = num_names
                is_personalized = True
                i += 1
                continue
            
            new_tokens.append(word)
            i += 1
        
        if is_personalized:
            personalized = ' '.join(new_tokens)
            # Fix common punctuation issues
            personalized = re.sub(r'\s+([,.!?])', r'\1', personalized)
            personalized = re.sub(r" 's", "'s", personalized)
            # Capitalize first letter
            if personalized:
                personalized = personalized[0].upper() + personalized[1:]
    
    return personalized, is_personalized

@app.route("/")
def index():
    return render_template("testindex.html")

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "Image is required"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        # Save file temporarily
        file.save(filepath)
        
        # Verify image is valid
        try:
            image = Image.open(filepath).convert("RGB")
            image.verify()  # Verify it's a valid image
            image = Image.open(filepath).convert("RGB")  # Reopen after verify
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        # Generate caption with attention mask
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        attention_mask = torch.ones_like(pixel_values)
        
        output_ids = model.generate(
            pixel_values,
            attention_mask=attention_mask,
            max_length=16,
            num_beams=4,
            early_stopping=True
        )
        
        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Recognize faces
        recognized_names = recognize_faces_in_image(filepath)
        
        # Personalize caption
        personalized_caption, is_personalized = personalize_caption(caption, recognized_names)
        
        os.remove(filepath)
        return jsonify({
            "caption": caption,
            "personalized_caption": personalized_caption,
            "is_personalized": is_personalized,
            "recognized_names": recognized_names,
            "faces_detected": len(recognized_names) > 0
        }), 200

    except Exception as e:
        logger.error(f"Caption generation error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

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
        embedding_result = DeepFace.represent(
            img_path=filepath, 
            model_name="Facenet", 
            enforce_detection=True,
            detector_backend='opencv'
        )

        if len(embedding_result) == 0:
            os.remove(filepath)
            return jsonify({"error": "No face detected"}), 400

        embedding = embedding_result[0]["embedding"]

        collection.insert_one({
            "name": name,
            "embedding": embedding
        })

        os.remove(filepath)
        return jsonify({"message": f"Face embedding for {name} saved successfully!"}), 200

    except Exception as e:
        logger.error(f"Face upload error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

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
        query_embedding_result = DeepFace.represent(
            img_path=filepath, 
            model_name="Facenet", 
            enforce_detection=True,
            detector_backend='opencv'
        )

        if len(query_embedding_result) == 0:
            os.remove(filepath)
            return jsonify({"error": "No face detected"}), 400

        query_embedding = query_embedding_result[0]["embedding"]
        stored_faces = list(collection.find({}, {"_id": 0, "name": 1, "embedding": 1}))

        if not stored_faces:
            os.remove(filepath)
            return jsonify({"error": "No faces stored in database"}), 400

        best_match = None
        best_score = float("inf")

        for face in stored_faces:
            stored_embedding = np.array(face["embedding"])
            similarity_score = cosine(query_embedding, stored_embedding)

            if similarity_score < best_score:
                best_score = similarity_score
                best_match = face["name"]

        os.remove(filepath)
        if best_match and best_score < 0.5:
            return jsonify({
                "message": f"Best match found: {best_match}",
                "name": best_match,
                "similarity": f"{1 - best_score:.2f}"
            }), 200
        else:
            return jsonify({"message": "No match found"}), 404

    except Exception as e:
        logger.error(f"Face search error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)