Title:Personalised-Image-Captioning-And-Smart-Image-Retrival-System

Introduction:

This project was built as a pre-final year project during our 6th semester. The basic idea of our project is "smart" or "speed" searching that means enabling faster search using custom captions in our gallery

This is not yet a full fledged project like a complete app or cloud service, but we plan to develop it into one. In this project, we extract captions and metadata from images using machine learning and DL such as Blip, vit-gpt2, Deepface facenet, YOLO and K-Nearest Neighbors. The extracted data is then stored in a DB

For searching, we use sematic search powered by the all-MiniLM-L6-v2 model . This allows us to search both: captions from images images from captions

Models: BLIP + ViT + GPT-2 Input: Image → Output: Generated Caption

Modules: DeepFace (FaceNet) Input: Image → Output: Face Embedding

YOLO (Ultralytics) Input: Image → Output: Detected Objects with Bounding Boxes

K-Nearest Neighbors (KNN) Input: Feature Vector → Output: Predicted Class

KMeans (Optional) Input: Feature Vectors → Output: Cluster Labels

SentenceTransformer (all-MiniLM-L6-v2) Input: Text or Caption → Output: Semantic Embedding Vector

MTCNN + InceptionResnetV1 (FaceNet via facenet-pytorch) Input: Image → Output: Aligned Face & 512-D Embedding

torch Core PyTorch library for deep learning and tensor computations.

transformers Hugging Face library for using pre-trained transformer models like GPT-2, BERT, etc.

sentence-transformers Enables semantic similarity and embedding generation for sentences using transformer models.

facenet-pytorch Face detection and recognition models (MTCNN + InceptionResnetV1) in PyTorch.

deepface Lightweight face recognition and facial attribute analysis framework.

ultralytics Official YOLOv5/YOLOv8 implementation for real-time object detection.

opencv-python OpenCV library for image and video processing tasks.

Pillow Python Imaging Library (PIL fork) for image manipulation and file I/O.

torchvision Image datasets, pre-trained models, and transforms for PyTorch.

matplotlib Visualization library for plotting graphs, images, and data.

scikit-learn Machine learning library for classification, clustering, and more.

numpy Core scientific computing library for numerical operations and arrays.

pymongo Python driver for MongoDB database interaction.
