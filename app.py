from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import logging
from demographic_utils import augment_features_with_demographics

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained LipNet embedding model
try:
    # We use the embedding model which outputs features from the last LSTM layer
    lipnet_model = load_model('models/lipnet_embedding_model.h5')
    logger.info("LipNet embedding model loaded successfully")
except Exception as e:
    logger.error(f"Error loading LipNet embedding model: {e}")
    lipnet_model = None

# Load the ICU classifier
try:
    with open('models/icu_classifier.pkl', 'rb') as f:
        icu_classifier = pickle.load(f)
    logger.info("ICU classifier loaded successfully")
except Exception as e:
    logger.error(f"Error loading ICU classifier: {e}")
    icu_classifier = None

# Define the ICU phrases
ICU_PHRASES = [
    "Call the nurse",
    "Help me",
    "I cant breathe",
    "I feel sick",
    "I feel tired"
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict ICU phrase from a lip-reading video"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty video file"}), 400
    
    # Get demographic information if provided
    demographic = request.form.get('demographic', 'male_under_50')  # Default to male_under_50 if not specified
    
    try:
        # Save the uploaded video temporarily
        temp_path = 'temp_video.mp4'
        video_file.save(temp_path)
        
        # Extract frames from the video
        frames = extract_frames(temp_path)
        
        # Get LipNet embeddings
        embeddings = get_lipnet_embeddings(frames)
        
        # Augment embeddings with demographic features
        video_path = temp_path  # Use the temp path for demographic extraction
        augmented_embeddings = augment_features_with_demographics(embeddings, video_path, demographic)
        
        # Make prediction using the classifier
        prediction_idx, confidence = predict_phrase(augmented_embeddings)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            "phrase": ICU_PHRASES[prediction_idx],
            "confidence": float(confidence)
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

def extract_frames(video_path, max_frames=75):
    """Extract frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame (grayscale, resize to match model input)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face and extract mouth region
        # For simplicity, we'll assume the video is already mouth-cropped
        # In a real implementation, you would add face detection and mouth extraction here
        
        # Resize to match LipNet input dimensions (46x140)
        resized = cv2.resize(gray, (140, 46))
        normalized = resized / 255.0  # Normalize pixel values
        
        frames.append(normalized)
    
    cap.release()
    
    # Pad if needed to reach max_frames
    if len(frames) < max_frames:
        last_frame = frames[-1] if frames else np.zeros((46, 140))
        frames.extend([last_frame] * (max_frames - len(frames)))
    
    return np.array(frames)

def get_lipnet_embeddings(frames):
    """Get embeddings from the LipNet model"""
    if lipnet_model is None:
        raise ValueError("LipNet embedding model not loaded")
    
    # Reshape for model input (add batch dimension and channel dimension)
    input_frames = frames.reshape(1, frames.shape[0], frames.shape[1], frames.shape[2], 1)
    
    # Get embeddings from the model
    # This will output the activations from the last LSTM layer
    embeddings = lipnet_model.predict(input_frames)
    
    # Flatten the embeddings to a 1D vector by taking the mean across the time dimension
    # This simplifies the classifier input
    flattened_embeddings = np.mean(embeddings, axis=1).flatten()
    
    return flattened_embeddings

def predict_phrase(embeddings):
    """Predict the ICU phrase using the enhanced classifier pipeline"""
    if icu_classifier is None:
        raise ValueError("ICU classifier not loaded")
    
    # For pipeline classifiers, we don't need to scale the data manually
    # as it's handled by the pipeline
    
    # Get prediction probabilities
    try:
        # First try the pipeline approach (new enhanced classifier)
        probs = icu_classifier.predict_proba([embeddings])[0]
    except AttributeError:
        # Fall back to the old classifier format if needed
        logger.warning("Using legacy classifier format")
        probs = icu_classifier.predict_proba([embeddings])[0]
    
    # Get the index of the highest probability
    prediction_idx = np.argmax(probs)
    confidence = probs[prediction_idx]
    
    # Add confidence threshold for more reliable predictions
    # If confidence is too low, we might want to indicate uncertainty
    if confidence < 0.4:
        logger.warning(f"Low confidence prediction: {confidence:.2f}")
    
    return prediction_idx, confidence

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
