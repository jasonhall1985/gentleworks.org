import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the ICU phrases
ICU_PHRASES = [
    "I need water",
    "I am in pain",
    "I need help",
    "I can't breathe",
    "Call the nurse",
    "I feel sick",
    "I'm cold",
    "I'm hot",
    "Thank you",
    "I need medication"
]

def load_lipnet_model():
    """Load the pre-trained LipNet embedding model"""
    try:
        # We use the embedding model which outputs features from the last LSTM layer
        model = load_model('models/lipnet_embedding_model.h5')
        logger.info("LipNet embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading LipNet embedding model: {e}")
        raise

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

def get_lipnet_embeddings(frames, model):
    """Get embeddings from the LipNet model"""
    # Reshape for model input (add batch dimension and channel dimension)
    input_frames = frames.reshape(1, frames.shape[0], frames.shape[1], frames.shape[2], 1)
    
    # Get embeddings from the model
    # This will output the activations from the last LSTM layer
    embeddings = model.predict(input_frames)
    
    # Flatten the embeddings to a 1D vector by taking the mean across the time dimension
    # This simplifies the classifier input
    flattened_embeddings = np.mean(embeddings, axis=1).flatten()
    
    return flattened_embeddings

def load_training_data(data_dir='data'):
    """Load training data from the data directory"""
    X = []
    y = []
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Load LipNet model
    lipnet_model = load_lipnet_model()
    
    # Process 5 videos for each of the 10 phrases
    for phrase_idx, phrase in enumerate(ICU_PHRASES):
        # Create a directory name from the phrase
        phrase_dir = phrase.lower().replace(" ", "_")
        phrase_path = os.path.join(data_dir, phrase_dir)
        
        if not os.path.exists(phrase_path):
            logger.warning(f"Directory not found: {phrase_path}")
            continue
        
        # Get all video files in the directory
        video_files = [f for f in os.listdir(phrase_path) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if len(video_files) == 0:
            logger.warning(f"No video files found in {phrase_path}")
            continue
        
        logger.info(f"Processing {len(video_files)} videos for phrase: {phrase}")
        
        # Process each video file
        for video_file in video_files[:5]:  # Limit to 5 videos per phrase
            video_path = os.path.join(phrase_path, video_file)
            
            try:
                # Extract frames
                frames = extract_frames(video_path)
                
                # Get LipNet embeddings
                embeddings = get_lipnet_embeddings(frames, lipnet_model)
                
                # Add to dataset
                X.append(embeddings)
                y.append(phrase_idx)
                
                logger.info(f"Processed {video_file} for phrase: {phrase}")
                
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
    
    return np.array(X), np.array(y)

def train_classifier(X, y):
    """Train a classifier on the embeddings"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Classifier accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=ICU_PHRASES))
    
    return clf

def save_classifier(clf, output_path='models/icu_classifier.pkl'):
    """Save the trained classifier"""
    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    logger.info(f"Classifier saved to {output_path}")

def main():
    """Main function to train and save the classifier"""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("Starting training process...")
    
    # Load training data
    logger.info("Loading training data...")
    X, y = load_training_data()
    
    if len(X) == 0:
        logger.error("No training data found. Please ensure data directory is populated.")
        return
    
    logger.info(f"Loaded {len(X)} samples across {len(set(y))} classes")
    
    # Train classifier
    logger.info("Training classifier...")
    clf = train_classifier(X, y)
    
    # Save classifier
    logger.info("Saving classifier...")
    save_classifier(clf)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
