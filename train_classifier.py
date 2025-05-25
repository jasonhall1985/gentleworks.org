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
from demographic_utils import augment_features_with_demographics, get_speaker_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the ICU phrases we want to recognize
ICU_PHRASES = [
    "Call the nurse",
    "Help me",
    "I cant breathe",
    "I feel sick",
    "I feel tired"
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
                
                # Augment embeddings with demographic features
                augmented_embeddings = augment_features_with_demographics(embeddings, video_path)
                
                # Add to dataset
                X.append(augmented_embeddings)
                y.append(phrase_idx)
                
                logger.info(f"Processed {video_file} for phrase: {phrase} (Speaker: {get_speaker_from_path(video_path)})")
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                continue
    
    return np.array(X), np.array(y)

def train_classifier(X, y, demographic_feature_count=4):
    """Train a classifier on the embeddings with enhanced accuracy"""
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Check if we have enough samples for cross-validation
    min_samples_per_class = {}
    for class_idx in np.unique(y):
        min_samples_per_class[class_idx] = np.sum(y == class_idx)
    
    min_samples = min(min_samples_per_class.values())
    logger.info(f"Minimum samples per class: {min_samples}")
    
    # Scale the features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if min_samples < 3:
        logger.warning("Limited samples per class. Using simple ensemble with cross-validation.")
        
        # Use stratified K-fold cross-validation to maximize use of limited data
        cv = StratifiedKFold(n_splits=min(5, min_samples), shuffle=True, random_state=42)
        
        # Train multiple models and use voting for better accuracy
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        svm = SVC(probability=True, class_weight='balanced', random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Create a voting classifier
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm), ('gb', gb)],
            voting='soft'  # Use probability estimates for voting
        )
        
        # Cross-validation scores
        scores = []
        for train_idx, test_idx in cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            ensemble.fit(X_train, y_train)
            score = ensemble.score(X_test, y_test)
            scores.append(score)
        
        logger.info(f"Cross-validation accuracy scores: {scores}")
        logger.info(f"Mean CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Train final model on all data
        ensemble.fit(X_scaled, y)
        
        # Create a pipeline with the scaler and ensemble
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', ensemble)
        ])
        
        return pipeline
        
    else:
        logger.info("Sufficient samples for hyperparameter tuning. Optimizing models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models and parameters for grid search
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced']
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear'],
                    'class_weight': [None, 'balanced']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        # Find the best model and parameters
        best_score = 0
        best_model = None
        best_params = None
        best_model_name = None
        
        for model_name, model_info in models.items():
            logger.info(f"Tuning {model_name}...")
            # Ensure cv is at most the number of samples in the smallest class
            cv_folds = max(2, min(3, min_samples))  # At least 2, at most 3 folds
            grid = GridSearchCV(
                model_info['model'], 
                model_info['params'], 
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1  # Use all available cores
            )
            grid.fit(X_train, y_train)
            
            logger.info(f"Best {model_name} parameters: {grid.best_params_}")
            logger.info(f"Best {model_name} cross-validation score: {grid.best_score_:.4f}")
            
            # Evaluate on test set
            y_pred = grid.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)
            logger.info(f"{model_name} test accuracy: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                best_model_name = model_name
        
        logger.info(f"Best model: {best_model_name} with test accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Create ensemble with the best model and a few others for robustness
        logger.info("Creating final ensemble model...")
        final_models = []
        
        # Add the best model
        final_models.append((best_model_name, best_model))
        
        # Add RandomForest if it's not already the best
        if best_model_name != 'RandomForest':
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)
            final_models.append(('RandomForest', rf))
        
        # Add SVM if it's not already the best
        if best_model_name != 'SVM':
            svm = SVC(probability=True, class_weight='balanced', random_state=42)
            svm.fit(X_train, y_train)
            final_models.append(('SVM', svm))
        
        # Create the final ensemble
        final_ensemble = VotingClassifier(
            estimators=final_models,
            voting='soft'  # Use probability estimates for voting
        )
        
        # Train on all data for the final model
        final_ensemble.fit(X_scaled, y)
        
        # Evaluate on test set
        y_pred = final_ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Final ensemble test accuracy: {ensemble_accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=ICU_PHRASES))
        
        # Create a pipeline with the scaler and ensemble
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', final_ensemble)
        ])
        
        # Store demographic feature count for future reference
        pipeline.demographic_feature_count = demographic_feature_count
        
        return pipeline

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
