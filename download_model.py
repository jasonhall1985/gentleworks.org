import os
import sys
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def create_lipnet_model():
    """Create the LipNet model architecture"""
    model = Sequential()

    # 3D convolutional layers
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    # Output layer
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    
    return model

def create_embedding_model(lipnet_model):
    """Create a model that outputs embeddings from the last LSTM layer"""
    # Create a new model that outputs the activations from the last LSTM layer
    embedding_model = Model(
        inputs=lipnet_model.input,
        outputs=lipnet_model.layers[-3].output  # Output from the last LSTM layer before dropout
    )
    return embedding_model

def download_pretrained_weights():
    """Download pre-trained weights for LipNet"""
    # URL for pre-trained weights
    # Note: This is a placeholder URL. In a real scenario, you would use the actual URL to the weights file
    weights_url = "https://github.com/BMehar98/Lip-Reading-Web-Application/raw/main/checkpoint"
    weights_path = os.path.join('models', 'lipnet_weights')
    
    try:
        logger.info(f"Downloading pre-trained weights from {weights_url}")
        response = requests.get(weights_url, stream=True)
        response.raise_for_status()
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Weights downloaded successfully to {weights_path}")
        return weights_path
    except Exception as e:
        logger.error(f"Error downloading weights: {e}")
        return None

def main():
    """Main function to download and prepare the LipNet model"""
    try:
        # Create the LipNet model
        logger.info("Creating LipNet model architecture")
        lipnet_model = create_lipnet_model()
        
        # Download pre-trained weights
        weights_path = download_pretrained_weights()
        if weights_path is None:
            logger.error("Failed to download weights. Exiting.")
            return
        
        # Load the weights
        try:
            logger.info("Loading pre-trained weights")
            lipnet_model.load_weights(weights_path)
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            logger.info("Proceeding with randomly initialized weights")
        
        # Create the embedding model
        logger.info("Creating embedding model")
        embedding_model = create_embedding_model(lipnet_model)
        
        # Save the models
        lipnet_model_path = os.path.join('models', 'lipnet_model.h5')
        embedding_model_path = os.path.join('models', 'lipnet_embedding_model.h5')
        
        logger.info(f"Saving LipNet model to {lipnet_model_path}")
        lipnet_model.save(lipnet_model_path)
        
        logger.info(f"Saving embedding model to {embedding_model_path}")
        embedding_model.save(embedding_model_path)
        
        logger.info("Models saved successfully!")
        
        # Print model summary
        logger.info("LipNet Model Summary:")
        lipnet_model.summary(print_fn=logger.info)
        
        logger.info("\nEmbedding Model Summary:")
        embedding_model.summary(print_fn=logger.info)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
