import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten, Input
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def create_lipnet_model():
    """Create the LipNet model architecture"""
    # Use the functional API for more flexibility
    inputs = Input(shape=(75, 46, 140, 1))
    
    # 3D convolutional layers
    x = Conv3D(128, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(75, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = TimeDistributed(Flatten())(x)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(.5)(x)
    
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(.5)(x)
    
    # Output layer
    outputs = Dense(41, kernel_initializer='he_normal', activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_embedding_model(lipnet_model):
    """Create a model that outputs embeddings from the last LSTM layer"""
    # Create a new model that outputs the activations from the last LSTM layer
    embedding_model = Model(
        inputs=lipnet_model.input,
        outputs=lipnet_model.layers[-3].output  # Output from the last LSTM layer before dropout
    )
    return embedding_model

def compile_model(model):
    """Compile the model with appropriate loss and optimizer"""
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    """Main function to prepare the LipNet model"""
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        
        # Create the LipNet model
        logger.info("Creating LipNet model architecture")
        lipnet_model = create_lipnet_model()
        
        # Compile the model
        compile_model(lipnet_model)
        
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
        
        logger.info("Models created and saved successfully!")
        
        # Print model summary
        logger.info("LipNet Model Summary:")
        lipnet_model.summary(print_fn=logger.info)
        
        logger.info("\nEmbedding Model Summary:")
        embedding_model.summary(print_fn=logger.info)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
