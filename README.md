# ICU-Lipreading MVP

A Python 3.9 backend project for ICU-lipreading that uses LipNet to extract features from mouth-cropped videos and predicts common ICU phrases.

## Project Overview

This project provides a Flask API that accepts mouth-cropped videos and predicts one of 10 common ICU phrases that a patient might be trying to communicate. It uses a pre-trained LipNet model for feature extraction and a custom-trained classifier for phrase prediction.

## Setup Instructions

### Prerequisites

- Python 3.9
- Git
- Virtual environment tool (e.g., `venv`)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd icu-lipreading-mvp
   ```

2. Run the setup script to initialize the project:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Initialize a Git repository
   - Create and activate a virtual environment
   - Install all dependencies
   - Download and prepare the pre-trained LipNet model

### Data Preparation

Place your training videos in the following structure:
```
data/
  i_need_water/
    video1.mp4
    video2.mp4
    ...
  i_am_in_pain/
    video1.mp4
    video2.mp4
    ...
  ...
```

Each phrase directory should contain at least 5 videos of mouth-cropped footage of someone saying that phrase.

## Training the Classifier

Run the training script to generate the classifier model:

```
python train_classifier.py
```

This will:
1. Extract frames from each video
2. Use the LipNet model to compute embeddings
3. Train a RandomForest classifier on those embeddings
4. Save the classifier as `models/icu_classifier.pkl`

## Running the API

Start the Flask application:

```
python app.py
```

Or with Gunicorn (production):

```
gunicorn --bind 0.0.0.0:5000 app:app
```

## API Endpoints

### Health Check

```
GET /health
```

Returns a 200 status code if the service is running.

### Predict Phrase

```
POST /predict
```

Accepts a form-data upload with a video file under the key 'video'. Returns JSON with the predicted phrase and confidence:

```json
{
  "phrase": "I need water",
  "confidence": 0.85
}
```

## LipNet Model

The project uses a pre-trained LipNet model for lip-reading feature extraction. The setup script automatically downloads and prepares this model for you. The model architecture consists of:

- 3D convolutional layers for spatiotemporal feature extraction
- Bidirectional LSTM layers for sequence modeling
- A dense output layer for classification

We use a modified version of the model that outputs embeddings from the last LSTM layer, which are then used to train our ICU-specific classifier.

## Docker Deployment

Build the Docker image:

```
docker build -t icu-lipreading .
```

Run the container:

```
docker run -p 5000:5000 icu-lipreading
```

## Supported ICU Phrases

1. "I need water"
2. "I am in pain"
3. "I need help"
4. "I can't breathe"
5. "Call the nurse"
6. "I feel sick"
7. "I'm cold"
8. "I'm hot"
9. "Thank you"
10. "I need medication"

## License

[MIT License](LICENSE)
