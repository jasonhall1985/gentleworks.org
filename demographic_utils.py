import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Define demographic categories
DEMOGRAPHIC_CATEGORIES = {
    'male_under_50': 0,
    'female_under_50': 1,
    'male_over_50': 2,
    'female_over_50': 3
}

# Map speakers to their demographic categories
SPEAKER_DEMOGRAPHICS = {
    'original': 'male_under_50',  # Original videos (yourself)
    'henry': 'male_over_50',      # Henry's videos
    # Add more speakers as you collect data
}

def get_speaker_from_path(video_path):
    """Extract speaker information from video path."""
    basename = os.path.basename(video_path).lower()
    
    if 'henry' in basename:
        return 'henry'
    else:
        return 'original'

def encode_demographics(demographic_category):
    """Convert demographic category to one-hot encoded feature vector."""
    # One-hot encoding for the 4 demographic categories
    encoding = np.zeros(len(DEMOGRAPHIC_CATEGORIES))
    
    if demographic_category in DEMOGRAPHIC_CATEGORIES:
        encoding[DEMOGRAPHIC_CATEGORIES[demographic_category]] = 1
    else:
        # Default to all zeros if unknown
        logger.warning(f"Unknown demographic category: {demographic_category}")
    
    return encoding

def get_demographic_features(video_path):
    """Get demographic features for a video."""
    speaker = get_speaker_from_path(video_path)
    demographic_category = SPEAKER_DEMOGRAPHICS.get(speaker, None)
    
    if demographic_category is None:
        logger.warning(f"Unknown speaker in video: {video_path}")
        # Default to zeros if speaker is unknown
        return np.zeros(len(DEMOGRAPHIC_CATEGORIES))
    
    return encode_demographics(demographic_category)

def augment_features_with_demographics(features, video_path, demographic_override=None):
    """Augment feature vector with demographic information.
    
    Args:
        features: The original feature vector
        video_path: Path to the video file
        demographic_override: Optional demographic category to override detection
    
    Returns:
        Augmented feature vector with demographic information
    """
    if demographic_override and demographic_override in DEMOGRAPHIC_CATEGORIES:
        # Use the provided demographic category
        demographic_features = encode_demographics(demographic_override)
        logger.info(f"Using provided demographic: {demographic_override}")
    else:
        # Detect demographic from video path
        demographic_features = get_demographic_features(video_path)
    
    # Concatenate the original features with demographic features
    augmented_features = np.concatenate([features, demographic_features])
    
    return augmented_features
