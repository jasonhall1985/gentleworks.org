import os
import cv2
import numpy as np
import argparse
import shutil
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the ICU phrases
ICU_PHRASES = [
    "Call the nurse",
    "Help me",
    "I cant breathe",
    "I feel sick",
    "I feel tired"
]

def create_directory_structure():
    """Create the necessary directory structure for training data"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    for phrase in ICU_PHRASES:
        phrase_dir = os.path.join(data_dir, phrase.lower().replace(" ", "_"))
        os.makedirs(phrase_dir, exist_ok=True)
        logger.info(f"Created directory: {phrase_dir}")
    
    return data_dir

def process_frame(frame):
    """Process a pre-cropped frame to prepare for LipNet"""
    # For pre-cropped videos, we just need to resize to the correct dimensions
    # Resize to match LipNet input dimensions (46x140)
    if frame is not None and frame.size > 0:
        processed_frame = cv2.resize(frame, (140, 46))
        return processed_frame
    
    return None

def process_video(video_path, output_dir, phrase_idx):
    """Process a pre-cropped video and save as a new video"""
    # Get phrase directory
    phrase = ICU_PHRASES[phrase_idx]
    phrase_dir = os.path.join(output_dir, phrase.lower().replace(" ", "_"))
    
    # Create a unique filename
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(phrase_dir, f"{name}_processed.mp4")
    
    # Check if already processed
    if os.path.exists(output_path):
        logger.info(f"Video already processed: {output_path}")
        return output_path
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = 140, 46  # Output dimensions for LipNet
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process the pre-cropped frame
        processed_frame = process_frame(frame)
        
        if processed_frame is not None:
            # Write the processed frame
            out.write(processed_frame)
            processed_count += 1
        else:
            # If processing failed, use a blank frame
            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(blank_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    logger.info(f"Processed {processed_count}/{frame_count} frames for {video_path}")
    
    return output_path

def copy_video(video_path, output_dir, phrase_idx):
    """Copy a video to the appropriate phrase directory"""
    # Get phrase directory
    phrase = ICU_PHRASES[phrase_idx]
    phrase_dir = os.path.join(output_dir, phrase.lower().replace(" ", "_"))
    
    # Create a unique filename
    base_name = os.path.basename(video_path)
    output_path = os.path.join(phrase_dir, base_name)
    
    # Check if already copied
    if os.path.exists(output_path):
        logger.info(f"Video already exists: {output_path}")
        return output_path
    
    # Copy the file
    shutil.copy2(video_path, output_path)
    logger.info(f"Copied {video_path} to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for LipNet training')
    parser.add_argument('--input_dir', type=str, help='Directory containing input videos')
    parser.add_argument('--process_type', type=str, default='crop', choices=['crop', 'copy'],
                        help='Type of processing: crop (extract mouth ROI) or copy (just organize files)')
    parser.add_argument('--subfolder_mode', action='store_true', help='If set, assumes videos are organized in subfolders named after phrases')
    args = parser.parse_args()
    
    # Create directory structure
    output_dir = create_directory_structure()
    
    if args.input_dir:
        if args.subfolder_mode or os.path.basename(args.input_dir) in [p.lower().replace(' ', '_') for p in ICU_PHRASES]:
            # Subfolder mode: Process videos from subfolders named after phrases
            logger.info("Using subfolder mode to process videos")
            
            # Check if input_dir itself is a phrase subfolder
            input_dir_name = os.path.basename(args.input_dir)
            if input_dir_name in [p.lower().replace(' ', '_') for p in ICU_PHRASES]:
                # This is a single phrase subfolder
                for idx, phrase in enumerate(ICU_PHRASES):
                    phrase_key = phrase.lower().replace(" ", "_")
                    if phrase_key == input_dir_name:
                        phrase_idx = idx
                        break
                
                # Process all videos in this subfolder
                for file in os.listdir(args.input_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(args.input_dir, file)
                        if args.process_type == 'crop':
                            process_video(video_path, output_dir, phrase_idx)
                        else:  # copy
                            copy_video(video_path, output_dir, phrase_idx)
            else:
                # Process all subfolders
                for phrase_idx, phrase in enumerate(ICU_PHRASES):
                    phrase_key = phrase.lower().replace(" ", "_")
                    phrase_dir = os.path.join(args.input_dir, phrase_key)
                    
                    # Skip if subfolder doesn't exist
                    if not os.path.exists(phrase_dir):
                        # Try with capitalized first letters
                        capitalized_key = '_'.join(word.capitalize() for word in phrase_key.split('_'))
                        phrase_dir = os.path.join(args.input_dir, capitalized_key)
                        if not os.path.exists(phrase_dir):
                            logger.warning(f"Subfolder for '{phrase}' not found. Skipping.")
                            continue
                    
                    logger.info(f"Processing videos for phrase: {phrase} from {phrase_dir}")
                    
                    # Process all videos in the subfolder
                    for file in os.listdir(phrase_dir):
                        if file.endswith(('.mp4', '.avi', '.mov')):
                            video_path = os.path.join(phrase_dir, file)
                            if args.process_type == 'crop':
                                process_video(video_path, output_dir, phrase_idx)
                            else:  # copy
                                copy_video(video_path, output_dir, phrase_idx)
        else:
            # Standard mode: Process all videos in the input directory
            logger.info("Using standard mode to process videos")
            for root, _, files in os.walk(args.input_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(root, file)
                        
                        # Determine phrase index from filename or prompt user
                        phrase_idx = None
                        
                        # Try to determine phrase from filename
                        for idx, phrase in enumerate(ICU_PHRASES):
                            phrase_key = phrase.lower().replace(" ", "_")
                            if phrase_key in file.lower():
                                phrase_idx = idx
                                break
                        
                        # If phrase not determined, prompt user
                        if phrase_idx is None:
                            print(f"\nVideo: {file}")
                            for idx, phrase in enumerate(ICU_PHRASES):
                                print(f"{idx+1}. {phrase}")
                            
                            while phrase_idx is None:
                                try:
                                    selection = int(input("Select phrase number (1-5): "))
                                    if 1 <= selection <= 5:
                                        phrase_idx = selection - 1
                                    else:
                                        print("Invalid selection. Please enter a number between 1 and 5.")
                                except ValueError:
                                    print("Invalid input. Please enter a number.")
                        
                        # Process the video
                        if args.process_type == 'crop':
                            process_video(video_path, output_dir, phrase_idx)
                        else:  # copy
                            copy_video(video_path, output_dir, phrase_idx)
    else:
        logger.info("No input directory specified. Created directory structure only.")
        logger.info("To process videos, run with --input_dir argument.")
        logger.info("Example: python preprocess_videos.py --input_dir /path/to/videos --process_type crop")
        logger.info("For videos organized in subfolders by phrase: python preprocess_videos.py --input_dir /path/to/videos --subfolder_mode --process_type crop")

if __name__ == "__main__":
    main()
