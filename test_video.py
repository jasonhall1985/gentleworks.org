import requests
import sys
import os
import argparse

def test_video(video_path, demographic=None):
    """Test a video with the ICU lipreading API
    
    Args:
        video_path: Path to the video file
        demographic: Optional demographic category (male_under_50, female_under_50, male_over_50, female_over_50)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Testing video: {video_path}")
    if demographic:
        print(f"Using demographic: {demographic}")
    
    # Prepare the file for upload
    with open(video_path, 'rb') as f:
        files = {'video': f}
        
        # Prepare form data with demographic if provided
        data = {}
        if demographic:
            data['demographic'] = demographic
        
        # Send the request to the API
        response = requests.post('http://127.0.0.1:5000/predict', files=files, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(f"Phrase: {result['phrase']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a video with the ICU lipreading API')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--demographic', choices=['male_under_50', 'female_under_50', 'male_over_50', 'female_over_50'],
                        help='Demographic category of the speaker')
    
    args = parser.parse_args()
    test_video(args.video_path, args.demographic)
