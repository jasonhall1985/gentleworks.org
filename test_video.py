import requests
import sys
import os

def test_video(video_path):
    """Test a video with the ICU lipreading API"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Testing video: {video_path}")
    
    # Prepare the file for upload
    with open(video_path, 'rb') as f:
        files = {'video': f}
        
        # Send the request to the API
        response = requests.post('http://127.0.0.1:5000/predict', files=files)
    
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
    if len(sys.argv) < 2:
        print("Usage: python test_video.py /path/to/your/video.mp4")
    else:
        test_video(sys.argv[1])
