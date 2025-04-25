import yaml
from video_detector import AdvancedVideoDetector
import time

def test_video_processing():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = AdvancedVideoDetector(config)
    
    # Test video path
    video_path = "uploads/Dashcam sample video.mp4"
    
    # Test prompts (starting from truck)
    prompts = [
        "truck",
        "motorcycle"
    ]
    
    # Test modes
    modes = [
        "full_scan",
        "frame_skip",
        "motion_filter",
        "track_then_match"
    ]
    
    total_tests = len(prompts) * len(modes)
    current_test = 9  # Starting from test 9 (after completing car and person)
    
    print(f"\nContinuing with remaining {total_tests} test cases...")
    print("=" * 50)
    
    # Run tests
    for prompt in prompts:
        print(f"\nTesting prompt: '{prompt}'")
        for mode in modes:
            print(f"\nTest {current_test}/16: Mode: {mode}")
            print("-" * 30)
            start_time = time.time()
            
            try:
                results = detector.process_video(video_path, prompt, mode)
                duration = time.time() - start_time
                
                print(f"Found {len(results)} matches")
                print(f"Processing time: {duration:.2f} seconds")
                
                if results:
                    print("Top matches:")
                    for i, result in enumerate(results[:3]):
                        print(f"  {i+1}. Frame {result['frame']} at {result['timestamp']:.2f}s")
                        print(f"     Similarity: {result['similarity']:.2f}")
                        if 'class_name' in result:
                            print(f"     Class: {result['class_name']}")
                else:
                    print("No matches found")
                
            except Exception as e:
                print(f"Error: {str(e)}")
            
            current_test += 1
            print("-" * 30)

if __name__ == "__main__":
    test_video_processing() 