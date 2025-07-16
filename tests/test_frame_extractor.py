"""
Tests for the FrameExtractor module.
"""

import unittest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
from video2svg.frame_extractor import FrameExtractor


class TestFrameExtractor(unittest.TestCase):
    """Tests for FrameExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FrameExtractor()
        
        # Create a temporary test video file
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('video2svg.frame_extractor.cv2')
    @patch('video2svg.frame_extractor.Path')
    def test_extract_frames(self, mock_path, mock_cv2):
        """Test extracting frames from a video."""
        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Mock video capture properties
        mock_video = MagicMock()
        mock_video.get.side_effect = lambda prop: {
            7: 30.0,  # FPS
            3: 640.0,  # Width
            4: 480.0,  # Height
        }.get(prop, 0)
        
        # Mock reading frames
        mock_video.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8) * 128),
            (False, None)
        ]
        
        mock_cv2.VideoCapture.return_value = mock_video
        mock_video.isOpened.return_value = True
        
        # Test frame extraction
        frames = self.extractor.extract_frames('dummy_path.mp4', max_frames=2)
        
        # Check results
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].shape, (480, 640, 3))
        self.assertEqual(frames[1].shape, (480, 640, 3))
        
        # Check first frame is all zeros
        self.assertTrue(np.all(frames[0] == 0))
        
        # Check second frame is all 128s
        self.assertTrue(np.all(frames[1] == 128))
    
    @patch('video2svg.frame_extractor.cv2')
    @patch('video2svg.frame_extractor.Path')
    def test_extract_frames_with_interval(self, mock_path, mock_cv2):
        """Test extracting frames with specific interval."""
        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Mock video capture properties
        mock_video = MagicMock()
        mock_video.get.return_value = 30.0  # FPS
        mock_video.isOpened.return_value = True
        
        # Mock reading frames
        mock_video.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8) * 128),
            (False, None)
        ]
        
        mock_cv2.VideoCapture.return_value = mock_video
        
        # Test frame extraction with interval - use correct method name
        frames = self.extractor.extract_frames(
            'dummy_path.mp4', frame_interval=30, max_frames=2
        )
        
        # Check results
        self.assertEqual(len(frames), 2)
        
        # Verify that we set the position correctly
        mock_video.set.assert_called_with(1, 30.0)  # Position to frame 30 (1 second at 30fps)


if __name__ == '__main__':
    unittest.main()
