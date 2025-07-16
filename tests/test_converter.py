"""
Tests for the VideoToSVGConverter module.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from video2svg.converter import VideoToSVGConverter
from video2svg.svg_builder import SVGAnimation


class TestVideoToSVGConverter(unittest.TestCase):
    """Tests for VideoToSVGConverter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = VideoToSVGConverter(config={})
        
        # Create test frames
        self.test_frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
        ]
    
    @patch('video2svg.converter.FrameExtractor')
    @patch('video2svg.converter.SVGBuilder')
    def test_convert_video_to_svg(self, mock_svg_builder_class, mock_frame_extractor_class):
        """Test converting a video to SVG."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = self.test_frames
        mock_frame_extractor_class.return_value = mock_extractor
        
        mock_builder = MagicMock()
        mock_builder.build_animated_svg.return_value = "<svg>Test SVG</svg>"
        mock_svg_builder_class.return_value = mock_builder
        
        # Test conversion
        result = self.converter.convert_video_to_svg(
            "dummy_video.mp4", 
            width=320, 
            height=240,
            animation=SVGAnimation(type='smil', fps=15)
        )
        
        # Verify results
        self.assertEqual(result, "<svg>Test SVG</svg>")
        
        # Check that methods were called with correct parameters
        mock_extractor.extract_frames.assert_called_once()
        mock_builder.build_animated_svg.assert_called_once()
        
        # Check width and height were passed correctly
        _, kwargs = mock_builder.build_animated_svg.call_args
        self.assertEqual(kwargs['width'], 320)
        self.assertEqual(kwargs['height'], 240)
        self.assertIsInstance(kwargs['animation'], SVGAnimation)
        self.assertEqual(kwargs['animation'].fps, 15)
    
    @patch('video2svg.converter.FrameExtractor')
    @patch('video2svg.converter.SVGBuilder')
    def test_convert_with_custom_config(self, mock_svg_builder_class, mock_frame_extractor_class):
        """Test converting with custom configuration."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = self.test_frames
        mock_frame_extractor_class.return_value = mock_extractor
        
        mock_builder = MagicMock()
        mock_builder.build_animated_svg.return_value = "<svg>Test SVG</svg>"
        mock_svg_builder_class.return_value = mock_builder
        
        # Create converter with custom config
        custom_config = {
            'max_frames': 10,
            'optimization': 'high'
        }
        custom_converter = VideoToSVGConverter(custom_config)
        
        # Test conversion
        result = custom_converter.convert_video_to_svg("dummy_video.mp4")
        
        # Verify the configuration was used
        mock_extractor.extract_frames.assert_called_with(
            "dummy_video.mp4", max_frames=10
        )


if __name__ == '__main__':
    unittest.main()
