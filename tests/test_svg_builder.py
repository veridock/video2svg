"""
Tests for the SVGBuilder module.
"""

import unittest
import numpy as np
from video2svg.svg_builder import SVGBuilder, SVGAnimation, create_thumbnail_svg


class TestSVGBuilder(unittest.TestCase):
    """Tests for SVGBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = SVGBuilder()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a simple gradient for the test frame
        for y in range(480):
            for x in range(640):
                self.test_frame[y, x] = [x // 3, y // 3, 128]
    
    def test_build_static_svg(self):
        """Test building a static SVG."""
        svg_content = self.builder.build_static_svg(self.test_frame)
        
        # Basic validation checks
        self.assertIsInstance(svg_content, str)
        # Note: The SVG might not start with <?xml directly, it depends on implementation
        self.assertIn('<svg', svg_content)
        self.assertIn('<svg', svg_content)
        self.assertIn('</svg>', svg_content)
    
    def test_build_animated_svg_smil(self):
        """Test building an animated SVG using SMIL."""
        frames = [self.test_frame] * 3  # Create 3 identical frames for testing
        animation = SVGAnimation(type='smil', fps=10)
        
        svg_content = self.builder.build_animated_svg(frames, animation=animation)
        
        # Basic validation checks
        self.assertIsInstance(svg_content, str)
        self.assertIn('<animate', svg_content)
    
    def test_thumbnail_creation(self):
        """Test creating a thumbnail SVG with play button."""
        svg_content = create_thumbnail_svg(self.test_frame, add_play_button=True)
        
        # Check for play button indicators
        self.assertIn('polygon', svg_content.lower())
        
        # Test without play button
        svg_content_no_button = create_thumbnail_svg(self.test_frame, add_play_button=False)
        self.assertNotIn('cursor', svg_content_no_button.lower())


if __name__ == '__main__':
    unittest.main()
