"""
Video2SVG - Convert videos to animated SVG files.

This package provides tools for converting video files to SVG animations
using various animation techniques (SMIL, CSS, JavaScript).
"""

__version__ = "0.1.0"

from .svg_builder import SVGBuilder, SVGAnimation, SVGFrame, create_thumbnail_svg
from .converter import VideoToSVGConverter
from .frame_extractor import FrameExtractor

__all__ = [
    'SVGBuilder',
    'SVGAnimation', 
    'SVGFrame',
    'VideoToSVGConverter',
    'FrameExtractor',
    'create_thumbnail_svg',
]
