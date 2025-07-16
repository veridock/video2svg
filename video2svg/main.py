#!/usr/bin/env python
"""
Video2SVG - Command line interface for converting videos to SVG.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .converter import VideoToSVGConverter
from .svg_builder import SVGAnimation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert video to SVG animation')
    
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output SVG file path')
    parser.add_argument('--width', type=int, default=640, help='Output width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Output height (default: 480)')
    parser.add_argument('--fps', type=float, default=10, help='Frames per second (default: 10)')
    parser.add_argument('--animation-type', choices=['smil', 'css', 'javascript'], 
                        default='smil', help='Animation type (default: smil)')
    parser.add_argument('--loop', action='store_true', help='Enable loop animation')
    parser.add_argument('--no-autoplay', action='store_false', dest='autoplay',
                        help='Disable autoplay')
    parser.add_argument('--controls', action='store_true', help='Add playback controls')
    parser.add_argument('--optimization', choices=['none', 'low', 'medium', 'high'],
                        default='medium', help='Optimization level (default: medium)')
    parser.add_argument('--config', help='Path to YAML configuration file')
    
    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file if provided."""
    config = {}
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create SVG animation configuration
    animation = SVGAnimation(
        type=args.animation_type,
        fps=args.fps,
        loop=args.loop,
        autoplay=args.autoplay,
        controls=args.controls,
        optimization_level=args.optimization
    )
    
    # Create converter
    converter = VideoToSVGConverter(config)
    
    try:
        # Convert video to SVG
        svg_content = converter.convert_video_to_svg(
            args.input,
            width=args.width,
            height=args.height,
            animation=animation
        )
        
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        logger.info(f"Successfully converted {args.input} to {args.output}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
