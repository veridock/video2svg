#!/usr/bin/env python3
"""
CLI wrapper for video2svg converter
"""

import sys
import argparse
from src.converter import VideoToSVGConverter, create_default_config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Convert video to SVG animation')
    parser.add_argument('input', help='Input video file or RTSP stream')
    parser.add_argument('output', help='Output SVG file')
    parser.add_argument('--config', help='Path to config file (YAML)')
    parser.add_argument('--width', type=int, default=640, help='Output width')
    parser.add_argument('--height', type=int, help='Output height (defaults to auto)')
    parser.add_argument('--fps', type=float, default=1, help='Output animation FPS')
    parser.add_argument('--frame-interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--color-mode', choices=['color', 'grayscale', 'monochrome'], 
                        default='color', help='Color mode')
    parser.add_argument('--animation-type', choices=['smil', 'css', 'javascript'], 
                        default='smil', help='Animation type')
    parser.add_argument('--optimize', action='store_true', help='Enable SVG optimization')
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use CLI arguments to build config
        config = create_default_config()
        config['extraction']['frame_interval'] = args.frame_interval
        config['processing']['output_width'] = args.width
        if args.height:
            config['processing']['output_height'] = args.height
        config['processing']['color_mode'] = args.color_mode
        config['svg']['output_fps'] = args.fps
        config['svg']['animation_type'] = args.animation_type
        config['svg']['optimization'] = args.optimize
    
    # Run conversion
    try:
        converter = VideoToSVGConverter(config)
        result = converter.convert(args.input, args.output)
        print(f"✅ Conversion completed: {args.output}")
        print(f"   Size: {result['size'] / 1024:.1f} KB")
        print(f"   Frames: {result['frames_count']}")
        print(f"   Duration: {result['duration']:.1f}s")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
