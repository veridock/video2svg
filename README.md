# Video2SVG

A Python package for converting videos to animated SVG files using various animation techniques.

## Features

- Convert video files to animated SVG
- Support for different animation techniques (SMIL, CSS, JavaScript)
- Frame extraction with customizable settings
- Interactive SVG controls
- SVG optimization

## Installation

```bash
pip install video2svg
```

## Usage

### Command Line

```bash
video2svg input.mp4 output.svg --width 640 --height 480 --fps 10 --animation-type smil
```

Options:
- `--width`: Output width (default: 640)
- `--height`: Output height (default: 480)
- `--fps`: Frames per second (default: 10)
- `--animation-type`: Animation type, one of: smil, css, javascript (default: smil)
- `--loop`: Enable loop animation
- `--no-autoplay`: Disable autoplay
- `--controls`: Add playback controls
- `--optimization`: Optimization level, one of: none, low, medium, high (default: medium)

### Python API

```python
from video2svg import VideoToSVGConverter, SVGAnimation

# Create converter
converter = VideoToSVGConverter()

# Set animation options
animation = SVGAnimation(
    type='smil',  # 'smil', 'css', or 'javascript'
    fps=10,
    loop=True,
    autoplay=True,
    controls=False,
    optimization_level='medium'  # 'none', 'low', 'medium', 'high'
)

# Convert video to SVG
svg_content = converter.convert_video_to_svg(
    'input.mp4',
    width=640,
    height=480,
    animation=animation
)

# Save to file
with open('output.svg', 'w', encoding='utf-8') as f:
    f.write(svg_content)
```

## Development

### Prerequisites

- Python 3.7+
- opencv-python
- numpy
- svgwrite
- PyYAML

### Local Development

```bash
# Clone the repository
git clone https://github.com/veridock/video2svg.git
cd video2svg

# Install development dependencies
pip install -e .

# Run tests
python -m pytest tests/
```

### Building the Package

```bash
python -m build
```

### Publishing to PyPI

```bash
python -m twine upload dist/*
```

## License

MIT License