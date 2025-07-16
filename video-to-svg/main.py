import argparse
from src.converter import VideoToSVGConverter
import yaml

def main():
    parser = argparse.ArgumentParser(description='Convert video to SVG')
    parser.add_argument('input', help='Input video file or RTSP stream')
    parser.add_argument('output', help='Output SVG file')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Wczytaj konfigurację
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Konwertuj
    converter = VideoToSVGConverter(config)
    converter.convert(args.input, args.output)
    
    print(f"✅ Konwersja zakończona: {args.output}")

if __name__ == '__main__':
    main()