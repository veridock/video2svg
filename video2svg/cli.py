#!/usr/bin/env python3
"""
CLI wrapper dla konwertera video2svg
"""

import sys
import os
import argparse
import yaml
from typing import Dict, Any

# Importy z własnego pakietu
from video2svg.converter import VideoToSVGConverter, create_default_config


def main():
    """Główny punkt wejścia dla CLI"""
    parser = argparse.ArgumentParser(description='Konwertuj wideo na animację SVG')
    parser.add_argument('input', help='Plik wideo lub strumień RTSP')
    parser.add_argument('output', help='Plik wyjściowy SVG')
    parser.add_argument('--config', help='Ścieżka do pliku konfiguracyjnego (YAML)')
    parser.add_argument('--width', type=int, default=640, help='Szerokość wyjściowa')
    parser.add_argument('--height', type=int, help='Wysokość wyjściowa (domyślnie auto)')
    parser.add_argument('--fps', type=float, default=1, help='FPS animacji wyjściowej')
    parser.add_argument('--frame-interval', type=int, default=30, help='Interwał ekstrakcji klatek')
    parser.add_argument('--color-mode', choices=['color', 'grayscale', 'monochrome'], 
                        default='color', help='Tryb kolorów')
    parser.add_argument('--animation-type', choices=['smil', 'css', 'javascript'], 
                        default='smil', help='Typ animacji')
    parser.add_argument('--optimize', action='store_true', help='Włącz optymalizację SVG')
    
    # Nowe opcje dla tekstu, audio i proporcji
    parser.add_argument('--extract-text', action='store_true', help='Wyodrębnij tekst z wideo (OCR)')
    parser.add_argument('--lang', default='pol', help='Język tekstu dla OCR i TTS (kod ISO)')
    parser.add_argument('--generate-audio', action='store_true', help='Generuj plik audio z rozpoznanego tekstu')
    parser.add_argument('--preserve-aspect-ratio', action='store_true', default=True, 
                        help='Zachowaj proporcje obrazu')
    
    args = parser.parse_args()
    
    # Utwórz konfigurację
    config: Dict[str, Any] = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Użyj argumentów CLI do utworzenia konfiguracji
        config = create_default_config()
        config['extraction']['frame_interval'] = args.frame_interval
        config['processing']['output_width'] = args.width
        if args.height:
            config['processing']['output_height'] = args.height
        config['processing']['color_mode'] = args.color_mode
        config['svg']['output_fps'] = args.fps
        config['svg']['animation_type'] = args.animation_type
        config['svg']['optimization'] = args.optimize
        
        # Nowe opcje
        config['ocr'] = {
            'enabled': args.extract_text,
            'language': args.lang,
        }
        config['tts'] = {
            'enabled': args.generate_audio,
            'language': args.lang,
        }
        config['svg']['preserve_aspect_ratio'] = args.preserve_aspect_ratio
    
    # Wykonaj konwersję
    try:
        converter = VideoToSVGConverter(config)
        result = converter.convert(args.input, args.output)
        print(f"Konwersja zakończona: {args.output}")
        print(f"  Rozmiar: {result['size'] / 1024:.1f} KB")
        print(f"  Klatki: {result['frames_count']}")
        print(f"  Czas trwania: {result['duration']:.1f}s")
        
        # Informacje o rozpoznanym tekście i audio
        if 'extracted_text_file' in result and result['extracted_text_file']:
            print(f"  Plik tekstowy: {result['extracted_text_file']}")
            if 'text_snippet' in result and result['text_snippet']:
                print(f"  Przykładowy tekst: {result['text_snippet'][:100]}...")
        
        if 'audio_file' in result and result['audio_file']:
            print(f"  Plik audio: {result['audio_file']}")
            
    except Exception as e:
        print(f"Błąd: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
