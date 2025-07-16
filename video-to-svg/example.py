# practical_example.py
"""
Praktyczny przykład użycia konwertera z optymalizacjami
dla rzeczywistych scenariuszy
"""

import yaml
from src.converter import VideoToSVGConverter
import logging
from pathlib import Path

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_optimized_config(scenario='default'):
    """
    Tworzy zoptymalizowaną konfigurację dla różnych scenariuszy

    Scenariusze:
    - 'icon': Mała ikona animowana (np. loading spinner)
    - 'presentation': Slajdy prezentacji
    - 'diagram': Animowany diagram
    - 'preview': Podgląd wideo
    """

    configs = {
        'icon': {
            'extraction': {
                'frame_interval': 10,  # Płynna animacja
                'start_time': 0,
                'end_time': 3  # Max 3 sekundy
            },
            'processing': {
                'output_width': 128,  # Mały rozmiar
                'output_height': 128,
                'compression_level': 9,
                'color_mode': 'color'
            },
            'svg': {
                'output_fps': 10,  # Płynność
                'optimization': True,
                'embed_method': 'base64',
                'animation_type': 'smil'
            }
        },

        'presentation': {
            'extraction': {
                'frame_interval': 30,  # 1 fps
                'start_time': 0,
                'end_time': None
            },
            'processing': {
                'output_width': 1024,  # HD
                'output_height': 768,
                'compression_level': 6,
                'color_mode': 'color'
            },
            'svg': {
                'output_fps': 1,
                'optimization': True,
                'embed_method': 'base64',
                'animation_type': 'smil',
                'add_controls': True  # Kontrolki
            }
        },

        'diagram': {
            'extraction': {
                'frame_interval': 60,  # 0.5 fps
                'start_time': 0,
                'end_time': 30  # Max 30 sekund
            },
            'processing': {
                'output_width': 800,
                'output_height': 600,
                'compression_level': 8,
                'color_mode': 'grayscale'  # Diagramy często monochromatyczne
            },
            'svg': {
                'output_fps': 0.5,
                'optimization': True,
                'embed_method': 'base64',
                'animation_type': 'smil'
            },
            'advanced': {
                'vectorize': True,  # Spróbuj wektoryzacji
                'vectorize_threshold': 200
            }
        },

        'preview': {
            'extraction': {
                'frame_interval': 120,  # Co 4 sekundy przy 30fps
                'start_time': 0,
                'end_time': 60  # Max 1 minuta
            },
            'processing': {
                'output_width': 320,
                'output_height': 240,
                'compression_level': 9,
                'color_mode': 'color'
            },
            'svg': {
                'output_fps': 0.25,  # Bardzo wolno
                'optimization': True,
                'embed_method': 'base64',
                'animation_type': 'smil'
            }
        }
    }

    # Domyślna konfiguracja z dodatkowymi optymalizacjami
    base_config = {
        'advanced': {
            'vectorize': False,
            'vectorize_threshold': 128,
            'max_file_size': 50,  # Ostrzeżenie przy 50MB
            'use_cache': True,
            'parallel_processing': True,
            'num_workers': 4
        }
    }

    # Scal konfiguracje
    if scenario in configs:
        config = configs[scenario]
        config['advanced'] = base_config['advanced']
        return config
    else:
        return base_config


def convert_with_size_limit(input_path, output_path, max_size_mb=10):
    """
    Konwertuje wideo z automatycznym dostosowaniem jakości
    aby nie przekroczyć limitu rozmiaru
    """
    logger.info(f"Konwertuję z limitem rozmiaru: {max_size_mb}MB")

    # Zacznij od wysokiej jakości
    quality_levels = [
        {'width': 640, 'interval': 30, 'compression': 6},
        {'width': 480, 'interval': 60, 'compression': 8},
        {'width': 320, 'interval': 120, 'compression': 9},
    ]

    for level in quality_levels:
        config = create_optimized_config('preview')
        config['processing']['output_width'] = level['width']
        config['extraction']['frame_interval'] = level['interval']
        config['processing']['compression_level'] = level['compression']

        converter = VideoToSVGConverter(config)

        try:
            result = converter.convert(input_path, output_path)
            size_mb = result['size'] / (1024 * 1024)

            if size_mb <= max_size_mb:
                logger.info(f"✅ Sukces! Rozmiar: {size_mb:.1f}MB")
                return result
            else:
                logger.warning(f"Plik za duży: {size_mb:.1f}MB, próbuję niższą jakość...")

        except Exception as e:
            logger.error(f"Błąd: {e}")
            continue

    raise Exception(f"Nie udało się utworzyć pliku poniżej {max_size_mb}MB")


def batch_convert_videos(video_list, output_dir, scenario='preview'):
    """
    Konwertuje wiele plików wideo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    config = create_optimized_config(scenario)
    converter = VideoToSVGConverter(config)

    results = []

    for video_path in video_list:
        video_path = Path(video_path)
        output_path = output_dir / f"{video_path.stem}.svg"

        logger.info(f"Konwertuję: {video_path.name}")

        try:
            result = converter.convert(str(video_path), str(output_path))
            results.append({
                'input': video_path.name,
                'output': output_path.name,
                'size_mb': result['size'] / (1024 * 1024),
                'duration': result['duration']
            })
        except Exception as e:
            logger.error(f"Błąd przy {video_path.name}: {e}")
            results.append({
                'input': video_path.name,
                'error': str(e)
            })

    return results


def create_animated_logo(video_path, output_path, duration=3):
    """
    Tworzy animowane logo z pierwszych sekund wideo
    """
    config = create_optimized_config('icon')
    config['extraction']['end_time'] = duration

    # Dodatkowe optymalizacje dla logo
    config['processing']['output_width'] = 200
    config['processing']['output_height'] = 200
    config['svg']['output_fps'] = 15  # Płynna animacja

    converter = VideoToSVGConverter(config)

    def progress_callback(percent, message):
        print(f"[{percent:3d}%] {message}")

    result = converter.convert(video_path, output_path, progress_callback)

    logger.info(f"Logo utworzone: {result['size'] / 1024:.1f}KB")
    return result


def extract_keyframes_only(video_path, output_path, num_frames=10):
    """
    Ekstraktuje tylko kluczowe klatki (np. zmiany scen)
    """
    # Specjalna konfiguracja dla klatek kluczowych
    config = create_optimized_config('preview')

    # Oblicz interwał na podstawie długości wideo
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    interval = max(1, total_frames // num_frames)
    config['extraction']['frame_interval'] = interval

    converter = VideoToSVGConverter(config)
    result = converter.convert(video_path, output_path)

    logger.info(f"Wyekstraktowano {result['frames_count']} klatek kluczowych")
    return result


# Przykłady użycia
if __name__ == '__main__':

    # 1. Konwersja z limitem rozmiaru
    print("\n=== Test 1: Konwersja z limitem rozmiaru ===")
    try:
        result = convert_with_size_limit(
            'samples/test_video.mp4',
            'output/test_limited.svg',
            max_size_mb=5
        )
        print(f"Utworzono: {result['output_file']}")
    except Exception as e:
        print(f"Błąd: {e}")

    # 2. Tworzenie animowanego logo
    print("\n=== Test 2: Animowane logo ===")
    try:
        create_animated_logo(
            'samples/intro.mp4',
            'output/animated_logo.svg',
            duration=2
        )
    except Exception as e:
        print(f"Błąd: {e}")

    # 3. Batch processing
    print("\n=== Test 3: Przetwarzanie wielu plików ===")
    videos = ['samples/video1.mp4', 'samples/video2.mp4']
    results = batch_convert_videos(videos, 'output/batch', scenario='preview')

    for result in results:
        if 'error' in result:
            print(f"❌ {result['input']}: {result['error']}")
        else:
            print(f"✅ {result['input']} -> {result['output']} "
                  f"({result['size_mb']:.1f}MB w {result['duration']:.1f}s)")

    # 4. Klatki kluczowe
    print("\n=== Test 4: Tylko klatki kluczowe ===")
    try:
        extract_keyframes_only(
            'samples/presentation.mp4',
            'output/keyframes.svg',
            num_frames=5
        )
    except Exception as e:
        print(f"Błąd: {e}")

    # 5. Przykład konfiguracji YAML
    print("\n=== Zapisywanie konfiguracji ===")
    for scenario in ['icon', 'presentation', 'diagram', 'preview']:
        config = create_optimized_config(scenario)
        with open(f'configs/{scenario}_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Zapisano: configs/{scenario}_config.yaml")