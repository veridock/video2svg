# src/converter.py
"""
Video to SVG Converter - Główny moduł konwersji
"""

import cv2
import base64
import numpy as np
from pathlib import Path
import svgwrite
from typing import List, Optional, Callable, Dict, Any
import logging
from datetime import datetime
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoToSVGError(Exception):
    """Bazowy wyjątek dla konwertera"""
    pass


class VideoNotFoundError(VideoToSVGError):
    """Gdy plik wideo nie został znaleziony"""
    pass


class ConversionError(VideoToSVGError):
    """Gdy konwersja się nie powiodła"""
    pass


class ConfigurationError(VideoToSVGError):
    """Gdy konfiguracja jest nieprawidłowa"""
    pass


class VideoToSVGConverter:
    """
    Główny konwerter video do SVG

    Przykład użycia:
        converter = VideoToSVGConverter(config)
        converter.convert('input.mp4', 'output.svg')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicjalizacja konwertera

        Args:
            config: Słownik z konfiguracją
        """
        self.config = self._validate_config(config)
        self.frames = []
        self.temp_dir = None
        self.rtsp_timeout = 10
        self.rtsp_retries = 3

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Walidacja i uzupełnienie domyślnych wartości konfiguracji"""
        defaults = {
            'extraction': {
                'frame_interval': 30,
                'start_time': 0,
                'end_time': None
            },
            'processing': {
                'output_width': 640,
                'output_height': None,
                'compression_level': 8,
                'color_mode': 'color'
            },
            'svg': {
                'output_fps': 1,
                'optimization': True,
                'embed_method': 'base64',
                'animation_type': 'smil'
            },
            'advanced': {
                'vectorize': False,
                'vectorize_threshold': 128,
                'max_file_size': 100,
                'use_cache': True,
                'parallel_processing': True,
                'num_workers': 4
            }
        }

        # Scalanie z domyślnymi wartościami
        for section, values in defaults.items():
            if section not in config:
                config[section] = values
            else:
                for key, value in values.items():
                    if key not in config[section]:
                        config[section][key] = value

        return config

    def convert(self, input_path: str, output_path: str,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Konwertuje video do SVG

        Args:
            input_path: Ścieżka do pliku wideo lub URL RTSP
            output_path: Ścieżka do pliku wyjściowego SVG
            progress_callback: Funkcja callback dla postępu (opcjonalna)

        Returns:
            dict: Informacje o konwersji

        Raises:
            VideoNotFoundError: Gdy plik nie istnieje
            ConversionError: Gdy konwersja się nie powiedzie
        """
        start_time = datetime.now()
        logger.info(f"Rozpoczynam konwersję: {input_path} -> {output_path}")

        try:
            # Sprawdź czy plik istnieje (jeśli nie jest to RTSP)
            if not input_path.startswith('rtsp://'):
                if not Path(input_path).exists():
                    raise VideoNotFoundError(f"Plik nie istnieje: {input_path}")

            # 1. Ekstrakcja klatek
            self._update_progress(progress_callback, 10, "Ekstraktuję klatki...")
            frames = self.extract_frames(input_path)

            if not frames:
                raise ConversionError("Nie udało się wyekstraktować żadnych klatek")

            # 2. Przetwarzanie klatek (opcjonalne)
            if self.config['advanced']['vectorize']:
                self._update_progress(progress_callback, 40, "Wektoryzuję klatki...")
                frames = self.vectorize_frames(frames)

            # 3. Budowanie SVG
            self._update_progress(progress_callback, 70, "Generuję SVG...")
            svg_content = self.build_svg(frames)

            # 4. Optymalizacja (opcjonalna)
            if self.config['svg']['optimization']:
                self._update_progress(progress_callback, 85, "Optymalizuję SVG...")
                svg_content = self._optimize_svg(svg_content)

            # 5. Zapis
            self._update_progress(progress_callback, 95, "Zapisuję plik...")
            output_size = self._save_svg(svg_content, output_path)

            # Sprawdź rozmiar
            max_size_mb = self.config['advanced']['max_file_size']
            if output_size > max_size_mb * 1024 * 1024:
                logger.warning(f"Plik SVG jest duży: {output_size / 1024 / 1024:.1f} MB")

            # Sukces
            self._update_progress(progress_callback, 100, "Konwersja zakończona!")

            duration = (datetime.now() - start_time).total_seconds()

            return {
                'status': 'success',
                'output_file': output_path,
                'size': output_size,
                'duration': duration,
                'frames_count': len(frames),
                'fps': self.config['svg']['output_fps']
            }

        except Exception as e:
            logger.error(f"Błąd podczas konwersji: {e}")
            raise ConversionError(f"Konwersja nie powiodła się: {e}")
        finally:
            # Czyszczenie
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)

    def extract_frames(self, video_path: str) -> List[str]:
        """
        Ekstraktuje klatki z wideo

        Args:
            video_path: Ścieżka do wideo lub URL RTSP

        Returns:
            Lista klatek w formacie base64
        """
        frames = []
        frame_interval = self.config['extraction']['frame_interval']
        start_time = self.config['extraction']['start_time']
        end_time = self.config['extraction']['end_time']

        # Otwórz wideo
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            if video_path.startswith('rtsp://'):
                # Próba ponownego połączenia dla RTSP
                for retry in range(self.rtsp_retries):
                    logger.info(f"Próba połączenia RTSP {retry + 1}/{self.rtsp_retries}")
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        break
                    import time
                    time.sleep(2)

                if not cap.isOpened():
                    raise VideoNotFoundError(f"Nie można połączyć się z: {video_path}")
            else:
                raise VideoNotFoundError(f"Nie można otworzyć pliku: {video_path}")

        # Pobierz informacje o wideo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ustaw pozycję startową
        if start_time > 0:
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Oblicz końcową klatkę
        end_frame = total_frames
        if end_time:
            end_frame = min(int(end_time * fps), total_frames)

        # Ekstrakcja klatek
        frame_count = 0
        extracted_count = 0

        # Użyj przetwarzania równoległego jeśli włączone
        if self.config['advanced']['parallel_processing']:
            frames = self._extract_frames_parallel(cap, frame_interval, end_frame)
        else:
            while cap.isOpened() and frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    processed_frame = self.process_frame(frame)
                    frames.append(processed_frame)
                    extracted_count += 1
                    logger.debug(f"Wyekstraktowano klatkę {extracted_count}")

                frame_count += 1

        cap.release()

        logger.info(f"Wyekstraktowano {len(frames)} klatek")
        return frames

    def _extract_frames_parallel(self, cap, frame_interval: int, end_frame: int) -> List[str]:
        """Równoległe przetwarzanie klatek"""
        frames_data = []
        frame_count = 0

        # Zbierz klatki do przetworzenia
        while cap.isOpened() and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames_data.append((frame_count, frame.copy()))

            frame_count += 1

        # Przetwarzaj równolegle
        frames = [None] * len(frames_data)
        with ThreadPoolExecutor(max_workers=self.config['advanced']['num_workers']) as executor:
            future_to_index = {
                executor.submit(self.process_frame, frame): i
                for i, (_, frame) in enumerate(frames_data)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                frames[index] = future.result()

        return frames

    def process_frame(self, frame: np.ndarray) -> str:
        """
        Przetwarza pojedynczą klatkę do base64

        Args:
            frame: Klatka jako numpy array

        Returns:
            Klatka w formacie base64
        """
        # Zmień rozmiar
        output_width = self.config['processing']['output_width']
        output_height = self.config['processing']['output_height']

        height, width = frame.shape[:2]

        if output_height is None:
            # Zachowaj proporcje
            output_height = int(height * output_width / width)

        if (width, height) != (output_width, output_height):
            frame = cv2.resize(frame, (output_width, output_height))

        # Zmień tryb kolorów jeśli potrzeba
        color_mode = self.config['processing']['color_mode']
        if color_mode == 'grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif color_mode == 'monochrome':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Kompresja PNG
        compression_level = self.config['processing']['compression_level']
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

        # Konwersja do base64
        _, buffer = cv2.imencode('.png', frame, encode_params)
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        return base64_frame

    def vectorize_frames(self, frames: List[str]) -> List[str]:
        """
        Wektoryzacja klatek (placeholder - wymaga implementacji z Potrace)

        Args:
            frames: Lista klatek w base64

        Returns:
            Lista zwektoryzowanych klatek
        """
        logger.warning("Wektoryzacja nie jest jeszcze zaimplementowana")
        return frames

    def build_svg(self, frames: List[str]) -> str:
        """
        Buduje animowany SVG z klatek

        Args:
            frames: Lista klatek w formacie base64

        Returns:
            Zawartość pliku SVG
        """
        width = self.config['processing']['output_width']
        height = self.config['processing']['output_height'] or 480
        fps = self.config['svg']['output_fps']
        duration = len(frames) / fps

        # Tworzenie SVG
        dwg = svgwrite.Drawing(size=(width, height))

        # Dodaj definicje dla optymalizacji
        if self.config['svg']['optimization']:
            defs = dwg.defs
            dwg.add(defs)

        # Dodaj każdą klatkę
        for i, frame_data in enumerate(frames):
            # Czas pokazywania klatki
            begin_time = i / fps
            end_time = (i + 1) / fps

            # Dodaj obraz
            image = dwg.image(
                f'data:image/png;base64,{frame_data}',
                insert=(0, 0),
                size=(width, height)
            )

            # Animacja typu SMIL
            if self.config['svg']['animation_type'] == 'smil':
                # Animacja opacity
                image.add(dwg.animate(
                    'opacity',
                    dur=f'{duration}s',
                    values='0;0;1;1;0;0',
                    keyTimes=f'0;{begin_time / duration};{begin_time / duration};'
                             f'{end_time / duration};{end_time / duration};1',
                    repeatCount='indefinite'
                ))

            dwg.add(image)

        # Dodaj kontrolki jeśli potrzebne
        if self.config.get('add_controls', False):
            self._add_controls(dwg, duration)

        return dwg.tostring()

    def _add_controls(self, dwg, duration: float):
        """Dodaje interaktywne kontrolki do SVG"""
        # Placeholder - można rozbudować o przyciski play/pause
        pass

    def _optimize_svg(self, svg_content: str) -> str:
        """
        Optymalizuje zawartość SVG

        Args:
            svg_content: Zawartość SVG

        Returns:
            Zoptymalizowana zawartość
        """
        # Podstawowa optymalizacja - usunięcie zbędnych spacji
        import re

        # Usuń zbędne spacje i nowe linie
        svg_content = re.sub(r'>\s+<', '><', svg_content)
        svg_content = re.sub(r'\s+', ' ', svg_content)

        # Usuń komentarze
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)

        return svg_content.strip()

    def _save_svg(self, svg_content: str, output_path: str) -> int:
        """
        Zapisuje SVG do pliku

        Args:
            svg_content: Zawartość SVG
            output_path: Ścieżka wyjściowa

        Returns:
            Rozmiar pliku w bajtach
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        return output_path.stat().st_size

    def _update_progress(self, callback: Optional[Callable],
                         percent: int, message: str):
        """Aktualizuje postęp jeśli podano callback"""
        if callback:
            callback(percent, message)
        logger.info(f"[{percent}%] {message}")


# Funkcje pomocnicze
def load_config(config_path: str) -> Dict[str, Any]:
    """Wczytuje konfigurację z pliku YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict[str, Any]:
    """Tworzy domyślną konfigurację"""
    return {
        'extraction': {
            'frame_interval': 30,
            'start_time': 0,
            'end_time': None
        },
        'processing': {
            'output_width': 640,
            'output_height': None,
            'compression_level': 8,
            'color_mode': 'color'
        },
        'svg': {
            'output_fps': 1,
            'optimization': True,
            'embed_method': 'base64',
            'animation_type': 'smil'
        },
        'advanced': {
            'vectorize': False,
            'vectorize_threshold': 128,
            'max_file_size': 100,
            'use_cache': True,
            'parallel_processing': True,
            'num_workers': 4
        }
    }


if __name__ == '__main__':
    # Test podstawowy
    config = create_default_config()
    converter = VideoToSVGConverter(config)

    try:
        result = converter.convert('test_video.mp4', 'output.svg')
        print(f"Konwersja zakończona: {result}")
    except Exception as e:
        print(f"Błąd: {e}")