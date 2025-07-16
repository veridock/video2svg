# src/frame_extractor.py
"""
Moduł odpowiedzialny za ekstrakcję klatek z plików wideo i strumieni RTSP.
Obsługuje różne formaty wideo i zapewnia wydajną ekstrakcję z konfigurowalnymi parametrami.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator, Dict, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import threading
import queue
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Konfiguracja loggera
logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Informacje o wyekstraktowanej klatce"""
    index: int
    timestamp: float
    data: np.ndarray
    is_keyframe: bool = False
    scene_change_score: float = 0.0


@dataclass
class VideoInfo:
    """Informacje o pliku wideo"""
    path: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    codec: str

    @property
    def duration_str(self) -> str:
        """Zwraca czas trwania jako string (HH:MM:SS)"""
        return str(timedelta(seconds=int(self.duration)))


class FrameExtractorError(Exception):
    """Bazowy wyjątek dla ekstraktora klatek"""
    pass


class VideoNotFoundError(FrameExtractorError):
    """Gdy nie można otworzyć pliku wideo"""
    pass


class RTSPConnectionError(FrameExtractorError):
    """Błąd połączenia RTSP"""
    pass


class FrameExtractor:
    """
    Główna klasa do ekstrakcji klatek z wideo.

    Przykład użycia:
        extractor = FrameExtractor()
        frames = extractor.extract_frames('video.mp4', interval=30)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicjalizacja ekstraktora.

        Args:
            config: Opcjonalna konfiguracja
        """
        self.config = config or {}
        self.rtsp_timeout = self.config.get('rtsp_timeout', 10)
        self.rtsp_retries = self.config.get('rtsp_retries', 3)
        self.buffer_size = self.config.get('buffer_size', 100)
        self.use_threading = self.config.get('use_threading', True)
        self.detect_keyframes = self.config.get('detect_keyframes', False)

    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Pobiera informacje o pliku wideo.

        Args:
            video_path: Ścieżka do pliku lub URL RTSP

        Returns:
            VideoInfo: Informacje o wideo

        Raises:
            VideoNotFoundError: Gdy nie można otworzyć pliku
        """
        cap = self._open_video(video_path)

        try:
            info = VideoInfo(
                path=video_path,
                fps=cap.get(cv2.CAP_PROP_FPS),
                total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                duration=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=self._get_codec(cap)
            )

            # Dla strumieni RTSP niektóre wartości mogą być niedostępne
            if video_path.startswith('rtsp://'):
                if info.total_frames <= 0:
                    info.total_frames = -1  # Nieznana liczba klatek
                if info.duration <= 0:
                    info.duration = -1  # Nieznany czas trwania

            return info

        finally:
            cap.release()

    def extract_frames(self,
                       video_path: str,
                       interval: int = 30,
                       start_time: float = 0,
                       end_time: Optional[float] = None,
                       max_frames: Optional[int] = None,
                       as_generator: bool = False) -> Union[List[FrameInfo], Generator[FrameInfo, None, None]]:
        """
        Ekstraktuje klatki z wideo.

        Args:
            video_path: Ścieżka do pliku lub URL RTSP
            interval: Co która klatka ma być ekstraktowana
            start_time: Czas rozpoczęcia w sekundach
            end_time: Czas zakończenia w sekundach (None = do końca)
            max_frames: Maksymalna liczba klatek do wyekstraktowania
            as_generator: Czy zwrócić generator zamiast listy

        Returns:
            Lista lub generator obiektów FrameInfo

        Raises:
            VideoNotFoundError: Gdy nie można otworzyć pliku
            FrameExtractorError: Przy innych błędach
        """
        if as_generator:
            return self._extract_frames_generator(
                video_path, interval, start_time, end_time, max_frames
            )
        else:
            return list(self._extract_frames_generator(
                video_path, interval, start_time, end_time, max_frames
            ))

    def extract_keyframes(self,
                          video_path: str,
                          threshold: float = 0.3,
                          min_interval: int = 30) -> List[FrameInfo]:
        """
        Ekstraktuje tylko klatki kluczowe (zmiany scen).

        Args:
            video_path: Ścieżka do pliku
            threshold: Próg detekcji zmiany sceny (0.0-1.0)
            min_interval: Minimalna liczba klatek między klatkami kluczowymi

        Returns:
            Lista klatek kluczowych
        """
        logger.info(f"Ekstraktuję klatki kluczowe z {video_path}")

        cap = self._open_video(video_path)
        keyframes = []
        prev_frame = None
        frame_count = 0
        last_keyframe_index = -min_interval

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None and frame_count - last_keyframe_index >= min_interval:
                    score = self._calculate_scene_change_score(prev_frame, frame)

                    if score > threshold:
                        timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                        keyframes.append(FrameInfo(
                            index=frame_count,
                            timestamp=timestamp,
                            data=frame.copy(),
                            is_keyframe=True,
                            scene_change_score=score
                        ))
                        last_keyframe_index = frame_count
                        logger.debug(f"Klatka kluczowa znaleziona: {frame_count} (score: {score:.3f})")

                prev_frame = frame.copy()
                frame_count += 1

        finally:
            cap.release()

        logger.info(f"Znaleziono {len(keyframes)} klatek kluczowych")
        return keyframes

    def extract_frames_uniform(self,
                               video_path: str,
                               num_frames: int) -> List[FrameInfo]:
        """
        Ekstraktuje określoną liczbę klatek równomiernie rozłożonych w czasie.

        Args:
            video_path: Ścieżka do pliku
            num_frames: Liczba klatek do wyekstraktowania

        Returns:
            Lista klatek
        """
        info = self.get_video_info(video_path)

        if info.total_frames <= 0:
            raise FrameExtractorError("Nie można określić liczby klatek w wideo")

        # Oblicz równomierne odstępy
        if num_frames >= info.total_frames:
            interval = 1
        else:
            interval = info.total_frames // num_frames

        return self.extract_frames(video_path, interval=interval, max_frames=num_frames)

    def extract_frames_parallel(self,
                                video_path: str,
                                interval: int = 30,
                                num_workers: int = 4) -> List[FrameInfo]:
        """
        Ekstraktuje klatki używając wielu wątków.

        Args:
            video_path: Ścieżka do pliku
            interval: Co która klatka
            num_workers: Liczba wątków

        Returns:
            Lista klatek
        """
        info = self.get_video_info(video_path)

        if info.total_frames <= 0:
            logger.warning("Przetwarzanie równoległe niedostępne dla tego źródła")
            return self.extract_frames(video_path, interval=interval)

        # Podziel pracę między wątki
        frames_per_worker = info.total_frames // num_workers
        tasks = []

        for i in range(num_workers):
            start_frame = i * frames_per_worker
            end_frame = (i + 1) * frames_per_worker if i < num_workers - 1 else info.total_frames

            tasks.append({
                'video_path': video_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'interval': interval
            })

        # Wykonaj równolegle
        all_frames = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(self._extract_frames_range, **task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                frames = future.result()
                all_frames.extend(frames)

        # Sortuj według indeksu
        all_frames.sort(key=lambda f: f.index)
        return all_frames

    def extract_rtsp_stream(self,
                            rtsp_url: str,
                            duration: float,
                            interval: int = 30,
                            callback: Optional[callable] = None) -> List[FrameInfo]:
        """
        Ekstraktuje klatki ze strumienia RTSP przez określony czas.

        Args:
            rtsp_url: URL strumienia RTSP
            duration: Czas nagrywania w sekundach
            interval: Co która klatka
            callback: Funkcja callback wywoływana dla każdej klatki

        Returns:
            Lista wyekstraktowanych klatek
        """
        logger.info(f"Rozpoczynam ekstrakcję z RTSP: {rtsp_url} przez {duration}s")

        frames = []
        frame_queue = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()

        # Wątek do przechwytywania klatek
        capture_thread = threading.Thread(
            target=self._capture_rtsp_frames,
            args=(rtsp_url, frame_queue, stop_event)
        )
        capture_thread.start()

        # Przetwarzaj klatki
        start_time = time.time()
        frame_count = 0
        extracted_count = 0

        try:
            while time.time() - start_time < duration:
                try:
                    frame = frame_queue.get(timeout=1.0)

                    if frame_count % interval == 0:
                        timestamp = time.time() - start_time
                        frame_info = FrameInfo(
                            index=frame_count,
                            timestamp=timestamp,
                            data=frame,
                            is_keyframe=False
                        )
                        frames.append(frame_info)
                        extracted_count += 1

                        if callback:
                            callback(frame_info)

                        logger.debug(f"Wyekstraktowano klatkę {extracted_count} (t={timestamp:.1f}s)")

                    frame_count += 1

                except queue.Empty:
                    continue

        finally:
            stop_event.set()
            capture_thread.join()

        logger.info(f"Zakończono ekstrakcję RTSP: {extracted_count} klatek")
        return frames

    def _open_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Otwiera plik wideo lub strumień RTSP.

        Args:
            video_path: Ścieżka do pliku lub URL

        Returns:
            Obiekt VideoCapture

        Raises:
            VideoNotFoundError: Gdy nie można otworzyć
        """
        is_rtsp = video_path.startswith('rtsp://')

        if is_rtsp:
            # Specjalne opcje dla RTSP
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Próby połączenia
            for retry in range(self.rtsp_retries):
                if cap.isOpened():
                    break

                logger.warning(f"Próba połączenia RTSP {retry + 1}/{self.rtsp_retries}")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                raise RTSPConnectionError(f"Nie można połączyć się z: {video_path}")

        else:
            # Zwykły plik
            if not Path(video_path).exists():
                raise VideoNotFoundError(f"Plik nie istnieje: {video_path}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise VideoNotFoundError(f"Nie można otworzyć pliku: {video_path}")

        return cap

    def _extract_frames_generator(self,
                                  video_path: str,
                                  interval: int,
                                  start_time: float,
                                  end_time: Optional[float],
                                  max_frames: Optional[int]) -> Generator[FrameInfo, None, None]:
        """
        Generator ekstraktujący klatki.
        """
        cap = self._open_video(video_path)

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Ustaw pozycję startową
            if start_time > 0:
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            else:
                start_frame = 0

            # Oblicz końcową klatkę
            if end_time is not None:
                end_frame = min(int(end_time * fps), total_frames)
            else:
                end_frame = total_frames if total_frames > 0 else float('inf')

            frame_count = start_frame
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= end_frame:
                    break

                if max_frames and extracted_count >= max_frames:
                    break

                if (frame_count - start_frame) % interval == 0:
                    timestamp = frame_count / fps if fps > 0 else frame_count

                    yield FrameInfo(
                        index=frame_count,
                        timestamp=timestamp,
                        data=frame.copy(),
                        is_keyframe=False
                    )

                    extracted_count += 1

                frame_count += 1

        finally:
            cap.release()

    def _extract_frames_range(self,
                              video_path: str,
                              start_frame: int,
                              end_frame: int,
                              interval: int) -> List[FrameInfo]:
        """
        Ekstraktuje klatki z określonego zakresu.
        """
        cap = self._open_video(video_path)
        frames = []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(start_frame, end_frame, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    frames.append(FrameInfo(
                        index=frame_idx,
                        timestamp=timestamp,
                        data=frame.copy(),
                        is_keyframe=False
                    ))

        finally:
            cap.release()

        return frames

    def _capture_rtsp_frames(self,
                             rtsp_url: str,
                             frame_queue: queue.Queue,
                             stop_event: threading.Event):
        """
        Wątek przechwytujący klatki z RTSP.
        """
        cap = None

        try:
            cap = self._open_video(rtsp_url)

            while not stop_event.is_set():
                ret, frame = cap.read()

                if ret:
                    try:
                        frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        # Usuń najstarszą klatkę i dodaj nową
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put(frame)
                        except queue.Empty:
                            pass
                else:
                    logger.warning("Utracono połączenie RTSP, próba ponownego połączenia...")
                    cap.release()
                    time.sleep(1)
                    cap = self._open_video(rtsp_url)

        except Exception as e:
            logger.error(f"Błąd w wątku RTSP: {e}")

        finally:
            if cap:
                cap.release()

    def _calculate_scene_change_score(self,
                                      frame1: np.ndarray,
                                      frame2: np.ndarray) -> float:
        """
        Oblicza wynik zmiany sceny między dwiema klatkami.

        Returns:
            Wynik od 0.0 (brak zmiany) do 1.0 (całkowita zmiana)
        """
        # Konwertuj do skali szarości
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Oblicz histogram
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalizuj
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Oblicz korelację (1.0 = identyczne, -1.0 = przeciwne)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Konwertuj na wynik zmiany sceny (0.0 = brak zmiany, 1.0 = duża zmiana)
        return 1.0 - max(0.0, correlation)

    def _get_codec(self, cap: cv2.VideoCapture) -> str:
        """
        Pobiera informację o kodeku.
        """
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec if codec.isprintable() else "Unknown"


# Funkcje pomocnicze
def extract_thumbnail(video_path: str,
                      timestamp: float = 0,
                      size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """
    Ekstraktuje miniaturkę z określonego momentu wideo.

    Args:
        video_path: Ścieżka do pliku
        timestamp: Czas w sekundach
        size: Rozmiar miniatury (width, height)

    Returns:
        Miniaturka jako numpy array
    """
    extractor = FrameExtractor()
    frames = extractor.extract_frames(
        video_path,
        interval=1,
        start_time=timestamp,
        max_frames=1
    )

    if not frames:
        raise FrameExtractorError("Nie można wyekstraktować miniatury")

    frame = frames[0].data
    return cv2.resize(frame, size)


def create_video_preview(video_path: str,
                         num_frames: int = 9,
                         output_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """
    Tworzy podgląd wideo jako siatkę klatek.

    Args:
        video_path: Ścieżka do pliku
        num_frames: Liczba klatek w siatce
        output_size: Rozmiar wyjściowy

    Returns:
        Obraz podglądu
    """
    extractor = FrameExtractor()
    frames = extractor.extract_frames_uniform(video_path, num_frames)

    if not frames:
        raise FrameExtractorError("Nie można utworzyć podglądu")

    # Oblicz układ siatki
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    cell_width = output_size[0] // grid_size
    cell_height = output_size[1] // grid_size

    # Utwórz pusty obraz
    preview = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Wypełnij siatkę
    for idx, frame_info in enumerate(frames[:num_frames]):
        row = idx // grid_size
        col = idx % grid_size

        # Zmień rozmiar klatki
        resized = cv2.resize(frame_info.data, (cell_width, cell_height))

        # Umieść w siatce
        y1 = row * cell_height
        y2 = (row + 1) * cell_height
        x1 = col * cell_width
        x2 = (col + 1) * cell_width

        preview[y1:y2, x1:x2] = resized

    return preview


if __name__ == '__main__':
    # Przykłady użycia
    import sys

    if len(sys.argv) > 1:
        video_file = sys.argv[1]

        # Test ekstraktora
        extractor = FrameExtractor()

        # Pobierz informacje o wideo
        print("\n=== Informacje o wideo ===")
        info = extractor.get_video_info(video_file)
        print(f"Plik: {info.path}")
        print(f"FPS: {info.fps}")
        print(f"Rozdzielczość: {info.width}x{info.height}")
        print(f"Czas trwania: {info.duration_str}")
        print(f"Liczba klatek: {info.total_frames}")
        print(f"Kodek: {info.codec}")

        # Ekstraktuj kilka klatek
        print("\n=== Ekstrakcja klatek ===")
        frames = extractor.extract_frames(video_file, interval=60, max_frames=5)
        print(f"Wyekstraktowano {len(frames)} klatek:")
        for frame in frames:
            print(f"  - Klatka {frame.index} (t={frame.timestamp:.1f}s)")

        # Znajdź klatki kluczowe
        print("\n=== Klatki kluczowe ===")
        keyframes = extractor.extract_keyframes(video_file)
        print(f"Znaleziono {len(keyframes)} klatek kluczowych")

    else:
        print("Użycie: python frame_extractor.py <plik_wideo>")