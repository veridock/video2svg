# tests/test_converter.py
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
from src.converter import VideoToSVGConverter
import base64
import tempfile
import os


class TestVideoToSVGConverter(unittest.TestCase):
    """Testy dla głównego konwertera"""

    def setUp(self):
        """Przygotowanie do testów"""
        self.config = {
            'frame_interval': 30,
            'output_width': 640,
            'output_height': 480,
            'output_fps': 1,
            'vectorize': False,
            'compression_level': 8
        }
        self.converter = VideoToSVGConverter(self.config)

    def test_initialization(self):
        """Test inicjalizacji konwertera"""
        self.assertEqual(self.converter.config, self.config)
        self.assertEqual(self.converter.frames, [])

    @patch('cv2.VideoCapture')
    def test_extract_frames(self, mock_capture):
        """Test ekstrakcji klatek z wideo"""
        # Przygotuj mock video capture
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap

        # Symuluj 3 klatki
        test_frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8) * 128),
            (True, np.ones((480, 640, 3), dtype=np.uint8) * 255),
            (False, None)  # Koniec wideo
        ]
        mock_cap.read.side_effect = test_frames
        mock_cap.isOpened.return_value = True

        # Testuj ekstrakcję
        frames = self.converter.extract_frames('test_video.mp4')

        # Sprawdź wyniki
        self.assertEqual(len(frames), 1)  # Tylko pierwsza klatka (frame_interval=30)
        mock_capture.assert_called_once_with('test_video.mp4')
        mock_cap.release.assert_called_once()

    def test_process_frame(self):
        """Test przetwarzania pojedynczej klatki"""
        # Stwórz testową klatkę
        test_frame = np.ones((960, 1280, 3), dtype=np.uint8) * 128

        # Przetwórz klatkę
        result = self.converter.process_frame(test_frame)

        # Sprawdź czy wynik to base64
        self.assertIsInstance(result, str)

        # Sprawdź czy można zdekodować
        decoded = base64.b64decode(result)
        self.assertGreater(len(decoded), 0)

    @patch('cv2.resize')
    def test_process_frame_resize(self, mock_resize):
        """Test zmiany rozmiaru klatki"""
        # Przygotuj dane
        test_frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_resize.return_value = np.ones((360, 640, 3), dtype=np.uint8)

        # Przetwórz
        self.converter.process_frame(test_frame)

        # Sprawdź czy resize został wywołany z poprawnymi parametrami
        mock_resize.assert_called_once()
        args = mock_resize.call_args[0]
        self.assertEqual(args[1], (640, 360))  # (width, height)

    def test_build_svg_structure(self):
        """Test budowania struktury SVG"""
        # Przygotuj testowe dane
        test_frames = ['base64_frame_1', 'base64_frame_2']

        # Zbuduj SVG
        svg_content = self.converter.build_svg(test_frames)

        # Sprawdź podstawową strukturę
        self.assertIn('<svg', svg_content)
        self.assertIn('</svg>', svg_content)
        self.assertIn('image', svg_content)
        self.assertIn('animate', svg_content)
        self.assertIn('data:image/png;base64,base64_frame_1', svg_content)

    def test_build_svg_animation_timing(self):
        """Test poprawności czasowania animacji"""
        # Przygotuj 4 klatki
        test_frames = ['frame1', 'frame2', 'frame3', 'frame4']

        # Zbuduj SVG
        svg_content = self.converter.build_svg(test_frames)

        # Sprawdź czasowanie
        self.assertIn('dur="4s"', svg_content)  # 4 klatki przy 1 fps = 4s
        self.assertIn('repeatCount="indefinite"', svg_content)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch.object(VideoToSVGConverter, 'extract_frames')
    @patch.object(VideoToSVGConverter, 'build_svg')
    def test_convert_full_process(self, mock_build_svg, mock_extract_frames, mock_open):
        """Test pełnego procesu konwersji"""
        # Przygotuj mocki
        mock_extract_frames.return_value = ['frame1', 'frame2']
        mock_build_svg.return_value = '<svg>test</svg>'

        # Wykonaj konwersję
        self.converter.convert('input.mp4', 'output.svg')

        # Sprawdź wywołania
        mock_extract_frames.assert_called_once_with('input.mp4')
        mock_build_svg.assert_called_once_with(['frame1', 'frame2'])
        mock_open.assert_called_once_with('output.svg', 'w')
        mock_open().write.assert_called_once_with('<svg>test</svg>')

    def test_convert_with_vectorization(self):
        """Test konwersji z wektoryzacją"""
        self.converter.config['vectorize'] = True

        with patch.object(self.converter, 'vectorize_frames') as mock_vectorize:
            with patch.object(self.converter, 'extract_frames') as mock_extract:
                with patch.object(self.converter, 'build_svg'):
                    with patch('builtins.open', unittest.mock.mock_open()):
                        mock_extract.return_value = ['frame1']
                        mock_vectorize.return_value = ['vectorized1']

                        self.converter.convert('input.mp4', 'output.svg')

                        mock_vectorize.assert_called_once_with(['frame1'])


class TestFrameExtractor(unittest.TestCase):
    """Testy dla ekstrakcji klatek"""

    @patch('cv2.VideoCapture')
    def test_rtsp_stream_handling(self, mock_capture):
        """Test obsługi strumieni RTSP"""
        config = {'frame_interval': 1, 'output_width': 640, 'output_height': 480}
        converter = VideoToSVGConverter(config)

        # Symuluj strumień RTSP
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(False, None)]  # Brak połączenia

        # Test
        frames = converter.extract_frames('rtsp://example.com/stream')

        # Sprawdź
        self.assertEqual(len(frames), 0)
        mock_capture.assert_called_with('rtsp://example.com/stream')

    def test_frame_interval_extraction(self):
        """Test ekstrakcji co n-tą klatkę"""
        # Różne interwały
        intervals = [1, 10, 30, 60]

        for interval in intervals:
            with self.subTest(interval=interval):
                config = {'frame_interval': interval, 'output_width': 640, 'output_height': 480}
                converter = VideoToSVGConverter(config)

                with patch('cv2.VideoCapture') as mock_capture:
                    mock_cap = MagicMock()
                    mock_capture.return_value = mock_cap
                    mock_cap.isOpened.return_value = True

                    # Symuluj 120 klatek
                    frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(120)]
                    frames.append((False, None))
                    mock_cap.read.side_effect = frames

                    # Ekstraktuj
                    result = converter.extract_frames('test.mp4')

                    # Sprawdź liczbę wyekstraktowanych klatek
                    expected_frames = 120 // interval
                    self.assertEqual(len(result), expected_frames)


class TestSVGBuilder(unittest.TestCase):
    """Testy dla budowania SVG"""

    def test_svg_output_validity(self):
        """Test poprawności wygenerowanego SVG"""
        config = {'output_width': 640, 'output_height': 480, 'output_fps': 1}
        converter = VideoToSVGConverter(config)

        # Generuj SVG
        svg_content = converter.build_svg(['frame1', 'frame2'])

        # Podstawowa walidacja XML
        self.assertTrue(svg_content.startswith('<?xml') or svg_content.startswith('<svg'))
        self.assertTrue(svg_content.endswith('</svg>'))
        self.assertEqual(svg_content.count('<svg'), 1)
        self.assertEqual(svg_content.count('</svg>'), 1)

    def test_svg_dimensions(self):
        """Test wymiarów SVG"""
        config = {'output_width': 1920, 'output_height': 1080, 'output_fps': 1}
        converter = VideoToSVGConverter(config)

        svg_content = converter.build_svg(['frame1'])

        # Sprawdź wymiary
        self.assertIn('width="1920"', svg_content)
        self.assertIn('height="1080"', svg_content)

    def test_empty_frames_handling(self):
        """Test obsługi pustej listy klatek"""
        config = {'output_width': 640, 'output_height': 480, 'output_fps': 1}
        converter = VideoToSVGConverter(config)

        # Powinien zwrócić pusty SVG
        svg_content = converter.build_svg([])

        self.assertIn('<svg', svg_content)
        self.assertIn('</svg>', svg_content)
        self.assertNotIn('<image', svg_content)


class TestIntegration(unittest.TestCase):
    """Testy integracyjne"""

    def test_end_to_end_conversion(self):
        """Test konwersji od początku do końca"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Przygotuj pliki
            input_path = os.path.join(temp_dir, 'test_video.mp4')
            output_path = os.path.join(temp_dir, 'output.svg')

            # Stwórz testowe wideo (3 klatki)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(input_path, fourcc, 1, (100, 100))

            for i in range(3):
                frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 85)
                video_writer.write(frame)

            video_writer.release()

            # Konwertuj
            config = {
                'frame_interval': 1,
                'output_width': 100,
                'output_height': 100,
                'output_fps': 1,
                'vectorize': False
            }
            converter = VideoToSVGConverter(config)
            converter.convert(input_path, output_path)

            # Sprawdź wynik
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, 'r') as f:
                svg_content = f.read()

            self.assertIn('<svg', svg_content)
            self.assertIn('</svg>', svg_content)
            self.assertIn('image', svg_content)
            self.assertIn('animate', svg_content)

    def test_error_handling(self):
        """Test obsługi błędów"""
        config = {'frame_interval': 30, 'output_width': 640, 'output_height': 480}
        converter = VideoToSVGConverter(config)

        # Test nieistniejącego pliku
        with self.assertRaises(Exception):
            converter.convert('non_existent_file.mp4', 'output.svg')

        # Test nieprawidłowego URL RTSP
        frames = converter.extract_frames('rtsp://invalid.url/stream')
        self.assertEqual(len(frames), 0)


if __name__ == '__main__':
    # Uruchom testy z raportowaniem
    unittest.main(verbosity=2)