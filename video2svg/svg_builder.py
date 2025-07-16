# src/svg_builder.py
"""
Moduł odpowiedzialny za budowanie animowanych plików SVG z klatek wideo.
Obsługuje różne metody animacji (SMIL, CSS) i optymalizację rozmiaru.
"""

import base64
import svgwrite
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass
from datetime import timedelta
import json
import re
import gzip
import logging
from io import BytesIO
import xml.etree.ElementTree as ET

# Konfiguracja loggera
logger = logging.getLogger(__name__)


@dataclass
class SVGFrame:
    """Reprezentacja pojedynczej klatki w SVG"""
    data: str  # Base64 lub ścieżka SVG
    timestamp: float
    duration: float
    index: int
    is_vectorized: bool = False
    size: int = 0  # Rozmiar w bajtach


@dataclass
class SVGAnimation:
    """Konfiguracja animacji SVG"""
    type: str = 'smil'  # 'smil', 'css', 'javascript'
    fps: float = 1.0
    loop: bool = True
    autoplay: bool = True
    controls: bool = False
    optimization_level: str = 'medium'  # 'none', 'low', 'medium', 'high'


class SVGBuilder:
    """
    Główna klasa do budowania animowanych plików SVG.
    
    Przykład użycia:
        builder = SVGBuilder()
        svg_content = builder.build_animated_svg(frames, width=640, height=480)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicjalizacja buildera.
        
        Args:
            config: Opcjonalna konfiguracja
        """
        self.config = config or {}
        self.templates = self._load_templates()
        
    def build_animated_svg(self,
                          frames: List[Union[np.ndarray, str]],
                          width: int = 640,
                          height: int = 480,
                          animation: Optional[SVGAnimation] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Buduje animowany SVG z listy klatek.
        
        Args:
            frames: Lista klatek (numpy arrays lub base64 strings)
            width: Szerokość SVG
            height: Wysokość SVG
            animation: Konfiguracja animacji
            metadata: Dodatkowe metadane
            
        Returns:
            Zawartość pliku SVG jako string
        """
        animation = animation or SVGAnimation()
        metadata = metadata or {}
        
        logger.info(f"Budowanie SVG: {len(frames)} klatek, {width}x{height}, animacja: {animation.type}")
        
        # Konwertuj klatki do formatu SVGFrame
        svg_frames = self._prepare_frames(frames, animation.fps)
        
        # Wybierz metodę budowania
        if animation.type == 'smil':
            svg_content = self._build_smil_animation(svg_frames, width, height, animation)
        elif animation.type == 'css':
            svg_content = self._build_css_animation(svg_frames, width, height, animation)
        elif animation.type == 'javascript':
            svg_content = self._build_javascript_animation(svg_frames, width, height, animation)
        else:
            raise ValueError(f"Nieznany typ animacji: {animation.type}")
        
        # Dodaj metadane
        if metadata:
            svg_content = self._add_metadata(svg_content, metadata)
        
        # Optymalizuj
        if animation.optimization_level != 'none':
            svg_content = self._optimize_svg(svg_content, animation.optimization_level)
        
        return svg_content
    
    def build_static_svg(self,
                        frame: Union[np.ndarray, str],
                        width: int = 640,
                        height: int = 480) -> str:
        """
        Buduje statyczny SVG z pojedynczej klatki.
        
        Args:
            frame: Klatka (numpy array lub base64 string)
            width: Szerokość SVG
            height: Wysokość SVG
            
        Returns:
            Zawartość pliku SVG
        """
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Konwertuj klatkę jeśli potrzeba
        if isinstance(frame, np.ndarray):
            frame_data = self._frame_to_base64(frame)
        else:
            frame_data = frame
        
        # Dodaj obraz
        image = dwg.image(
            f'data:image/png;base64,{frame_data}',
            insert=(0, 0),
            size=(width, height)
        )
        dwg.add(image)
        
        return dwg.tostring()
    
    def build_sprite_sheet_svg(self,
                             frames: List[Union[np.ndarray, str]],
                             grid_cols: int = 5,
                             frame_width: int = 128,
                             frame_height: int = 96) -> str:
        """
        Buduje SVG ze sprite sheet wszystkich klatek.
        
        Args:
            frames: Lista klatek
            grid_cols: Liczba kolumn w siatce
            frame_width: Szerokość pojedynczej klatki
            frame_height: Wysokość pojedynczej klatki
            
        Returns:
            Zawartość pliku SVG
        """
        num_frames = len(frames)
        grid_rows = (num_frames + grid_cols - 1) // grid_cols
        
        total_width = grid_cols * frame_width
        total_height = grid_rows * frame_height
        
        dwg = svgwrite.Drawing(size=(total_width, total_height))
        
        for idx, frame in enumerate(frames):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * frame_width
            y = row * frame_height
            
            # Konwertuj klatkę
            if isinstance(frame, np.ndarray):
                frame_data = self._frame_to_base64(frame, (frame_width, frame_height))
            else:
                frame_data = frame
            
            # Dodaj do sprite sheet
            image = dwg.image(
                f'data:image/png;base64,{frame_data}',
                insert=(x, y),
                size=(frame_width, frame_height)
            )
            dwg.add(image)
        
        return dwg.tostring()
    
    def build_interactive_svg(self,
                            frames: List[Union[np.ndarray, str]],
                            width: int = 640,
                            height: int = 480,
                            controls_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Buduje interaktywny SVG z kontrolkami odtwarzania.
        
        Args:
            frames: Lista klatek
            width: Szerokość SVG
            height: Wysokość SVG
            controls_config: Konfiguracja kontrolek
            
        Returns:
            Zawartość pliku SVG z interaktywnymi kontrolkami
        """
        controls_config = controls_config or {
            'show_play_button': True,
            'show_progress_bar': True,
            'show_frame_counter': True,
            'show_speed_control': False
        }
        
        # Podstawowa animacja
        animation = SVGAnimation(type='javascript', controls=True)
        svg_content = self.build_animated_svg(frames, width, height, animation)
        
        # Dodaj kontrolki
        svg_content = self._add_interactive_controls(svg_content, controls_config, len(frames))
        
        return svg_content
    
    def _prepare_frames(self, 
                       frames: List[Union[np.ndarray, str]], 
                       fps: float) -> List[SVGFrame]:
        """
        Przygotowuje klatki do animacji SVG.
        """
        svg_frames = []
        frame_duration = 1.0 / fps
        
        for idx, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                frame_data = self._frame_to_base64(frame)
            else:
                frame_data = frame
            
            svg_frame = SVGFrame(
                data=frame_data,
                timestamp=idx * frame_duration,
                duration=frame_duration,
                index=idx,
                is_vectorized=False,
                size=len(frame_data)
            )
            svg_frames.append(svg_frame)
        
        return svg_frames
    
    def _build_smil_animation(self,
                            frames: List[SVGFrame],
                            width: int,
                            height: int,
                            animation: SVGAnimation) -> str:
        """
        Buduje animację używając SMIL.
        """
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Dodaj definicje dla optymalizacji
        defs = dwg.defs
        dwg.add(defs)
        
        total_duration = len(frames) / animation.fps
        
        # Dodaj każdą klatkę z animacją
        for frame in frames:
            # Unikalne ID dla klatki
            frame_id = f"frame_{frame.index}"
            
            # Dodaj obraz
            image = dwg.image(
                f'data:image/png;base64,{frame.data}',
                insert=(0, 0),
                size=(width, height),
                id=frame_id
            )
            
            # Oblicz czasy animacji
            begin_time = frame.timestamp
            end_time = frame.timestamp + frame.duration
            
            # Animacja widoczności
            values = self._calculate_visibility_values(
                begin_time, end_time, total_duration
            )
            
            animate = dwg.animate(
                'opacity',
                dur=f'{total_duration}s',
                values=values['values'],
                keyTimes=values['keyTimes'],
                repeatCount='indefinite' if animation.loop else '1'
            )
            
            image.add(animate)
            dwg.add(image)
        
        return dwg.tostring()
    
    def _build_css_animation(self,
                           frames: List[SVGFrame],
                           width: int,
                           height: int,
                           animation: SVGAnimation) -> str:
        """
        Buduje animację używając CSS.
        """
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Styl CSS z animacją
        total_duration = len(frames) / animation.fps
        css_rules = [
            f"@keyframes slideshow {{",
        ]
        
        # Generuj keyframes
        for i, frame in enumerate(frames):
            start_percent = (i / len(frames)) * 100
            end_percent = ((i + 1) / len(frames)) * 100
            
            css_rules.extend([
                f"  {start_percent:.1f}% {{ opacity: 1; }}",
                f"  {end_percent:.1f}% {{ opacity: 0; }}"
            ])
        
        css_rules.append("}")
        
        # Dodaj reguły dla każdej klatki
        for i, frame in enumerate(frames):
            delay = (i / len(frames)) * total_duration
            css_rules.extend([
                f"#frame_{i} {{",
                f"  opacity: 0;",
                f"  animation: slideshow {total_duration}s infinite;",
                f"  animation-delay: {delay}s;",
                f"}}"
            ])
        
        # Dodaj styl do SVG
        style = dwg.style("\n".join(css_rules))
        dwg.add(style)
        
        # Dodaj klatki
        for frame in frames:
            image = dwg.image(
                f'data:image/png;base64,{frame.data}',
                insert=(0, 0),
                size=(width, height),
                id=f"frame_{frame.index}"
            )
            dwg.add(image)
        
        return dwg.tostring()
    
    def _build_javascript_animation(self,
                                  frames: List[SVGFrame],
                                  width: int,
                                  height: int,
                                  animation: SVGAnimation) -> str:
        """
        Buduje animację używając JavaScript.
        """
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Dodaj klatki (początkowo ukryte)
        for frame in frames:
            image = dwg.image(
                f'data:image/png;base64,{frame.data}',
                insert=(0, 0),
                size=(width, height),
                id=f"frame_{frame.index}",
                style="display: none;"
            )
            dwg.add(image)
        
        # JavaScript do animacji
        js_code = f"""
        <![CDATA[
        (function() {{
            const frames = {len(frames)};
            const fps = {animation.fps};
            const loop = {str(animation.loop).lower()};
            const autoplay = {str(animation.autoplay).lower()};
            
            let currentFrame = 0;
            let isPlaying = autoplay;
            let intervalId = null;
            
            function showFrame(index) {{
                // Ukryj wszystkie klatki
                for (let i = 0; i < frames; i++) {{
                    const frame = document.getElementById('frame_' + i);
                    if (frame) frame.style.display = 'none';
                }}
                
                // Pokaż aktualną klatkę
                const frame = document.getElementById('frame_' + index);
                if (frame) frame.style.display = 'block';
            }}
            
            function nextFrame() {{
                currentFrame = (currentFrame + 1) % frames;
                if (!loop && currentFrame === 0) {{
                    stop();
                    return;
                }}
                showFrame(currentFrame);
            }}
            
            function play() {{
                if (!isPlaying) {{
                    isPlaying = true;
                    intervalId = setInterval(nextFrame, 1000 / fps);
                }}
            }}
            
            function pause() {{
                if (isPlaying) {{
                    isPlaying = false;
                    clearInterval(intervalId);
                }}
            }}
            
            function stop() {{
                pause();
                currentFrame = 0;
                showFrame(currentFrame);
            }}
            
            function setFrame(index) {{
                if (index >= 0 && index < frames) {{
                    currentFrame = index;
                    showFrame(currentFrame);
                }}
            }}
            
            // API publiczne
            window.svgAnimation = {{
                play: play,
                pause: pause,
                stop: stop,
                setFrame: setFrame,
                getCurrentFrame: () => currentFrame,
                getTotalFrames: () => frames,
                isPlaying: () => isPlaying
            }};
            
            // Inicjalizacja
            showFrame(0);
            if (autoplay) {{
                play();
            }}
        }})();
        ]]>
        """
        
        script = dwg.script(js_code, type="text/javascript")
        dwg.add(script)
        
        # Dodaj kontrolki jeśli włączone
        if animation.controls:
            controls = self._create_basic_controls(width, height)
            dwg.add(controls)
        
        return dwg.tostring()
    
    def _add_interactive_controls(self,
                                svg_content: str,
                                controls_config: Dict[str, Any],
                                num_frames: int) -> str:
        """
        Dodaje interaktywne kontrolki do SVG.
        """
        # Parse SVG
        root = ET.fromstring(svg_content)
        
        # Utwórz grupę kontrolek
        controls_group = ET.Element('g', attrib={
            'id': 'controls',
            'transform': f'translate(10, {root.get("height", "480")} - 50)'
        })
        
        y_offset = 0
        
        # Przycisk Play/Pause
        if controls_config.get('show_play_button', True):
            play_button = self._create_play_button()
            controls_group.append(play_button)
        
        # Pasek postępu
        if controls_config.get('show_progress_bar', True):
            progress_bar = self._create_progress_bar(300, num_frames)
            progress_bar.set('transform', f'translate(50, {y_offset})')
            controls_group.append(progress_bar)
        
        # Licznik klatek
        if controls_config.get('show_frame_counter', True):
            counter = self._create_frame_counter()
            counter.set('transform', f'translate(370, {y_offset + 5})')
            controls_group.append(counter)
        
        # Kontrola prędkości
        if controls_config.get('show_speed_control', False):
            speed_control = self._create_speed_control()
            speed_control.set('transform', f'translate(450, {y_offset})')
            controls_group.append(speed_control)
        
        # Dodaj kontrolki do SVG
        root.append(controls_group)
        
        # Dodaj dodatkowy JavaScript dla kontrolek
        controls_js = self._create_controls_javascript()
        script = ET.Element('script', attrib={'type': 'text/javascript'})
        script.text = controls_js
        root.append(script)
        
        return ET.tostring(root, encoding='unicode')
    
    def _create_play_button(self) -> ET.Element:
        """Tworzy przycisk play/pause."""
        group = ET.Element('g', attrib={
            'id': 'playButton',
            'cursor': 'pointer',
            'onclick': 'togglePlayPause()'
        })
        
        # Tło przycisku
        rect = ET.Element('rect', attrib={
            'width': '40',
            'height': '40',
            'rx': '5',
            'fill': '#333',
            'stroke': '#fff',
            'stroke-width': '2'
        })
        group.append(rect)
        
        # Ikona play
        play = ET.Element('polygon', attrib={
            'id': 'playIcon',
            'points': '15,10 15,30 30,20',
            'fill': '#fff'
        })
        group.append(play)
        
        # Ikona pause (początkowo ukryta)
        pause = ET.Element('g', attrib={
            'id': 'pauseIcon',
            'style': 'display: none;'
        })
        rect1 = ET.Element('rect', attrib={
            'x': '14', 'y': '10',
            'width': '6', 'height': '20',
            'fill': '#fff'
        })
        rect2 = ET.Element('rect', attrib={
            'x': '24', 'y': '10',
            'width': '6', 'height': '20',
            'fill': '#fff'
        })
        pause.append(rect1)
        pause.append(rect2)
        group.append(pause)
        
        return group
    
    def _create_progress_bar(self, width: int, num_frames: int) -> ET.Element:
        """Tworzy pasek postępu."""
        group = ET.Element('g', attrib={'id': 'progressBar'})
        
        # Tło paska
        bg = ET.Element('rect', attrib={
            'width': str(width),
            'height': '10',
            'rx': '5',
            'fill': '#555'
        })
        group.append(bg)
        
        # Wypełnienie paska
        fill = ET.Element('rect', attrib={
            'id': 'progressFill',
            'width': '0',
            'height': '10',
            'rx': '5',
            'fill': '#4CAF50'
        })
        group.append(fill)
        
        # Uchwyt
        handle = ET.Element('circle', attrib={
            'id': 'progressHandle',
            'cx': '0',
            'cy': '5',
            'r': '8',
            'fill': '#fff',
            'stroke': '#4CAF50',
            'stroke-width': '2',
            'cursor': 'pointer'
        })
        group.append(handle)
        
        return group
    
    def _create_frame_counter(self) -> ET.Element:
        """Tworzy licznik klatek."""
        text = ET.Element('text', attrib={
            'id': 'frameCounter',
            'font-family': 'Arial',
            'font-size': '14',
            'fill': '#fff'
        })
        text.text = '0 / 0'
        return text
    
    def _create_speed_control(self) -> ET.Element:
        """Tworzy kontrolkę prędkości."""
        group = ET.Element('g', attrib={'id': 'speedControl'})
        
        # Etykieta
        label = ET.Element('text', attrib={
            'x': '0',
            'y': '-5',
            'font-family': 'Arial',
            'font-size': '12',
            'fill': '#fff'
        })
        label.text = 'Speed:'
        group.append(label)
        
        # Przyciski prędkości
        speeds = ['0.5x', '1x', '2x']
        for i, speed in enumerate(speeds):
            btn = ET.Element('rect', attrib={
                'x': str(i * 35),
                'y': '0',
                'width': '30',
                'height': '20',
                'rx': '3',
                'fill': '#666' if speed != '1x' else '#4CAF50',
                'cursor': 'pointer',
                'onclick': f"setSpeed({speed[:-1]})"
            })
            group.append(btn)
            
            text = ET.Element('text', attrib={
                'x': str(i * 35 + 15),
                'y': '14',
                'text-anchor': 'middle',
                'font-family': 'Arial',
                'font-size': '11',
                'fill': '#fff',
                'pointer-events': 'none'
            })
            text.text = speed
            group.append(text)
        
        return group
    
    def _create_controls_javascript(self) -> str:
        """Tworzy JavaScript dla kontrolek."""
        return """
        <![CDATA[
        function togglePlayPause() {
            if (window.svgAnimation.isPlaying()) {
                window.svgAnimation.pause();
                document.getElementById('playIcon').style.display = 'block';
                document.getElementById('pauseIcon').style.display = 'none';
            } else {
                window.svgAnimation.play();
                document.getElementById('playIcon').style.display = 'none';
                document.getElementById('pauseIcon').style.display = 'block';
            }
        }
        
        function updateProgress() {
            const current = window.svgAnimation.getCurrentFrame();
            const total = window.svgAnimation.getTotalFrames();
            const percent = (current / total) * 100;
            
            // Aktualizuj pasek
            const fill = document.getElementById('progressFill');
            const handle = document.getElementById('progressHandle');
            const bar = document.getElementById('progressBar');
            
            if (fill && handle && bar) {
                const barWidth = bar.children[0].getAttribute('width');
                const fillWidth = (percent / 100) * barWidth;
                
                fill.setAttribute('width', fillWidth);
                handle.setAttribute('cx', fillWidth);
            }
            
            // Aktualizuj licznik
            const counter = document.getElementById('frameCounter');
            if (counter) {
                counter.textContent = `${current + 1} / ${total}`;
            }
        }
        
        function setSpeed(speed) {
            // Implementacja zmiany prędkości
            console.log('Set speed:', speed);
        }
        
        // Aktualizuj co klatkę
        setInterval(updateProgress, 100);
        ]]>
        """
    
    def _calculate_visibility_values(self,
                                   begin_time: float,
                                   end_time: float,
                                   total_duration: float) -> Dict[str, str]:
        """
        Oblicza wartości dla animacji widoczności SMIL.
        """
        # Normalizuj czasy do zakresu 0-1
        begin_norm = begin_time / total_duration
        end_norm = end_time / total_duration
        
        # Wartości opacity: 0 przed, 1 podczas, 0 po
        values = f"0;0;1;1;0;0"
        
        # Czasy kluczowe
        # Mała delta dla płynnego przejścia
        delta = 0.001
        keyTimes = f"0;{begin_norm - delta};{begin_norm};{end_norm};{end_norm + delta};1"
        
        return {
            'values': values,
            'keyTimes': keyTimes
        }
    
    def _frame_to_base64(self, 
                        frame: np.ndarray,
                        size: Optional[Tuple[int, int]] = None,
                        compression: int = 8) -> str:
        """
        Konwertuje klatkę numpy array do base64.
        """
        # Zmień rozmiar jeśli podano
        if size:
            frame = cv2.resize(frame, size)
        
        # Koduj do PNG
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        _, buffer = cv2.imencode('.png', frame, encode_params)
        
        # Konwertuj do base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def _optimize_svg(self, svg_content: str, level: str) -> str:
        """
        Optymalizuje zawartość SVG.
        """
        if level == 'low':
            # Podstawowa optymalizacja - usuń zbędne spacje
            svg_content = re.sub(r'>\s+<', '><', svg_content)
            svg_content = re.sub(r'\s+', ' ', svg_content)
            
        elif level == 'medium':
            # Średnia optymalizacja
            svg_content = self._optimize_svg(svg_content, 'low')
            # Usuń komentarze
            svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)
            # Skróć liczby
            svg_content = re.sub(r'(\d+\.\d{3})\d+', r'\1', svg_content)
            
        elif level == 'high':
            # Wysoka optymalizacja
            svg_content = self._optimize_svg(svg_content, 'medium')
            # Usuń niepotrzebne atrybuty
            svg_content = re.sub(r'\s+id="[^"]*"', '', svg_content)
            # Kompresja atrybutów
            svg_content = self._compress_attributes(svg_content)
        
        return svg_content.strip()
    
    def _compress_attributes(self, svg_content: str) -> str:
        """
        Kompresuje atrybuty SVG.
        """
        # Zastąp długie nazwy atrybutów skrótami
        replacements = {
            'stroke-width': 'sw',
            'stroke-linecap': 'slc',
            'stroke-linejoin': 'slj',
            'font-family': 'ff',
            'font-size': 'fs',
            'text-anchor': 'ta'
        }
        
        for long_name, short_name in replacements.items():
            svg_content = svg_content.replace(long_name, short_name)
        
        return svg_content
    
    def _add_metadata(self, svg_content: str, metadata: Dict[str, Any]) -> str:
        """
        Dodaje metadane do SVG.
        """
        # Parse SVG
        root = ET.fromstring(svg_content)
        
        # Utwórz element metadata
        metadata_elem = ET.Element('metadata')
        
        # Dodaj informacje
        desc = ET.SubElement(metadata_elem, 'desc')
        desc.text = json.dumps(metadata, indent=2)
        
        # Wstaw na początku
        root.insert(0, metadata_elem)
        
        return ET.tostring(root, encoding='unicode')
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Ładuje szablony SVG.
        """
        templates = {
            'basic': """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
{content}
</svg>""",
            
            'with_controls': """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
<defs>
  <style type="text/css">
    .control-button {{ cursor: pointer; }}
    .control-button:hover {{ opacity: 0.8; }}
  </style>
</defs>
{content}
{controls}
</svg>"""
        }
        
        return templates
    
    def _create_basic_controls(self, width: int, height: int) -> svgwrite.container.Group:
        """
        Tworzy podstawowe kontrolki odtwarzania.
        """
        group = svgwrite.container.Group(id='basic-controls')
        
        # Pozycja kontrolek
        controls_y = height - 50
        
        # Tło dla kontrolek
        bg = svgwrite.shapes.Rect(
            insert=(0, controls_y),
            size=(width, 50),
            fill='rgba(0,0,0,0.7)'
        )
        group.add(bg)
        
        # Przycisk play/pause
        play_btn = svgwrite.container.Group(
            id='play-pause-btn',
            transform=f'translate(20, {controls_y + 10})'
        )
        
        # Ikona play
        play_icon = svgwrite.shapes.Polygon(
            points=[(0, 0), (0, 30), (25, 15)],
            fill='white',
            id='play-icon'
        )
        play_btn.add(play_icon)
        
        group.add(play_btn)
        
        return group


# Funkcje pomocnicze
def create_thumbnail_svg(video_frame: np.ndarray,
                        size: Tuple[int, int] = (320, 240),
                        add_play_button: bool = True) -> str:
    """
    Tworzy SVG miniaturkę z opcjonalnym przyciskiem play.
    """
    builder = SVGBuilder()
    svg = builder.build_static_svg(video_frame, size[0], size[1])
    
    if add_play_button:
        # Dodaj przycisk play
        root = ET.fromstring(svg)
        
        # Półprzezroczyste tło
        overlay = ET.Element('rect', attrib={
            'width': str(size[0]),
            'height': str(size[1]),
            'fill': 'rgba(0,0,0,0.3)'
        })
        root.append(overlay)
        
        # Przycisk play
        play_button = ET.Element('polygon', attrib={
            'points': f'{size[0]//2-20},{size[1]//2-25} '
                     f'{size[0]//2-20},{size[1]//2+25} '
                     f'{size[0]//2+30},{size[1]//2}',
            'fill': 'rgba(255,255,255,0.9)',
            'cursor': 'pointer'
        })
        root.append(play_button)
        
        svg = ET.tostring(root, encoding='unicode')
    
    return svg


def estimate_svg_size(num_frames: int,
                     frame_width: int,
                     frame_height: int,
                     compression_level: int = 8) -> int:
    """
    Szacuje rozmiar końcowego pliku SVG.
    
    Returns:
        Szacowany rozmiar w bajtach
    """
    # Średni rozmiar skompresowanego PNG
    avg_png_size = (frame_width * frame_height * 3) // (compression_level + 1)
    
    # Base64 zwiększa rozmiar o ~33%
    base64_size = avg_png_size * 1.33
    
    # Dodatkowy narzut SVG (tagi, animacja)
    svg_overhead = 1000 + (num_frames * 200)  # Szacunkowy narzut na klatkę
    
    total_size = int(num_frames * base64_size + svg_overhead)
    
    return total_size


if __name__ == '__main__':
    # Przykład użycia
    print("SVG Builder - Przykłady")
    
    # Test z przykładowymi danymi
    width, height = 640, 480
    num_frames = 10
    
    # Generuj przykładowe klatki
    frames = []
    for i in range(num_frames):
        # Gradient kolorów
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [i * 25, 0, 255 - i * 25]  # Gradient od czerw