# -*- coding: utf-8 -*-
"""video2svg.converter - Core conversion logic copied from src/converter.py"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import svgwrite
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "VideoToSVGConverter",
    "VideoToSVGError",
    "VideoNotFoundError",
    "ConversionError",
    "ConfigurationError",
    "create_default_config",
]


class VideoToSVGError(Exception):
    """Base exception"""


class VideoNotFoundError(VideoToSVGError):
    """Raised when input file not found"""


class ConversionError(VideoToSVGError):
    """Raised on conversion failure"""


class ConfigurationError(VideoToSVGError):
    """Raised on invalid config"""


class VideoToSVGConverter:
    """Main converter class (see original src/converter.py for detailed docstrings)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        self.frames: List[str] = []
        self.temp_dir: Optional[str] = None
        self.rtsp_timeout = 10
        self.rtsp_retries = 3

    # --- configuration helpers -------------------------------------------------
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        defaults = create_default_config()
        for section, values in defaults.items():
            if section not in config:
                config[section] = values
            else:
                for key, value in values.items():
                    config[section].setdefault(key, value)
        return config

    # --- public API ------------------------------------------------------------
    def convert(self, input_path: str, output_path: str,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = datetime.now()
        logger.info("Starting conversion %s -> %s", input_path, output_path)

        try:
            # validate file existence unless RTSP
            if not input_path.startswith("rtsp://") and not Path(input_path).exists():
                raise VideoNotFoundError(f"File not found: {input_path}")

            # 1. extract frames
            self._update_progress(progress_callback, 10, "Extracting frames…")
            frames = self.extract_frames(input_path)
            if not frames:
                raise ConversionError("Frame extraction returned 0 frames")

            # 2. optional vectorize (placeholder)
            if self.config["advanced"]["vectorize"]:
                self._update_progress(progress_callback, 40, "Vectorizing frames…")
                frames = self.vectorize_frames(frames)

            # 3. build SVG
            self._update_progress(progress_callback, 70, "Building SVG…")
            svg_content = self.build_svg(frames)

            # 4. optimize
            if self.config["svg"]["optimization"]:
                self._update_progress(progress_callback, 85, "Optimizing SVG…")
                svg_content = self._optimize_svg(svg_content)

            # 5. save
            self._update_progress(progress_callback, 95, "Saving…")
            output_size = self._save_svg(svg_content, output_path)

            duration = (datetime.now() - start_time).total_seconds()
            self._update_progress(progress_callback, 100, "Done!")
            return {
                "status": "success",
                "output_file": output_path,
                "size": output_size,
                "duration": duration,
                "frames_count": len(frames),
                "fps": self.config["svg"]["output_fps"],
            }
        except Exception as exc:
            logger.error("Conversion error: %s", exc)
            raise
        finally:
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)

    # -------------------------------------------------------------------------
    def extract_frames(self, video_path: str) -> List[str]:
        frames: List[str] = []
        frame_interval = self.config["extraction"]["frame_interval"]
        start_time = self.config["extraction"]["start_time"]
        end_time = self.config["extraction"]["end_time"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoNotFoundError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        end_frame = total_frames if end_time is None else min(int(end_time * fps), total_frames)

        if self.config["advanced"]["parallel_processing"]:
            frames = self._extract_frames_parallel(cap, frame_interval, end_frame)
        else:
            frame_idx = 0
            while cap.isOpened() and frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frames.append(self.process_frame(frame))
                frame_idx += 1
        cap.release()
        logger.info("Extracted %d frames", len(frames))
        return frames

    def _extract_frames_parallel(self, cap, frame_interval: int, end_frame: int) -> List[str]:
        gather: List[tuple[int, np.ndarray]] = []
        idx = 0
        while cap.isOpened() and idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                gather.append((idx, frame.copy()))
            idx += 1
        frames: List[str] = [None] * len(gather)  # type: ignore
        with ThreadPoolExecutor(max_workers=self.config["advanced"]["num_workers"]) as executor:
            future_to_i = {executor.submit(self.process_frame, f): i for i, (_, f) in enumerate(gather)}
            for fut in as_completed(future_to_i):
                frames[future_to_i[fut]] = fut.result()
        return frames

    def process_frame(self, frame: np.ndarray) -> str:
        w_target = self.config["processing"]["output_width"]
        h_target = self.config["processing"]["output_height"]
        h, w = frame.shape[:2]
        if h_target is None:
            h_target = int(h * w_target / w)
        if (w, h) != (w_target, h_target):
            frame = cv2.resize(frame, (w_target, h_target))

        mode = self.config["processing"]["color_mode"]
        if mode == "grayscale":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif mode == "monochrome":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, self.config["processing"]["compression_level"]]
        _, buf = cv2.imencode(".png", frame, encode_params)
        return base64.b64encode(buf).decode()

    # placeholder vectorization
    def vectorize_frames(self, frames: List[str]) -> List[str]:
        logger.warning("Vectorization not implemented yet")
        return frames

    def build_svg(self, frames: List[str]) -> str:
        width = self.config["processing"]["output_width"]
        height = self.config["processing"]["output_height"] or 480
        fps = self.config["svg"]["output_fps"]
        duration = len(frames) / fps

        dwg = svgwrite.Drawing(size=(width, height))
        if self.config["svg"]["optimization"]:
            dwg.add(dwg.defs)

        for i, data in enumerate(frames):
            begin = i / fps
            end = (i + 1) / fps
            img = dwg.image(f"data:image/png;base64,{data}", insert=(0, 0), size=(width, height))
            if self.config["svg"]["animation_type"] == "smil":
                img.add(dwg.animate(
                    "opacity",
                    dur=f"{duration}s",
                    values="0;0;1;1;0;0",
                    keyTimes=f"0;{begin/duration};{begin/duration};{end/duration};{end/duration};1",
                    repeatCount="indefinite",
                ))
            dwg.add(img)
        return dwg.tostring()

    # dumb optimization stub
    def _optimize_svg(self, svg: str) -> str:
        return svg.replace("  ", " ")

    def _save_svg(self, svg: str, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return os.path.getsize(path)

    def _update_progress(self, cb, pct: int, msg: str):
        if cb:
            cb(pct, msg)


# -----------------------------------------------------------------------------

def create_default_config() -> Dict[str, Any]:
    return {
        "extraction": {"frame_interval": 30, "start_time": 0, "end_time": None},
        "processing": {
            "output_width": 640,
            "output_height": None,
            "compression_level": 8,
            "color_mode": "color",
        },
        "svg": {
            "output_fps": 1,
            "optimization": True,
            "embed_method": "base64",
            "animation_type": "smil",
        },
        "advanced": {
            "vectorize": False,
            "vectorize_threshold": 128,
            "max_file_size": 100,
            "use_cache": True,
            "parallel_processing": True,
            "num_workers": 4,
        },
    }