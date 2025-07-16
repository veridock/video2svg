"""
Module for extracting and processing text from video frames.
"""

import os
import pytesseract
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
from pathlib import Path


class TextProcessor:
    """
    Handles text extraction from video frames using OCR and text-to-speech conversion.
    """
    
    def __init__(self, lang="pol+eng"):
        """
        Initialize TextProcessor.
        
        Args:
            lang (str): Language for OCR. Default is Polish+English.
        """
        self.lang = lang
        self.extracted_text = ""
        
    def extract_text_from_frames(self, frames):
        """
        Extract text from a list of video frames using OCR.
        
        Args:
            frames (list): List of numpy arrays representing video frames.
            
        Returns:
            str: Extracted text from all frames.
        """
        all_text = []
        
        # Process frames in batches for efficiency
        for i, frame in enumerate(frames):
            # Convert frame to PIL Image
            pil_img = Image.fromarray(frame)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(pil_img, lang=self.lang)
            
            # Add non-empty text to results
            if text.strip():
                all_text.append(text.strip())
        
        # Combine all extracted text
        self.extracted_text = "\n".join(all_text)
        return self.extracted_text
    
    def text_to_speech(self, text=None, output_path=None, lang="pl"):
        """
        Convert text to speech and save as MP3.
        
        Args:
            text (str, optional): Text to convert. If None, uses previously extracted text.
            output_path (str, optional): Path to save the MP3 file.
            lang (str): Language for TTS. Default is Polish.
            
        Returns:
            str: Path to the generated MP3 file.
        """
        if text is None:
            text = self.extracted_text
            
        if not text:
            return None
            
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "extracted_text.mp3")
            
        # Convert text to speech
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        
        return output_path
    
    def save_text_file(self, text=None, output_path=None):
        """
        Save extracted text to a file.
        
        Args:
            text (str, optional): Text to save. If None, uses previously extracted text.
            output_path (str, optional): Path to save the text file.
            
        Returns:
            str: Path to the saved text file.
        """
        if text is None:
            text = self.extracted_text
            
        if not text:
            return None
            
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "extracted_text.txt")
            
        # Save text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return output_path
