"""
Multimodal Document Processors for Graph RAG
Handles image (OCR/captioning) and audio (transcription) processing
"""

import os
import tempfile
import base64
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path

# Core imports
from agno.document import Document

# Image processing imports
try:
    from PIL import Image
    import pytesseract
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OpenAI for vision and audio
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images using OCR and AI vision models"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
    
    def process_image(self, image_path: str, use_vision_model: bool = True) -> Dict[str, Any]:
        """Process image with both OCR and AI vision"""
        results = {
            "ocr_text": "",
            "vision_description": "",
            "metadata": {
                "file_path": image_path,
                "file_size": 0,
                "dimensions": None,
                "format": None,
                "processed_at": datetime.now().isoformat()
            }
        }
        
        try:
            # Get image metadata
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    results["metadata"]["dimensions"] = img.size
                    results["metadata"]["format"] = img.format
                    results["metadata"]["file_size"] = os.path.getsize(image_path)
            
            # OCR extraction
            if PIL_AVAILABLE:
                try:
                    results["ocr_text"] = pytesseract.image_to_string(Image.open(image_path))
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            # AI Vision description
            if use_vision_model and self.openai_client:
                try:
                    results["vision_description"] = self._get_vision_description(image_path)
                except Exception as e:
                    logger.warning(f"Vision model failed: {e}")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
        
        return results
    
    def _get_vision_description(self, image_path: str) -> str:
        """Get AI description of image"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. Include any text you can see, objects, people, scenes, and any other relevant information that would be useful for document search and knowledge extraction."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Vision API failed: {e}")
            return ""
    
    def create_document_from_image(self, image_path: str, use_vision_model: bool = True) -> Document:
        """Create Agno Document from image"""
        results = self.process_image(image_path, use_vision_model)
        
        # Combine OCR text and vision description
        content_parts = []
        if results["ocr_text"].strip():
            content_parts.append(f"**Extracted Text (OCR):**\n{results['ocr_text'].strip()}")
        
        if results["vision_description"].strip():
            content_parts.append(f"**Image Description:**\n{results['vision_description'].strip()}")
        
        content = "\n\n".join(content_parts) if content_parts else "No text or meaningful content extracted from image."
        
        # Create document with metadata
        doc = Document(
            content=content,
            meta_data={
                "type": "image",
                "source_file": os.path.basename(image_path),
                "file_path": image_path,
                "modality": "image",
                "processing_method": "ocr_and_vision",
                "ocr_text": results["ocr_text"],
                "vision_description": results["vision_description"],
                **results["metadata"]
            }
        )
        
        return doc


class AudioProcessor:
    """Process audio files using transcription"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file with transcription"""
        results = {
            "transcription": "",
            "metadata": {
                "file_path": audio_path,
                "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
                "duration": None,
                "sample_rate": None,
                "processed_at": datetime.now().isoformat()
            }
        }
        
        try:
            # Get audio metadata
            if LIBROSA_AVAILABLE:
                try:
                    y, sr = librosa.load(audio_path, sr=None)
                    results["metadata"]["duration"] = len(y) / sr
                    results["metadata"]["sample_rate"] = sr
                except Exception as e:
                    logger.warning(f"Audio metadata extraction failed: {e}")
            
            # Transcription
            if self.openai_client:
                try:
                    results["transcription"] = self._transcribe_audio(audio_path)
                except Exception as e:
                    logger.warning(f"Transcription failed: {e}")
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
        
        return results
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using OpenAI Whisper"""
        try:
            # Check file size (OpenAI has 25MB limit)
            file_size = os.path.getsize(audio_path)
            if file_size > 25 * 1024 * 1024:  # 25MB
                logger.warning("Audio file too large for OpenAI API, attempting to compress...")
                audio_path = self._compress_audio(audio_path)
            
            with open(audio_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription API failed: {e}")
            return ""
    
    def _compress_audio(self, audio_path: str) -> str:
        """Compress audio file if it's too large"""
        try:
            if not LIBROSA_AVAILABLE:
                return audio_path
            
            # Load and resample to lower quality
            y, sr = librosa.load(audio_path, sr=16000)  # 16kHz is sufficient for speech
            
            # Create temporary compressed file
            compressed_path = audio_path.replace(Path(audio_path).suffix, "_compressed.wav")
            sf.write(compressed_path, y, sr)
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"Audio compression failed: {e}")
            return audio_path
    
    def create_document_from_audio(self, audio_path: str) -> Document:
        """Create Agno Document from audio"""
        results = self.process_audio(audio_path)
        
        content = f"**Audio Transcription:**\n{results['transcription']}" if results["transcription"] else "No transcription available for this audio file."
        
        # Create document with metadata
        doc = Document(
            content=content,
            meta_data={
                "type": "audio",
                "source_file": os.path.basename(audio_path),
                "file_path": audio_path,
                "modality": "audio",
                "processing_method": "whisper_transcription",
                "transcription": results["transcription"],
                **results["metadata"]
            }
        )
        
        return doc


class VideoProcessor:
    """Process video files by extracting frames and audio"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        self.image_processor = ImageProcessor(openai_api_key)
        self.audio_processor = AudioProcessor(openai_api_key)
    
    def process_video(self, video_path: str, extract_frames: int = 5) -> Dict[str, Any]:
        """Process video by extracting frames and audio"""
        results = {
            "frames_analysis": [],
            "audio_transcription": "",
            "metadata": {
                "file_path": video_path,
                "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                "duration": None,
                "fps": None,
                "resolution": None,
                "processed_at": datetime.now().isoformat()
            }
        }
        
        try:
            # Extract frames
            if CV2_AVAILABLE:
                try:
                    frame_paths = self._extract_frames(video_path, extract_frames)
                    
                    # Process each frame
                    for frame_path in frame_paths:
                        frame_analysis = self.image_processor.process_image(frame_path, use_vision_model=True)
                        results["frames_analysis"].append({
                            "frame_path": frame_path,
                            "timestamp": self._get_frame_timestamp(frame_path),
                            **frame_analysis
                        })
                        
                        # Clean up frame file
                        try:
                            os.remove(frame_path)
                        except:
                            pass
                            
                except Exception as e:
                    logger.warning(f"Frame extraction failed: {e}")
            
            # Extract and process audio
            try:
                audio_path = self._extract_audio(video_path)
                if audio_path:
                    audio_results = self.audio_processor.process_audio(audio_path)
                    results["audio_transcription"] = audio_results["transcription"]
                    
                    # Clean up audio file
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")
        
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
        
        return results
    
    def _extract_frames(self, video_path: str, num_frames: int = 5) -> List[str]:
        """Extract frames from video"""
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to extract
            if total_frames > 0:
                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Save frame to temporary file
                        frame_path = os.path.join(
                            tempfile.gettempdir(),
                            f"frame_{i}_{frame_idx}.jpg"
                        )
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frame_paths
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video"""
        try:
            audio_path = os.path.join(
                tempfile.gettempdir(),
                f"extracted_audio_{datetime.now().timestamp()}.wav"
            )
            
            # Use cv2 to extract audio (basic approach)
            # For production, consider using FFmpeg
            cap = cv2.VideoCapture(video_path)
            # Note: cv2 doesn't directly extract audio
            # This is a placeholder - in production use FFmpeg or similar
            cap.release()
            
            # Return None since cv2 can't extract audio
            # In production, use: ffmpeg -i video.mp4 -q:a 0 -map a audio.wav
            return None
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _get_frame_timestamp(self, frame_path: str) -> float:
        """Get timestamp for frame from filename"""
        try:
            filename = os.path.basename(frame_path)
            frame_idx = int(filename.split('_')[-1].split('.')[0])
            # This is approximate - in production, calculate based on FPS
            return frame_idx * 0.1  # Assuming 10 FPS for simplicity
        except:
            return 0.0
    
    def create_document_from_video(self, video_path: str, extract_frames: int = 5) -> Document:
        """Create Agno Document from video"""
        results = self.process_video(video_path, extract_frames)
        
        # Combine all extracted content
        content_parts = []
        
        # Add audio transcription
        if results["audio_transcription"]:
            content_parts.append(f"**Audio Transcription:**\n{results['audio_transcription']}")
        
        # Add frame analysis
        if results["frames_analysis"]:
            content_parts.append("**Frame Analysis:**")
            for i, frame in enumerate(results["frames_analysis"]):
                content_parts.append(f"\n**Frame {i+1} (t={frame['timestamp']:.1f}s):**")
                if frame.get("vision_description"):
                    content_parts.append(f"Description: {frame['vision_description']}")
                if frame.get("ocr_text"):
                    content_parts.append(f"Text: {frame['ocr_text']}")
        
        content = "\n".join(content_parts) if content_parts else "No meaningful content extracted from video."
        
        # Create document with metadata
        doc = Document(
            content=content,
            meta_data={
                "type": "video",
                "source_file": os.path.basename(video_path),
                "file_path": video_path,
                "modality": "video",
                "processing_method": "frame_extraction_and_audio_transcription",
                "frames_count": len(results["frames_analysis"]),
                "has_audio": bool(results["audio_transcription"]),
                **results["metadata"]
            }
        )
        
        return doc


class MultimodalDocumentLoader:
    """Unified loader for all document types including multimodal"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.image_processor = ImageProcessor(openai_api_key)
        self.audio_processor = AudioProcessor(openai_api_key)
        self.video_processor = VideoProcessor(openai_api_key)
    
    def load_document(self, file_path: str) -> Optional[Document]:
        """Load document based on file type"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Image files
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                return self.image_processor.create_document_from_image(file_path)
            
            # Audio files
            elif file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                return self.audio_processor.create_document_from_audio(file_path)
            
            # Video files
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                return self.video_processor.create_document_from_video(file_path)
            
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    def get_supported_extensions(self) -> Dict[str, List[str]]:
        """Get supported file extensions by type"""
        return {
            "image": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
            "audio": ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
            "video": ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
            "text": ['.pdf', '.txt', '.csv', '.md']
        }
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available"""
        return {
            "PIL (images)": PIL_AVAILABLE,
            "OpenAI (vision/audio)": OPENAI_AVAILABLE,
            "librosa (audio)": LIBROSA_AVAILABLE,
            "cv2 (video)": CV2_AVAILABLE,
            "pytesseract (OCR)": PIL_AVAILABLE  # Assuming if PIL is available, tesseract might be too
        }


def get_missing_dependencies() -> List[str]:
    """Get list of missing dependencies for multimodal support"""
    missing = []
    
    if not PIL_AVAILABLE:
        missing.append("Pillow pytesseract")
    if not OPENAI_AVAILABLE:
        missing.append("openai")
    if not LIBROSA_AVAILABLE:
        missing.append("librosa soundfile")
    if not CV2_AVAILABLE:
        missing.append("opencv-python")
    
    return missing


if __name__ == "__main__":
    # Test multimodal processing
    loader = MultimodalDocumentLoader()
    
    print("Multimodal Document Loader Test")
    print("=" * 40)
    
    # Check dependencies
    deps = loader.check_dependencies()
    print("Dependencies:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    # Show supported extensions
    print("\nSupported Extensions:")
    for doc_type, extensions in loader.get_supported_extensions().items():
        print(f"  {doc_type}: {', '.join(extensions)}")
    
    # Show missing dependencies
    missing = get_missing_dependencies()
    if missing:
        print(f"\nTo install missing dependencies:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n✅ All dependencies available!")