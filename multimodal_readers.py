"""
Agno Document Readers for Multimodal Content
Custom readers that integrate with Agno's document system
"""

import os
import tempfile
from typing import List, Union, BinaryIO
from pathlib import Path
import logging

from agno.document import Document
from agno.document.reader.base import Reader

# Import our multimodal processors
try:
    from multimodal_processors import (
        ImageProcessor, AudioProcessor, VideoProcessor, 
        MultimodalDocumentLoader, get_missing_dependencies
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageReader(Reader):
    """Agno Document Reader for Images (OCR + Vision)"""
    
    def __init__(self, use_vision_model: bool = True, openai_api_key: str = None):
        self.use_vision_model = use_vision_model
        self.processor = ImageProcessor(openai_api_key) if MULTIMODAL_AVAILABLE else None
        
        if not MULTIMODAL_AVAILABLE:
            logger.warning("Multimodal processors not available. Install dependencies.")
    
    def read(self, image: Union[str, Path, BinaryIO]) -> List[Document]:
        """Read image and extract text/description"""
        if not self.processor:
            return [Document(
                content="Multimodal processing not available. Install required dependencies.",
                meta_data={"type": "error", "modality": "image"}
            )]
        
        # Handle different input types
        if isinstance(image, (str, Path)):
            file_path = str(image)
        elif hasattr(image, 'name'):
            file_path = image.name
        else:
            # Handle uploaded file or BinaryIO
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                if hasattr(image, 'read'):
                    tmp_file.write(image.read())
                else:
                    tmp_file.write(image)
                file_path = tmp_file.name
        
        try:
            # Process the image
            doc = self.processor.create_document_from_image(
                file_path, 
                use_vision_model=self.use_vision_model
            )
            
            # Add reader metadata
            if doc.meta_data:
                doc.meta_data["reader"] = "ImageReader"
                doc.meta_data["reader_version"] = "1.0"
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return [Document(
                content=f"Error processing image: {str(e)}",
                meta_data={"type": "error", "modality": "image", "error": str(e)}
            )]
        finally:
            # Clean up temporary file if created
            if hasattr(image, 'read') and file_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(file_path)
                except:
                    pass


class AudioReader(Reader):
    """Agno Document Reader for Audio (Transcription)"""
    
    def __init__(self, openai_api_key: str = None):
        self.processor = AudioProcessor(openai_api_key) if MULTIMODAL_AVAILABLE else None
        
        if not MULTIMODAL_AVAILABLE:
            logger.warning("Multimodal processors not available. Install dependencies.")
    
    def read(self, audio: Union[str, Path, BinaryIO]) -> List[Document]:
        """Read audio and extract transcription"""
        if not self.processor:
            return [Document(
                content="Multimodal processing not available. Install required dependencies.",
                meta_data={"type": "error", "modality": "audio"}
            )]
        
        # Handle different input types
        if isinstance(audio, (str, Path)):
            file_path = str(audio)
        elif hasattr(audio, 'name'):
            file_path = audio.name
        else:
            # Handle uploaded file or BinaryIO
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                if hasattr(audio, 'read'):
                    tmp_file.write(audio.read())
                else:
                    tmp_file.write(audio)
                file_path = tmp_file.name
        
        try:
            # Process the audio
            doc = self.processor.create_document_from_audio(file_path)
            
            # Add reader metadata
            if doc.meta_data:
                doc.meta_data["reader"] = "AudioReader"
                doc.meta_data["reader_version"] = "1.0"
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            return [Document(
                content=f"Error processing audio: {str(e)}",
                meta_data={"type": "error", "modality": "audio", "error": str(e)}
            )]
        finally:
            # Clean up temporary file if created
            if hasattr(audio, 'read') and file_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(file_path)
                except:
                    pass


class VideoReader(Reader):
    """Agno Document Reader for Video (Frame extraction + Audio transcription)"""
    
    def __init__(self, extract_frames: int = 5, openai_api_key: str = None):
        self.extract_frames = extract_frames
        self.processor = VideoProcessor(openai_api_key) if MULTIMODAL_AVAILABLE else None
        
        if not MULTIMODAL_AVAILABLE:
            logger.warning("Multimodal processors not available. Install dependencies.")
    
    def read(self, video: Union[str, Path, BinaryIO]) -> List[Document]:
        """Read video and extract frames + audio"""
        if not self.processor:
            return [Document(
                content="Multimodal processing not available. Install required dependencies.",
                meta_data={"type": "error", "modality": "video"}
            )]
        
        # Handle different input types
        if isinstance(video, (str, Path)):
            file_path = str(video)
        elif hasattr(video, 'name'):
            file_path = video.name
        else:
            # Handle uploaded file or BinaryIO
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                if hasattr(video, 'read'):
                    tmp_file.write(video.read())
                else:
                    tmp_file.write(video)
                file_path = tmp_file.name
        
        try:
            # Process the video
            doc = self.processor.create_document_from_video(
                file_path, 
                extract_frames=self.extract_frames
            )
            
            # Add reader metadata
            if doc.meta_data:
                doc.meta_data["reader"] = "VideoReader"
                doc.meta_data["reader_version"] = "1.0"
                doc.meta_data["frames_extracted"] = self.extract_frames
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Error reading video: {e}")
            return [Document(
                content=f"Error processing video: {str(e)}",
                meta_data={"type": "error", "modality": "video", "error": str(e)}
            )]
        finally:
            # Clean up temporary file if created
            if hasattr(video, 'read') and file_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(file_path)
                except:
                    pass


class MultimodalReader(Reader):
    """Universal Agno Document Reader for all file types"""
    
    def __init__(self, 
                 use_vision_model: bool = True, 
                 extract_video_frames: int = 5,
                 openai_api_key: str = None):
        self.use_vision_model = use_vision_model
        self.extract_video_frames = extract_video_frames
        self.openai_api_key = openai_api_key
        
        # Initialize specialized readers
        self.image_reader = ImageReader(use_vision_model, openai_api_key)
        self.audio_reader = AudioReader(openai_api_key)
        self.video_reader = VideoReader(extract_video_frames, openai_api_key)
        
        # Import standard readers
        try:
            from agno.document.reader.pdf_reader import PDFReader
            from agno.document.reader.text_reader import TextReader
            from agno.document.reader.csv_reader import CSVReader
            
            self.pdf_reader = PDFReader()
            self.text_reader = TextReader()
            self.csv_reader = CSVReader()
        except ImportError as e:
            logger.warning(f"Some standard readers not available: {e}")
    
    def read(self, source: Union[str, Path, BinaryIO]) -> List[Document]:
        """Read any supported file type"""
        
        # Determine file type
        if isinstance(source, (str, Path)):
            file_path = str(source)
            file_ext = Path(file_path).suffix.lower()
        elif hasattr(source, 'name'):
            file_path = source.name
            file_ext = Path(file_path).suffix.lower()
        else:
            # Can't determine type without filename, try image first
            return self.image_reader.read(source)
        
        try:
            # Route to appropriate reader based on extension
            
            # Image files
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                return self.image_reader.read(source)
            
            # Audio files
            elif file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                return self.audio_reader.read(source)
            
            # Video files
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                return self.video_reader.read(source)
            
            # Text files
            elif file_ext == '.pdf':
                return self.pdf_reader.read(source)
            elif file_ext == '.csv':
                return self.csv_reader.read(source)
            elif file_ext in ['.txt', '.md']:
                return self.text_reader.read(source)
            
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return [Document(
                    content=f"Unsupported file type: {file_ext}",
                    meta_data={
                        "type": "error", 
                        "error": f"Unsupported file extension: {file_ext}",
                        "file_path": file_path if isinstance(source, (str, Path)) else "unknown"
                    }
                )]
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return [Document(
                content=f"Error reading file: {str(e)}",
                meta_data={
                    "type": "error", 
                    "error": str(e),
                    "file_path": file_path if isinstance(source, (str, Path)) else "unknown"
                }
            )]
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        extensions = []
        
        # Multimodal extensions
        extensions.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'])  # Images
        extensions.extend(['.mp3', '.wav', '.m4a', '.flac', '.ogg'])  # Audio
        extensions.extend(['.mp4', '.avi', '.mov', '.mkv', '.wmv'])  # Video
        
        # Text extensions
        extensions.extend(['.pdf', '.txt', '.csv', '.md'])
        
        return extensions
    
    def check_capabilities(self) -> dict:
        """Check what capabilities are available"""
        return {
            "multimodal_available": MULTIMODAL_AVAILABLE,
            "missing_dependencies": get_missing_dependencies() if MULTIMODAL_AVAILABLE else ["multimodal_processors"],
            "supported_extensions": self.get_supported_extensions(),
            "vision_model_enabled": self.use_vision_model,
            "video_frame_extraction": self.extract_video_frames
        }


def get_reader_for_file(file_path: str, **kwargs) -> Reader:
    """Get appropriate reader for a specific file"""
    file_ext = Path(file_path).suffix.lower()
    
    # Image files
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
        return ImageReader(**kwargs)
    
    # Audio files
    elif file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
        return AudioReader(**kwargs)
    
    # Video files
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        return VideoReader(**kwargs)
    
    # Fallback to multimodal reader
    else:
        return MultimodalReader(**kwargs)


# Convenience functions for easy integration
def read_image(image_path: str, use_vision: bool = True, openai_api_key: str = None) -> List[Document]:
    """Quick function to read an image"""
    reader = ImageReader(use_vision_model=use_vision, openai_api_key=openai_api_key)
    return reader.read(image_path)


def read_audio(audio_path: str, openai_api_key: str = None) -> List[Document]:
    """Quick function to read audio"""
    reader = AudioReader(openai_api_key=openai_api_key)
    return reader.read(audio_path)


def read_video(video_path: str, frames: int = 5, openai_api_key: str = None) -> List[Document]:
    """Quick function to read video"""
    reader = VideoReader(extract_frames=frames, openai_api_key=openai_api_key)
    return reader.read(video_path)


if __name__ == "__main__":
    # Test the multimodal readers
    print("Multimodal Agno Readers Test")
    print("=" * 40)
    
    # Test multimodal reader capabilities
    reader = MultimodalReader()
    capabilities = reader.check_capabilities()
    
    print("Capabilities:")
    print(f"  Multimodal Available: {capabilities['multimodal_available']}")
    print(f"  Vision Model: {capabilities['vision_model_enabled']}")
    print(f"  Video Frames: {capabilities['video_frame_extraction']}")
    
    if capabilities['missing_dependencies']:
        print(f"  Missing: {', '.join(capabilities['missing_dependencies'])}")
    
    print(f"\nSupported Extensions: {', '.join(capabilities['supported_extensions'])}")
    
    # Example usage
    print("\nExample Usage:")
    print("from multimodal_readers import MultimodalReader")
    print("reader = MultimodalReader()")
    print("docs = reader.read('image.jpg')  # Works with any supported file")
    print("docs = reader.read('audio.mp3')")
    print("docs = reader.read('video.mp4')")