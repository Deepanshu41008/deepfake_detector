"""
File utility functions for the Deepfake Detection API
"""

import os
import shutil
import platform
import magic
from typing import Dict, Any
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.exceptions import UnsupportedFileTypeError


def validate_file_type(file_extension: str) -> str:
    """
    Validate file type and return the media type
    
    Args:
        file_extension: File extension (e.g., '.mp4', '.jpg')
        
    Returns:
        Media type: 'video', 'image', or 'audio'
        
    Raises:
        UnsupportedFileTypeError: If file type is not supported
    """
    
    file_extension = file_extension.lower()
    
    if file_extension in settings.ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    elif file_extension in settings.ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    elif file_extension in settings.ALLOWED_AUDIO_EXTENSIONS:
        return "audio"
    else:
        raise UnsupportedFileTypeError(file_extension)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = os.stat(file_path)
    
    # Get MIME type
    try:
        mime_type = magic.from_file(file_path, mime=True)
    except:
        mime_type = "unknown"
    
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Determine media type
    try:
        media_type = validate_file_type(file_extension)
    except UnsupportedFileTypeError:
        media_type = "unknown"
    
    return {
        "size": stat.st_size,
        "created_time": stat.st_ctime,
        "modified_time": stat.st_mtime,
        "mime_type": mime_type,
        "extension": file_extension,
        "type": media_type,
        "readable": os.access(file_path, os.R_OK),
        "writable": os.access(file_path, os.W_OK)
    }


def create_upload_directory() -> str:
    """
    Create upload directory if it doesn't exist
    
    Returns:
        Path to the upload directory
    """
    
    upload_dir = settings.UPLOAD_DIR
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
    
    return upload_dir


def cleanup_old_files(max_age_hours: int = 24):
    """
    Clean up old uploaded files
    
    Args:
        max_age_hours: Maximum age of files in hours before deletion
    """
    
    upload_dir = settings.UPLOAD_DIR
    
    if not os.path.exists(upload_dir):
        return
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    
    deleted_count = 0
    
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
    
    return deleted_count


def get_directory_size(directory: str) -> int:
    """
    Get total size of all files in a directory
    
    Args:
        directory: Path to the directory
        
    Returns:
        Total size in bytes
    """
    
    total_size = 0
    
    if not os.path.exists(directory):
        return 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                pass
    
    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size_value = float(size_bytes)
    
    while size_value >= 1024 and i < len(size_names) - 1:
        size_value /= 1024.0
        i += 1
    
    return f"{size_value:.1f} {size_names[i]}"


def is_file_corrupted(file_path: str) -> bool:
    """
    Check if a file is corrupted (basic check)
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file appears corrupted, False otherwise
    """
    
    try:
        # Check if file exists and has size > 0
        if not os.path.exists(file_path):
            return True
        
        if os.path.getsize(file_path) == 0:
            return True
        
        # Try to read the first few bytes
        with open(file_path, 'rb') as f:
            f.read(1024)
        
        return False
        
    except Exception:
        return True


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of the file hash
    """
    
    import hashlib
    
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def get_disk_usage(path: str) -> Dict[str, int]:
    """
    Get disk usage information for a given path (cross-platform)
    
    Args:
        path: Path to check disk usage for
        
    Returns:
        Dictionary with 'total', 'used', and 'free' space in bytes
    """
    try:
        # Use shutil.disk_usage which is available on all platforms (Python 3.3+)
        total, used, free = shutil.disk_usage(path)
        
        return {
            "total": total,
            "used": used, 
            "free": free
        }
    except Exception as e:
        logger.warning(f"Failed to get disk usage for {path}: {e}")
        return {
            "total": 0,
            "used": 0,
            "free": 0
        }