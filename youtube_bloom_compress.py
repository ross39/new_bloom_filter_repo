import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import yt_dlp
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union, Any

# Import the existing compressor
from video_delta_compress import VideoDeltaCompressor

# Add compression options for color data
import zlib
from io import BytesIO


class YouTubeBloomCompressor:
    """
    Class to download, compress, and analyze YouTube videos using Bloom filter compression.
    """
    
    def __init__(self, temp_dir: str = "temp_youtube_downloads"):
        """
        Initialize the YouTube Bloom compressor.
        
        Args:
            temp_dir: Directory to store temporary downloaded files
        """
        self.temp_dir = temp_dir
        self.video_compressor = VideoDeltaCompressor()
        os.makedirs(temp_dir, exist_ok=True)
    
    def download_video(self, url: str, resolution: str = "360p", 
                      output_path: Optional[str] = None) -> str:
        """
        Download a YouTube video using yt-dlp.
        
        Args:
            url: YouTube video URL
            resolution: Video resolution to download
            output_path: Optional path to save the video
            
        Returns:
            Path to the downloaded video file
        """
        print(f"Downloading YouTube video: {url}")
        try:
            # Create a timestamp-based filename
            timestamp = int(time.time())
            filename = f"video_{timestamp}.mp4"
            
            # Determine output path
            if output_path:
                video_path = output_path
            else:
                video_path = os.path.join(self.temp_dir, filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # Map resolution string to actual format
            if resolution == "360p":
                format_str = "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/best"
            elif resolution == "480p":
                format_str = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best"
            elif resolution == "720p":
                format_str = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best"
            else:
                format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
            
            # Set up yt-dlp options
            ydl_opts = {
                'format': format_str,
                'outtmpl': video_path,
                'quiet': False,
                'no_warnings': False,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                'prefer_ffmpeg': True,
                'noplaylist': True,
            }
            
            # Download video using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Verify the file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Downloaded file not found at {video_path}")
                
            print(f"Video downloaded to: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            print("Check if the YouTube URL is valid and accessible.")
            raise
    
    def preprocess_video(self, video_path: str, fps: int = 30, 
                        max_frames: int = 0, scale: float = 1.0,
                        binarize: bool = True, threshold: int = 127) -> List[np.ndarray]:
        """
        Preprocess a video for compression.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract
            max_frames: Maximum number of frames to process (0 for all frames)
            scale: Scale factor for resizing frames
            binarize: Whether to binarize the frames
            threshold: Threshold for binarization (0-255)
            
        Returns:
            List of preprocessed video frames
        """
        print(f"Preprocessing video: {video_path}")
        print(f"Parameters: fps={fps}, max_frames={'all' if max_frames <= 0 else max_frames}, scale={scale}, "
              f"binarize={binarize}, threshold={threshold}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video information
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Original video: {original_width}x{original_height}, {original_fps} fps, {total_frames} frames")
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            print(f"Resized dimensions: {new_width}x{new_height}")
            
            # Calculate frame interval to achieve desired fps
            frame_interval = max(1, int(original_fps / fps))
            
            # If max_frames is 0 or negative, process all frames
            frames_to_process = total_frames if max_frames <= 0 else max_frames
            
            print(f"Processing up to {frames_to_process} frames at {fps} fps")
            
            # Process frames
            frames = []
            frame_count = 0
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames > 0 and frame_count >= max_frames):
                    break
                
                # Process every nth frame according to the desired fps
                if frame_idx % frame_interval == 0:
                    # Resize frame
                    if scale != 1.0:
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Convert to grayscale
                    if len(frame.shape) > 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Binarize if requested
                    if binarize:
                        _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
                    
                    frames.append(frame)
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames...")
                
                frame_idx += 1
            
            cap.release()
            
            print(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"Error preprocessing video: {e}")
            raise
    
    def preprocess_video_with_color(self, video_path: str, fps: int = 30, 
                                 max_frames: int = 0, scale: float = 1.0,
                                 binarize: bool = True, threshold: int = 127) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Preprocess a video while preserving color information for later restoration.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract
            max_frames: Maximum number of frames to process (0 for all frames)
            scale: Scale factor for resizing frames
            binarize: Whether to binarize grayscale frames
            threshold: Threshold for binarization (0-255)
            
        Returns:
            Tuple of (grayscale_frames, color_frames)
        """
        print(f"Preprocessing video with color preservation: {video_path}")
        print(f"Parameters: fps={fps}, max_frames={'all' if max_frames <= 0 else max_frames}, scale={scale}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video information
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Original video: {original_width}x{original_height}, {original_fps} fps, {total_frames} frames")
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            print(f"Resized dimensions: {new_width}x{new_height}")
            
            # Calculate frame interval to achieve desired fps
            frame_interval = max(1, int(original_fps / fps))
            
            # If max_frames is 0 or negative, process all frames
            frames_to_process = total_frames if max_frames <= 0 else max_frames
            
            print(f"Processing up to {frames_to_process} frames at {fps} fps")
            
            # Process frames
            grayscale_frames = []
            color_frames = []
            frame_count = 0
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames > 0 and frame_count >= max_frames):
                    break
                
                # Process every nth frame according to the desired fps
                if frame_idx % frame_interval == 0:
                    # Resize frame
                    if scale != 1.0:
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Store the color frame (convert from BGR to RGB)
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    color_frames.append(color_frame)
                    
                    # Convert to grayscale for compression
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Binarize if requested
                    if binarize:
                        _, gray_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
                    
                    grayscale_frames.append(gray_frame)
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames...")
                
                frame_idx += 1
            
            cap.release()
            
            print(f"Extracted {len(grayscale_frames)} frames from video (with color preservation)")
            return grayscale_frames, color_frames
            
        except Exception as e:
            print(f"Error preprocessing video with color: {e}")
            raise

    def extract_color_information(self, color_frames: List[np.ndarray], 
                               grayscale_frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract color difference information between grayscale and color frames.
        
        Args:
            color_frames: List of original color frames
            grayscale_frames: List of grayscale frames used for compression
            
        Returns:
            List of tuples containing chrominance channels (U, V) for each frame
        """
        print("Extracting color information...")
        
        color_data = []
        
        for i, (color_frame, gray_frame) in enumerate(zip(color_frames, grayscale_frames)):
            # Convert grayscale to YUV equivalent (Y channel = grayscale, U/V = 0)
            gray_yuv = np.zeros((*gray_frame.shape, 3), dtype=np.uint8)
            gray_yuv[:, :, 0] = gray_frame  # Y channel = grayscale
            
            # Convert color frame to YUV
            # OpenCV uses BGR format, so convert RGB->BGR first
            bgr_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            yuv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
            
            # Extract just the chrominance channels (U, V)
            u_channel = yuv_frame[:, :, 1]
            v_channel = yuv_frame[:, :, 2]
            
            color_data.append((u_channel, v_channel))
            
            if i % 100 == 0 and i > 0:
                print(f"Processed color data for {i} frames...")
        
        return color_data

    def compress_color_information(self, color_data: List[Tuple[np.ndarray, np.ndarray]], 
                                keyframe_indices: List[int],
                                chroma_downscale: float = 0.5) -> List[bytes]:
        """
        Compress color information efficiently.
        
        Args:
            color_data: List of (U, V) channel pairs for each frame
            keyframe_indices: Indices of keyframes
            chroma_downscale: Downscaling factor for chrominance in non-keyframes
            
        Returns:
            List of compressed color data for each frame
        """
        print(f"Compressing color information (chroma downscale: {chroma_downscale})...")
        
        compressed_color = []
        
        for i, (u_channel, v_channel) in enumerate(color_data):
            is_keyframe = i in keyframe_indices
            
            if is_keyframe:
                # Store full resolution color for keyframes
                compressed = self._compress_color_channels(u_channel, v_channel, is_keyframe=True)
            else:
                # For non-keyframes, downsample chrominance
                if chroma_downscale < 1.0:
                    h, w = u_channel.shape
                    new_w, new_h = int(w * chroma_downscale), int(h * chroma_downscale)
                    u_downsampled = cv2.resize(u_channel, (new_w, new_h))
                    v_downsampled = cv2.resize(v_channel, (new_w, new_h))
                    compressed = self._compress_color_channels(u_downsampled, v_downsampled, is_keyframe=False)
                else:
                    compressed = self._compress_color_channels(u_channel, v_channel, is_keyframe=False)
            
            compressed_color.append(compressed)
            
            if i % 100 == 0 and i > 0:
                print(f"Compressed color data for {i} frames...")
        
        return compressed_color

    def _compress_color_channels(self, u_channel: np.ndarray, v_channel: np.ndarray, 
                              is_keyframe: bool = False) -> bytes:
        """
        Compress U and V channels using zlib.
        
        Args:
            u_channel: U channel of YUV color space
            v_channel: V channel of YUV color space
            is_keyframe: Whether this is a keyframe (affects compression level)
            
        Returns:
            Compressed binary data
        """
        # Store shape information
        shape = u_channel.shape
        
        # Flatten and combine channels
        u_flat = u_channel.flatten()
        v_flat = v_channel.flatten()
        
        # Serialize using BytesIO
        buffer = BytesIO()
        
        # Store shape information
        buffer.write(np.array(shape, dtype=np.int32).tobytes())
        
        # Store is_keyframe flag
        buffer.write(np.array([1 if is_keyframe else 0], dtype=np.uint8).tobytes())
        
        # Store pixel data
        buffer.write(u_flat.astype(np.uint8).tobytes())
        buffer.write(v_flat.astype(np.uint8).tobytes())
        
        # Get the bytes and compress with zlib
        raw_data = buffer.getvalue()
        
        # Use different compression levels for keyframes vs non-keyframes
        compression_level = 6 if is_keyframe else 9
        compressed_data = zlib.compress(raw_data, level=compression_level)
        
        return compressed_data

    def _decompress_color_channels(self, compressed_data: bytes) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Decompress color channels from compressed binary data.
        
        Args:
            compressed_data: Compressed binary data
            
        Returns:
            Tuple of (u_channel, v_channel, is_keyframe)
        """
        # Decompress the data
        raw_data = zlib.decompress(compressed_data)
        buffer = BytesIO(raw_data)
        
        # Read shape information (2 integers, 4 bytes each)
        shape_data = buffer.read(8)
        shape = tuple(np.frombuffer(shape_data, dtype=np.int32))
        
        # Read keyframe flag
        keyframe_data = buffer.read(1)
        is_keyframe = bool(np.frombuffer(keyframe_data, dtype=np.uint8)[0])
        
        # Calculate the size of each channel
        channel_size = shape[0] * shape[1]
        
        # Read U channel data
        u_data = buffer.read(channel_size)
        u_channel = np.frombuffer(u_data, dtype=np.uint8).reshape(shape)
        
        # Read V channel data
        v_data = buffer.read(channel_size)
        v_channel = np.frombuffer(v_data, dtype=np.uint8).reshape(shape)
        
        return u_channel, v_channel, is_keyframe

    def restore_color(self, grayscale_frames: List[np.ndarray], 
                   compressed_color: List[bytes]) -> List[np.ndarray]:
        """
        Restore full color to decompressed grayscale frames.
        
        Args:
            grayscale_frames: List of grayscale frames
            compressed_color: List of compressed color data
            
        Returns:
            List of restored color frames
        """
        print("Restoring color to grayscale frames...")
        
        color_frames = []
        
        for i, gray_frame in enumerate(grayscale_frames):
            # Decompress color channels
            u_channel, v_channel, is_keyframe = self._decompress_color_channels(compressed_color[i])
            
            # Check if we need to upsample the chrominance channels
            if u_channel.shape != gray_frame.shape:
                u_channel = cv2.resize(u_channel, (gray_frame.shape[1], gray_frame.shape[0]))
                v_channel = cv2.resize(v_channel, (gray_frame.shape[1], gray_frame.shape[0]))
            
            # Create YUV image
            yuv_image = np.zeros((gray_frame.shape[0], gray_frame.shape[1], 3), dtype=np.uint8)
            yuv_image[:, :, 0] = gray_frame       # Y channel from grayscale
            yuv_image[:, :, 1] = u_channel        # U channel
            yuv_image[:, :, 2] = v_channel        # V channel
            
            # Convert back to RGB
            bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            color_frames.append(rgb_image)
            
            if i % 100 == 0 and i > 0:
                print(f"Restored color for {i} frames...")
        
        return color_frames

    def _save_color_video(self, color_frames: List[np.ndarray], 
                      output_path: str, fps: int = 30) -> None:
        """
        Save color frames as a video.
        
        Args:
            color_frames: List of color frames
            output_path: Path to save the video
            fps: Frames per second
        """
        if not color_frames:
            print("No frames to save")
            return
        
        print(f"Saving color video to {output_path} at {fps} fps...")
        
        # Get frame dimensions
        height, width = color_frames[0].shape[:2]
        
        # Try using ffmpeg directly first (most reliable method)
        try:
            # Create a temp directory for frames
            temp_dir = f"{os.path.splitext(output_path)[0]}_temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            print(f"Saving frames to temporary directory: {temp_dir}")
            
            # Save frames as images
            for i, frame in enumerate(color_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if i % 100 == 0 and i > 0:
                    print(f"Saved {i} frames...")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Use ffmpeg to create the video
            ffmpeg_output_path = f"{os.path.splitext(output_path)[0]}_ffmpeg.mp4"
            
            # Build ffmpeg command with optimal parameters for playback compatibility
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-profile:v", "high",
                "-crf", "23",         # Quality setting - lower is better but larger
                "-pix_fmt", "yuv420p", # Required for broad compatibility
                "-movflags", "+faststart", # Enables progressive downloading
                "-vf", f"scale={width}:{height}:flags=lanczos",  # Ensure dimensions are even
                ffmpeg_output_path
            ]
            
            print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(ffmpeg_output_path) and os.path.getsize(ffmpeg_output_path) > 0:
                print(f"Successfully created video with ffmpeg: {ffmpeg_output_path}")
                
                # Clean up temporary frames
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary frames directory: {temp_dir}")
                except Exception as e:
                    print(f"Note: Could not clean up temp directory: {e}")
                
                return
            else:
                print("FFmpeg direct encoding failed, falling back to OpenCV")
                print(f"FFmpeg error: {result.stderr}")
                
                # Keep the frames for OpenCV fallback
                
        except Exception as e:
            print(f"Error using direct ffmpeg encoding: {e}")
            print("Falling back to OpenCV encoders")
        
        # Try different codec options in order of preference
        codec_options = [
            ('mp4v', '.mp4'),  # Default MP4 codec
            ('avc1', '.mp4'),  # H.264 codec
            ('XVID', '.avi'),  # XVID codec (widely compatible)
            ('MJPG', '.avi')   # Motion JPEG codec (very compatible)
        ]
        
        # Try each codec until one works
        for codec, extension in codec_options:
            try:
                # Adjust output path if needed
                if not output_path.lower().endswith(extension):
                    file_path, _ = os.path.splitext(output_path)
                    adjusted_path = file_path + extension
                else:
                    adjusted_path = output_path
                
                print(f"Attempting to save video with codec {codec} to {adjusted_path}")
                
                # Create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(adjusted_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print(f"Failed to open VideoWriter with codec {codec}")
                    continue
                
                # Write frames
                for i, frame in enumerate(color_frames):
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                    
                    if i % 100 == 0 and i > 0:
                        print(f"Written {i} frames to video...")
                
                # Release video writer
                out.release()
                
                # Verify the file was created and has size
                if os.path.exists(adjusted_path) and os.path.getsize(adjusted_path) > 0:
                    print(f"Color video saved successfully to {adjusted_path}")
                    
                    # Try to use ffmpeg to convert to a more compatible format
                    try:
                        improved_path = os.path.splitext(adjusted_path)[0] + "_ffmpeg.mp4"
                        print(f"Attempting to create more compatible version with ffmpeg: {improved_path}")
                        
                        # Build ffmpeg command
                        ffmpeg_cmd = [
                            "ffmpeg", "-y",
                            "-i", adjusted_path,
                            "-c:v", "libx264",
                            "-profile:v", "high",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-vf", "format=yuv420p",
                            "-movflags", "+faststart",
                            improved_path
                        ]
                        
                        # Run ffmpeg
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0 and os.path.exists(improved_path) and os.path.getsize(improved_path) > 0:
                            print(f"Created improved playback version with ffmpeg: {improved_path}")
                        else:
                            print("FFmpeg processing failed, using original OpenCV output")
                            print(f"FFmpeg error: {result.stderr}")
                    
                    except Exception as e:
                        print(f"FFmpeg processing failed: {e}")
                        print("Using original OpenCV output")
                    
                    return
                else:
                    print(f"Failed to create valid video file with codec {codec}")
            
            except Exception as e:
                print(f"Error saving video with codec {codec}: {e}")
        
        # If all codecs fail, try to save individual frames as a fallback
        fallback_dir = os.path.splitext(output_path)[0] + "_frames"
        print(f"All video codecs failed. Saving individual frames to {fallback_dir}")
        
        # Save all frames, not just samples
        os.makedirs(fallback_dir, exist_ok=True)
        for i, frame in enumerate(color_frames):
            frame_path = os.path.join(fallback_dir, f"frame_{i:06d}.png")
            Image.fromarray(frame).save(frame_path)
            if i % 100 == 0 and i > 0:
                print(f"Saved {i} frames as images...")
        
        # Create a simple HTML viewer for the frames
        self._create_html_frame_viewer(fallback_dir, len(color_frames), fps)

    def _create_html_frame_viewer(self, frames_dir: str, num_frames: int, fps: int = 30) -> None:
        """
        Create a simple HTML viewer for frame-by-frame playback.
        
        Args:
            frames_dir: Directory containing frame images
            num_frames: Number of frames
            fps: Frames per second for playback
        """
        html_path = os.path.join(frames_dir, "viewer.html")
        
        # Generate sample indices based on the actual saved frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        with open(html_path, 'w') as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>Frame Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; }}
        #frameViewer {{ max-width: 100%; max-height: 80vh; margin-bottom: 20px; }}
        .controls {{ margin: 20px 0; }}
        button {{ padding: 10px 15px; margin: 0 5px; }}
    </style>
</head>
<body>
    <h1>Frame Viewer</h1>
    <img id="frameViewer" src="" alt="Frame">
    <div class="controls">
        <button id="playBtn">Play</button>
        <button id="pauseBtn">Pause</button>
        <input type="range" id="frameSlider" min="0" max="{len(frame_files)-1}" value="0" style="width: 80%;">
    </div>
    <div>
        Frame: <span id="frameCounter">0</span> / {len(frame_files)-1}
    </div>
    <script>
        const frames = {str(frame_files).replace("'", '"')};
        const baseDir = '{frames_dir.replace(os.sep, "/")}';
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval;
        const fps = {fps};
        const frameInterval = 1000 / fps;
        
        const frameViewer = document.getElementById('frameViewer');
        const frameSlider = document.getElementById('frameSlider');
        const frameCounter = document.getElementById('frameCounter');
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        
        function updateFrame() {{
            if (currentFrame < frames.length) {{
                frameViewer.src = baseDir + '/' + frames[currentFrame];
                frameSlider.value = currentFrame;
                frameCounter.textContent = currentFrame;
            }}
        }}
        
        function play() {{
            if (!isPlaying) {{
                isPlaying = true;
                playInterval = setInterval(() => {{
                    currentFrame++;
                    if (currentFrame >= frames.length) {{
                        currentFrame = 0;
                    }}
                    updateFrame();
                }}, frameInterval);
            }}
        }}
        
        function pause() {{
            isPlaying = false;
            clearInterval(playInterval);
        }}
        
        playBtn.addEventListener('click', play);
        pauseBtn.addEventListener('click', pause);
        
        frameSlider.addEventListener('input', () => {{
            pause();
            currentFrame = parseInt(frameSlider.value);
            updateFrame();
        }});
        
        // Initial frame
        updateFrame();
    </script>
</body>
</html>''')
        
        print(f"Created HTML viewer at {html_path}")
        print(f"Open this file in a web browser to view the frames as a video")

    def compress_video(self, frames: List[np.ndarray], output_dir: str, 
                      experiment_name: str = "youtube_compression",
                      batch_size: int = 0) -> Tuple[List[bytes], Dict, Dict]:
        """
        Compress video frames using Bloom filter compression.
        
        Args:
            frames: List of video frames
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            batch_size: Number of frames to process at once (0 for all at once)
            
        Returns:
            Tuple of (compressed_frames, metadata, performance_metrics)
        """
        os.makedirs(output_dir, exist_ok=True)
        compressed_path = os.path.join(output_dir, f"{experiment_name}_compressed.bin")
        
        # If batch processing is disabled, process all frames at once
        if batch_size <= 0 or batch_size >= len(frames):
            print(f"Compressing {len(frames)} frames all at once...")
            start_time = time.time()
            
            # Compress the video frames
            compressed_frames, metadata = self.video_compressor.compress_video(
                frames, output_path=compressed_path)
            
            compress_time = time.time() - start_time
        else:
            # Process frames in batches for memory efficiency
            print(f"Compressing {len(frames)} frames in batches of {batch_size}...")
            start_time = time.time()
            
            # Initialize variables to accumulate results
            all_compressed_frames = []
            combined_metadata = {
                'frame_count': len(frames),
                'frame_shape': frames[0].shape if frames else None,
                'keyframes': [],
                'densities': [],
                'compression_ratios': []
            }
            
            # Process frames in batches
            for i in range(0, len(frames), batch_size):
                batch_end = min(i + batch_size, len(frames))
                print(f"Processing batch {i//batch_size + 1}: frames {i}-{batch_end-1}")
                
                # Extract batch of frames
                batch_frames = frames[i:batch_end]
                
                # Create temporary output path for this batch
                batch_path = os.path.join(output_dir, f"{experiment_name}_batch_{i//batch_size + 1}.bin")
                
                # Compress this batch
                batch_compressed, batch_metadata = self.video_compressor.compress_video(
                    batch_frames, output_path=batch_path)
                
                # Accumulate results
                all_compressed_frames.extend(batch_compressed)
                
                # Update metadata
                if 'keyframes' in batch_metadata:
                    # Adjust keyframe indices for global position
                    combined_metadata['keyframes'].extend([kf + i for kf in batch_metadata['keyframes']])
                
                if 'densities' in batch_metadata:
                    combined_metadata['densities'].extend(batch_metadata['densities'])
                
                if 'compression_ratios' in batch_metadata:
                    combined_metadata['compression_ratios'].extend(batch_metadata['compression_ratios'])
            
            # Save all compressed frames to the final output file
            compressed_frames = all_compressed_frames
            metadata = combined_metadata
            
            # TODO: Add code to properly save the combined result
            
            compress_time = time.time() - start_time
        
        # Calculate size metrics
        original_size = sum(frame.nbytes for frame in frames)
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = compressed_size / original_size
        
        # Save the first frame for reference
        if len(frames) > 0:
            first_frame_path = os.path.join(output_dir, f"{experiment_name}_first_frame.png")
            Image.fromarray(frames[0]).save(first_frame_path)
        
        # Performance metrics
        performance = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time": compress_time,
            "frames_per_second": len(frames) / compress_time if compress_time > 0 else 0,
            "bytes_per_frame": compressed_size / len(frames) if len(frames) > 0 else 0
        }
        
        # Generate compression report
        self._generate_compression_report(
            frames, metadata, performance, 
            os.path.join(output_dir, f"{experiment_name}_report.txt"))
        
        # Plot and save compression statistics
        self._plot_compression_stats(
            metadata, performance, 
            os.path.join(output_dir, f"{experiment_name}_stats.png"))
        
        return compressed_frames, metadata, performance
    
    def decompress_video(self, compressed_frames: List[bytes], metadata: Dict, 
                        output_dir: str, experiment_name: str = "youtube_compression") -> Tuple[List[np.ndarray], float]:
        """
        Decompress video frames.
        
        Args:
            compressed_frames: List of compressed frame data
            metadata: Metadata from compression
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            
        Returns:
            Tuple of (decompressed_frames, decompression_time)
        """
        os.makedirs(output_dir, exist_ok=True)
        decompressed_path = os.path.join(output_dir, f"{experiment_name}_decompressed.mp4")
        
        print(f"Decompressing {len(compressed_frames)} frames...")
        start_time = time.time()
        
        # Decompress the video frames
        decompressed_frames = self.video_compressor.decompress_video(
            compressed_frames, metadata, output_path=decompressed_path)
        
        decompress_time = time.time() - start_time
        
        print(f"Decompression completed in {decompress_time:.2f} seconds")
        print(f"Decompressed video saved to: {decompressed_path}")
        
        # Save some sample frames
        self._save_sample_frames(
            decompressed_frames, 
            os.path.join(output_dir, f"{experiment_name}_decompressed_frames"))
        
        return decompressed_frames, decompress_time
    
    def verify_lossless(self, original_frames: List[np.ndarray], 
                      decompressed_frames: List[np.ndarray], 
                      output_dir: str,
                      experiment_name: str = "youtube_compression") -> Tuple[bool, float]:
        """
        Verify if decompression is lossless by comparing original and decompressed frames.
        
        Args:
            original_frames: List of original video frames
            decompressed_frames: List of decompressed video frames
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            
        Returns:
            Tuple of (is_lossless, average_frame_difference)
        """
        if len(original_frames) != len(decompressed_frames):
            print(f"Warning: Frame count mismatch. Original: {len(original_frames)}, "
                  f"Decompressed: {len(decompressed_frames)}")
            return False, float('inf')
        
        differences = []
        is_lossless = True
        
        for i, (orig, decomp) in enumerate(zip(original_frames, decompressed_frames)):
            # Ensure shapes match
            if orig.shape != decomp.shape:
                print(f"Warning: Frame {i} shape mismatch. Original: {orig.shape}, "
                      f"Decompressed: {decomp.shape}")
                is_lossless = False
                continue
            
            # Calculate absolute difference
            diff = np.abs(orig.astype(np.float32) - decomp.astype(np.float32))
            mean_diff = np.mean(diff)
            differences.append(mean_diff)
            
            if mean_diff > 0:
                is_lossless = False
        
        avg_difference = np.mean(differences) if differences else 0
        
        print(f"Lossless verification results:")
        print(f"  Perfect reconstruction: {is_lossless}")
        print(f"  Average frame difference: {avg_difference:.6f}")
        
        # Save visualization of differences
        if not is_lossless and len(original_frames) > 0:
            self._visualize_differences(
                original_frames, decompressed_frames, differences,
                os.path.join(output_dir, f"{experiment_name}_differences.png"))
        
        return is_lossless, avg_difference
    
    def _generate_compression_report(self, frames: List[np.ndarray], 
                                   metadata: Dict, performance: Dict, 
                                   output_path: str) -> None:
        """Generate a detailed compression report."""
        with open(output_path, 'w') as f:
            f.write("=== Bloom Filter Video Compression Report ===\n\n")
            
            f.write("Video Statistics:\n")
            f.write(f"  Frame count: {len(frames)}\n")
            if len(frames) > 0:
                f.write(f"  Frame dimensions: {frames[0].shape[1]}x{frames[0].shape[0]}\n")
            f.write(f"  Original size: {performance['original_size']:,} bytes\n")
            f.write(f"  Compressed size: {performance['compressed_size']:,} bytes\n")
            f.write(f"  Compression ratio: {performance['compression_ratio']:.6f}\n")
            f.write(f"  Space savings: {(1 - performance['compression_ratio']) * 100:.2f}%\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Compression time: {performance['compression_time']:.2f} seconds\n")
            f.write(f"  Frames per second: {performance['frames_per_second']:.2f}\n")
            f.write(f"  Bytes per frame: {performance['bytes_per_frame']:.2f}\n\n")
            
            f.write("Bloom Filter Statistics:\n")
            if 'keyframes' in metadata:
                f.write(f"  Number of keyframes: {len(metadata['keyframes'])}\n")
            if 'densities' in metadata:
                avg_density = sum(metadata['densities']) / len(metadata['densities']) if metadata['densities'] else 0
                f.write(f"  Average frame difference density: {avg_density:.6f}\n")
            if 'compression_ratios' in metadata:
                avg_ratio = sum(metadata['compression_ratios']) / len(metadata['compression_ratios']) if metadata['compression_ratios'] else 0
                f.write(f"  Average per-frame compression ratio: {avg_ratio:.6f}\n")
        
        print(f"Compression report saved to: {output_path}")
    
    def _plot_compression_stats(self, metadata: Dict, performance: Dict, output_path: str) -> None:
        """Plot compression statistics and save to file."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Frame difference densities
        if 'densities' in metadata and metadata['densities']:
            axs[0, 0].plot(range(len(metadata['densities'])), metadata['densities'], 'o-')
            axs[0, 0].axhline(y=0.32453, color='r', linestyle='--', label='Threshold (0.32453)')
            axs[0, 0].set_xlabel('Frame')
            axs[0, 0].set_ylabel('Difference Density')
            axs[0, 0].set_title('Frame Difference Densities')
            axs[0, 0].grid(True)
            axs[0, 0].legend()
        
        # Plot 2: Per-frame compression ratios
        if 'compression_ratios' in metadata and metadata['compression_ratios']:
            axs[0, 1].plot(range(len(metadata['compression_ratios'])), metadata['compression_ratios'], 'o-')
            axs[0, 1].axhline(y=1.0, color='r', linestyle='--', label='No Compression')
            axs[0, 1].set_xlabel('Frame')
            axs[0, 1].set_ylabel('Compression Ratio')
            axs[0, 1].set_title('Per-Frame Compression Ratios')
            axs[0, 1].grid(True)
            axs[0, 1].legend()
        
        # Plot 3: Histogram of difference densities
        if 'densities' in metadata and metadata['densities']:
            axs[1, 0].hist(metadata['densities'], bins=20, alpha=0.7, color='green')
            axs[1, 0].axvline(x=0.32453, color='r', linestyle='--', label='Threshold (0.32453)')
            axs[1, 0].set_xlabel('Difference Density')
            axs[1, 0].set_ylabel('Frequency')
            axs[1, 0].set_title('Histogram of Frame Difference Densities')
            axs[1, 0].grid(True)
            axs[1, 0].legend()
        
        # Plot 4: Summary information
        axs[1, 1].axis('off')
        summary_text = (
            f"Compression Summary:\n\n"
            f"Original Size: {performance['original_size']:,} bytes\n"
            f"Compressed Size: {performance['compressed_size']:,} bytes\n"
            f"Overall Compression Ratio: {performance['compression_ratio']:.6f}\n"
            f"Space Savings: {(1 - performance['compression_ratio']) * 100:.2f}%\n\n"
            f"Compression Time: {performance['compression_time']:.2f} seconds\n"
            f"Frames Per Second: {performance['frames_per_second']:.2f}\n"
            f"Bytes Per Frame: {performance['bytes_per_frame']:.2f}"
        )
        axs[1, 1].text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Compression statistics plot saved to: {output_path}")
    
    def _save_sample_frames(self, frames: List[np.ndarray], output_dir: str) -> None:
        """Save sample frames from the video."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine indices of frames to save
        if len(frames) <= 5:
            indices = list(range(len(frames)))
        else:
            indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        
        # Save selected frames
        for i in indices:
            if i < len(frames):
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                Image.fromarray(frames[i]).save(frame_path)
        
        print(f"Sample frames saved to: {output_dir}")
    
    def _visualize_differences(self, original_frames: List[np.ndarray], 
                            decompressed_frames: List[np.ndarray],
                            differences: List[float],
                            output_path: str) -> None:
        """Visualize differences between original and decompressed frames."""
        # Find the frame with the highest difference
        if not differences:
            return
            
        max_diff_idx = differences.index(max(differences))
        mid_frame_idx = len(original_frames) // 2
        
        plt.figure(figsize=(15, 10))
        
        # Show original vs decompressed for middle frame
        plt.subplot(2, 3, 1)
        plt.imshow(original_frames[mid_frame_idx], cmap='gray')
        plt.title(f"Original Frame {mid_frame_idx}")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(decompressed_frames[mid_frame_idx], cmap='gray')
        plt.title(f"Decompressed Frame {mid_frame_idx}")
        plt.axis('off')
        
        # Show difference for middle frame
        diff = np.abs(original_frames[mid_frame_idx].astype(np.float32) - 
                    decompressed_frames[mid_frame_idx].astype(np.float32))
        plt.subplot(2, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title(f"Difference (MAE: {differences[mid_frame_idx]:.6f})")
        plt.axis('off')
        
        # Show original vs decompressed for max difference frame
        plt.subplot(2, 3, 4)
        plt.imshow(original_frames[max_diff_idx], cmap='gray')
        plt.title(f"Original Frame {max_diff_idx}")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(decompressed_frames[max_diff_idx], cmap='gray')
        plt.title(f"Decompressed Frame {max_diff_idx}")
        plt.axis('off')
        
        # Show difference for max difference frame
        diff = np.abs(original_frames[max_diff_idx].astype(np.float32) - 
                    decompressed_frames[max_diff_idx].astype(np.float32))
        plt.subplot(2, 3, 6)
        plt.imshow(diff, cmap='hot')
        plt.title(f"Max Difference (MAE: {differences[max_diff_idx]:.6f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Difference visualization saved to: {output_path}")

    def compress_video_with_color(self, grayscale_frames: List[np.ndarray], 
                             color_frames: List[np.ndarray],
                             output_dir: str, 
                             experiment_name: str = "youtube_color_compression",
                             batch_size: int = 0,
                             chroma_downscale: float = 0.5) -> Tuple[List[bytes], List[bytes], Dict, Dict]:
        """
        Compress video frames with color preservation.
        
        Args:
            grayscale_frames: List of grayscale video frames for Bloom filter compression
            color_frames: List of original color frames
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            batch_size: Number of frames to process at once (0 for all at once)
            chroma_downscale: Downscaling factor for chrominance in non-keyframes
            
        Returns:
            Tuple of (compressed_grayscale, compressed_color, metadata, performance_metrics)
        """
        os.makedirs(output_dir, exist_ok=True)
        compressed_gray_path = os.path.join(output_dir, f"{experiment_name}_compressed_gray.bin")
        compressed_color_path = os.path.join(output_dir, f"{experiment_name}_compressed_color.bin")
        
        # Step 1: Compress grayscale frames using Bloom filter compression
        print(f"Compressing {len(grayscale_frames)} grayscale frames...")
        start_time = time.time()
        
        # If batch processing is disabled, process all frames at once
        if batch_size <= 0 or batch_size >= len(grayscale_frames):
            compressed_gray, metadata = self.video_compressor.compress_video(
                grayscale_frames, output_path=compressed_gray_path)
        else:
            # Process in batches (reusing existing batch processing code)
            compressed_gray, metadata = self._batch_process_grayscale(
                grayscale_frames, output_dir, experiment_name, batch_size)
        
        gray_compress_time = time.time() - start_time
        print(f"Grayscale compression completed in {gray_compress_time:.2f} seconds")
        
        # Step 2: Extract and compress color information
        print("Processing color information...")
        color_start_time = time.time()
        
        # Extract color difference information
        color_data = self.extract_color_information(color_frames, grayscale_frames)
        
        # Compress color information
        keyframe_indices = metadata.get('keyframes', [])
        compressed_color = self.compress_color_information(
            color_data, keyframe_indices, chroma_downscale)
        
        # Save compressed color data
        with open(compressed_color_path, 'wb') as f:
            # Write number of frames
            f.write(len(compressed_color).to_bytes(4, byteorder='little'))
            
            # Write each compressed color frame
            for compressed in compressed_color:
                # Write size of current frame data
                f.write(len(compressed).to_bytes(4, byteorder='little'))
                # Write frame data
                f.write(compressed)
        
        color_compress_time = time.time() - color_start_time
        total_compress_time = time.time() - start_time
        
        # Calculate size metrics
        original_gray_size = sum(frame.nbytes for frame in grayscale_frames)
        original_color_size = sum(frame.nbytes for frame in color_frames)
        compressed_gray_size = os.path.getsize(compressed_gray_path)
        compressed_color_size = os.path.getsize(compressed_color_path)
        compressed_total_size = compressed_gray_size + compressed_color_size
        
        # Calculate compression ratios
        gray_ratio = compressed_gray_size / original_gray_size
        color_ratio = compressed_color_size / original_color_size
        total_ratio = compressed_total_size / original_color_size
        
        # Update metadata
        metadata['color_preservation'] = True
        metadata['chroma_downscale'] = chroma_downscale
        metadata['color_compressed_size'] = compressed_color_size
        metadata['grayscale_compressed_size'] = compressed_gray_size
        metadata['total_compressed_size'] = compressed_total_size
        
        # Performance metrics
        performance = {
            "original_grayscale_size": original_gray_size,
            "original_color_size": original_color_size,
            "compressed_grayscale_size": compressed_gray_size,
            "compressed_color_size": compressed_color_size,
            "total_compressed_size": compressed_total_size,
            "grayscale_compression_ratio": gray_ratio,
            "color_compression_ratio": color_ratio,
            "total_compression_ratio": total_ratio,
            "grayscale_compression_time": gray_compress_time,
            "color_compression_time": color_compress_time,
            "total_compression_time": total_compress_time,
            "frames_per_second": len(grayscale_frames) / total_compress_time if total_compress_time > 0 else 0,
            "bytes_per_frame": compressed_total_size / len(grayscale_frames) if len(grayscale_frames) > 0 else 0
        }
        
        # Generate compression report
        self._generate_color_compression_report(
            grayscale_frames, color_frames, metadata, performance, 
            os.path.join(output_dir, f"{experiment_name}_color_report.txt"))
        
        # Plot and save compression statistics
        self._plot_color_compression_stats(
            metadata, performance, 
            os.path.join(output_dir, f"{experiment_name}_color_stats.png"))
        
        # Save sample frames
        if len(color_frames) > 0:
            self._save_sample_color_frames(
                color_frames, os.path.join(output_dir, f"{experiment_name}_original_color_samples"))
        
        return compressed_gray, compressed_color, metadata, performance

    def _batch_process_grayscale(self, grayscale_frames: List[np.ndarray],
                              output_dir: str, 
                              experiment_name: str,
                              batch_size: int) -> Tuple[List[bytes], Dict]:
        """
        Process grayscale frames in batches for memory efficiency.
        
        Args:
            grayscale_frames: List of grayscale video frames
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            batch_size: Number of frames to process at once
            
        Returns:
            Tuple of (all_compressed_frames, combined_metadata)
        """
        # Initialize variables to accumulate results
        all_compressed_frames = []
        combined_metadata = {
            'frame_count': len(grayscale_frames),
            'frame_shape': grayscale_frames[0].shape if grayscale_frames else None,
            'keyframes': [],
            'densities': [],
            'compression_ratios': []
        }
        
        # Process frames in batches
        for i in range(0, len(grayscale_frames), batch_size):
            batch_end = min(i + batch_size, len(grayscale_frames))
            print(f"Processing grayscale batch {i//batch_size + 1}: frames {i}-{batch_end-1}")
            
            # Extract batch of frames
            batch_frames = grayscale_frames[i:batch_end]
            
            # Create temporary output path for this batch
            batch_path = os.path.join(output_dir, f"{experiment_name}_gray_batch_{i//batch_size + 1}.bin")
            
            # Compress this batch
            batch_compressed, batch_metadata = self.video_compressor.compress_video(
                batch_frames, output_path=batch_path)
            
            # Accumulate results
            all_compressed_frames.extend(batch_compressed)
            
            # Update metadata
            if 'keyframes' in batch_metadata:
                # Adjust keyframe indices for global position
                combined_metadata['keyframes'].extend([kf + i for kf in batch_metadata['keyframes']])
            
            if 'densities' in batch_metadata:
                combined_metadata['densities'].extend(batch_metadata['densities'])
            
            if 'compression_ratios' in batch_metadata:
                combined_metadata['compression_ratios'].extend(batch_metadata['compression_ratios'])
        
        return all_compressed_frames, combined_metadata

    def decompress_video_with_color(self, compressed_gray: List[bytes], 
                                 compressed_color: List[bytes],
                                 metadata: Dict, 
                                 output_dir: str, 
                                 experiment_name: str = "youtube_color_compression",
                                 fps: int = 30) -> Tuple[List[np.ndarray], float, str]:
        """
        Decompress video frames with color restoration.
        
        Args:
            compressed_gray: List of compressed grayscale frame data
            compressed_color: List of compressed color frame data
            metadata: Metadata from compression
            output_dir: Directory to save output files
            experiment_name: Name for the experiment files
            fps: Frames per second for the output video
            
        Returns:
            Tuple of (color_frames, decompression_time, output_video_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        decompressed_gray_path = os.path.join(output_dir, f"{experiment_name}_decompressed_gray.mp4")
        decompressed_color_path = os.path.join(output_dir, f"{experiment_name}_decompressed_color.mp4")
        
        print(f"Decompressing {len(compressed_gray)} frames with color restoration...")
        start_time = time.time()
        
        # Step 1: Decompress grayscale frames
        decompressed_gray = self.video_compressor.decompress_video(
            compressed_gray, metadata, output_path=decompressed_gray_path)
        
        gray_decompress_time = time.time() - start_time
        print(f"Grayscale decompression completed in {gray_decompress_time:.2f} seconds")
        
        # Step 2: Restore color
        color_start_time = time.time()
        restored_color_frames = self.restore_color(decompressed_gray, compressed_color)
        
        # Save the color video - the improved path will be returned later
        self._save_color_video(
            restored_color_frames, 
            decompressed_color_path, 
            fps)
        
        color_decompress_time = time.time() - color_start_time
        total_decompress_time = time.time() - start_time
        
        print(f"Color restoration completed in {color_decompress_time:.2f} seconds")
        print(f"Total decompression with color completed in {total_decompress_time:.2f} seconds")
        
        # Check for the ffmpeg version which should be more compatible
        ffmpeg_path = os.path.splitext(decompressed_color_path)[0] + "_ffmpeg.mp4"
        if os.path.exists(ffmpeg_path) and os.path.getsize(ffmpeg_path) > 0:
            output_video_path = ffmpeg_path
            print(f"Using ffmpeg-enhanced video: {output_video_path}")
        else:
            output_video_path = decompressed_color_path
            print(f"Using standard video: {output_video_path}")
        
        # Save some sample frames
        self._save_sample_color_frames(
            restored_color_frames, 
            os.path.join(output_dir, f"{experiment_name}_decompressed_color_samples"))
        
        return restored_color_frames, total_decompress_time, output_video_path

    def _generate_color_compression_report(self, grayscale_frames: List[np.ndarray],
                                        color_frames: List[np.ndarray],
                                        metadata: Dict, performance: Dict, 
                                        output_path: str) -> None:
        """Generate a detailed compression report with color metrics."""
        with open(output_path, 'w') as f:
            f.write("=== Bloom Filter Video Compression Report with Color Preservation ===\n\n")
            
            f.write("Video Statistics:\n")
            f.write(f"  Frame count: {len(grayscale_frames)}\n")
            if len(grayscale_frames) > 0:
                f.write(f"  Frame dimensions: {grayscale_frames[0].shape[1]}x{grayscale_frames[0].shape[0]}\n")
            
            f.write("\nSize Metrics:\n")
            f.write(f"  Original color size: {performance['original_color_size']:,} bytes\n")
            f.write(f"  Grayscale compressed size: {performance['compressed_grayscale_size']:,} bytes\n")
            f.write(f"  Color data compressed size: {performance['compressed_color_size']:,} bytes\n")
            f.write(f"  Total compressed size: {performance['total_compressed_size']:,} bytes\n")
            
            f.write("\nCompression Ratios:\n")
            f.write(f"  Grayscale compression ratio: {performance['grayscale_compression_ratio']:.6f}\n")
            f.write(f"  Color compression ratio: {performance['color_compression_ratio']:.6f}\n")
            f.write(f"  Total compression ratio: {performance['total_compression_ratio']:.6f}\n")
            f.write(f"  Space savings: {(1 - performance['total_compression_ratio']) * 100:.2f}%\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Grayscale compression time: {performance['grayscale_compression_time']:.2f} seconds\n")
            f.write(f"  Color compression time: {performance['color_compression_time']:.2f} seconds\n")
            f.write(f"  Total compression time: {performance['total_compression_time']:.2f} seconds\n")
            f.write(f"  Frames per second: {performance['frames_per_second']:.2f}\n")
            f.write(f"  Bytes per frame: {performance['bytes_per_frame']:.2f}\n\n")
            
            f.write("Color Preservation Settings:\n")
            f.write(f"  Chrominance downscaling: {metadata.get('chroma_downscale', 'N/A')}\n")
            
            f.write("\nBloom Filter Statistics:\n")
            if 'keyframes' in metadata:
                f.write(f"  Number of keyframes: {len(metadata['keyframes'])}\n")
            if 'densities' in metadata:
                avg_density = sum(metadata['densities']) / len(metadata['densities']) if metadata['densities'] else 0
                f.write(f"  Average frame difference density: {avg_density:.6f}\n")
        
        print(f"Color compression report saved to: {output_path}")

    def _plot_color_compression_stats(self, metadata: Dict, performance: Dict, output_path: str) -> None:
        """Plot compression statistics with color metrics and save to file."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Frame difference densities
        if 'densities' in metadata and metadata['densities']:
            axs[0, 0].plot(range(len(metadata['densities'])), metadata['densities'], 'o-')
            axs[0, 0].axhline(y=0.32453, color='r', linestyle='--', label='Threshold (0.32453)')
            axs[0, 0].set_xlabel('Frame')
            axs[0, 0].set_ylabel('Difference Density')
            axs[0, 0].set_title('Frame Difference Densities')
            axs[0, 0].grid(True)
            axs[0, 0].legend()
        
        # Plot 2: Compression ratio breakdown
        labels = ['Grayscale', 'Color', 'Total']
        compression_ratios = [
            performance['grayscale_compression_ratio'],
            performance['color_compression_ratio'],
            performance['total_compression_ratio']
        ]
        
        axs[0, 1].bar(labels, compression_ratios, color=['blue', 'green', 'red'])
        axs[0, 1].axhline(y=1.0, color='black', linestyle='--', label='No Compression')
        axs[0, 1].set_ylabel('Compression Ratio')
        axs[0, 1].set_title('Compression Ratio by Component')
        axs[0, 1].grid(True, axis='y')
        
        # Plot 3: Size breakdown
        sizes = [
            performance['compressed_grayscale_size'] / 1024,  # KB
            performance['compressed_color_size'] / 1024,      # KB
            performance['total_compressed_size'] / 1024       # KB
        ]
        original_size = performance['original_color_size'] / 1024  # KB
        
        axs[1, 0].bar(labels, sizes, color=['blue', 'green', 'red'])
        axs[1, 0].axhline(y=original_size, color='black', linestyle='--', 
                      label=f'Original Size ({original_size:.1f} KB)')
        axs[1, 0].set_ylabel('Size (KB)')
        axs[1, 0].set_title('Compressed Size by Component')
        axs[1, 0].grid(True, axis='y')
        axs[1, 0].legend()
        
        # Plot 4: Summary information
        axs[1, 1].axis('off')
        summary_text = (
            f"Color Compression Summary:\n\n"
            f"Original Size: {performance['original_color_size']/1024:.1f} KB\n"
            f"Compressed Size: {performance['total_compressed_size']/1024:.1f} KB\n"
            f"Overall Compression Ratio: {performance['total_compression_ratio']:.4f}\n"
            f"Space Savings: {(1 - performance['total_compression_ratio']) * 100:.2f}%\n\n"
            f"Chrominance Downscale: {metadata.get('chroma_downscale', 'N/A')}\n"
            f"Compression Time: {performance['total_compression_time']:.2f} seconds\n"
            f"Frames Per Second: {performance['frames_per_second']:.2f}\n"
        )
        axs[1, 1].text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Color compression statistics plot saved to: {output_path}")

    def _save_sample_color_frames(self, color_frames: List[np.ndarray], output_dir: str) -> None:
        """Save sample color frames from the video."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine indices of frames to save
        if len(color_frames) <= 5:
            indices = list(range(len(color_frames)))
        else:
            indices = [0, len(color_frames)//4, len(color_frames)//2, 3*len(color_frames)//4, len(color_frames)-1]
        
        # Save selected frames
        for i in indices:
            if i < len(color_frames):
                frame_path = os.path.join(output_dir, f"color_frame_{i:04d}.png")
                Image.fromarray(color_frames[i]).save(frame_path)
        
        print(f"Sample color frames saved to: {output_dir}")

    def process_youtube_video_with_color(self, url: str, 
                                      output_dir: str, 
                                      resolution: str = "360p",
                                      fps: int = 30,
                                      max_frames: int = 0,
                                      scale: float = 0.5,
                                      binarize: bool = True,
                                      threshold: int = 127,
                                      chroma_downscale: float = 0.5,
                                      batch_size: int = 0,
                                      experiment_name: str = None) -> Dict:
        """
        Complete pipeline to download, compress and restore color for a YouTube video.
        
        Args:
            url: YouTube video URL
            output_dir: Directory to save output files
            resolution: Video resolution to download
            fps: Target frames per second
            max_frames: Maximum number of frames to process (0 for all)
            scale: Scale factor for resizing frames
            binarize: Whether to binarize grayscale frames
            threshold: Threshold for binarization
            chroma_downscale: Downscaling factor for chrominance
            batch_size: Number of frames to process at once (0 for all at once)
            experiment_name: Name for the experiment (defaults to timestamp)
            
        Returns:
            Dictionary with performance metrics and file paths
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate experiment name if not provided
            if not experiment_name:
                timestamp = int(time.time())
                experiment_name = f"youtube_color_{timestamp}"
            
            print(f"Starting color-preserving compression for YouTube video: {url}")
            print(f"Experiment name: {experiment_name}")
            
            # Step 1: Download YouTube video
            video_path = self.download_video(url, resolution)
            
            # Step 2: Extract both grayscale and color frames
            grayscale_frames, color_frames = self.preprocess_video_with_color(
                video_path, fps, max_frames, scale, binarize, threshold)
            
            if not grayscale_frames or not color_frames:
                raise ValueError("No frames were extracted from the video")
            
            # Step 3: Compress with color information
            compressed_gray, compressed_color, metadata, performance = self.compress_video_with_color(
                grayscale_frames, color_frames, output_dir, experiment_name, 
                batch_size, chroma_downscale)
            
            # Step 4: Decompress and restore color
            restored_frames, decompress_time, output_video_path = self.decompress_video_with_color(
                compressed_gray, compressed_color, metadata, output_dir, experiment_name, fps)
            
            # Update performance metrics with decompression time
            performance['decompression_time'] = decompress_time
            
            # Save all performance metrics
            with open(os.path.join(output_dir, f"{experiment_name}_performance.txt"), 'w') as f:
                f.write("=== Performance Metrics ===\n\n")
                for key, value in performance.items():
                    f.write(f"{key}: {value}\n")
            
            return {
                'experiment_name': experiment_name,
                'output_dir': output_dir,
                'video_path': video_path,
                'frame_count': len(grayscale_frames),
                'compression_ratio': performance['total_compression_ratio'],
                'compression_time': performance['total_compression_time'],
                'decompression_time': decompress_time,
                'color_video_path': output_video_path,
            }
            
        except Exception as e:
            print(f"Error in color processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def load_compressed_color_data(self, compressed_color_path: str) -> List[bytes]:
        """
        Load compressed color data from a file.
        
        Args:
            compressed_color_path: Path to the compressed color data file
            
        Returns:
            List of compressed color frame data
        """
        print(f"Loading compressed color data from {compressed_color_path}...")
        
        compressed_color = []
        
        with open(compressed_color_path, 'rb') as f:
            # Read number of frames
            num_frames = int.from_bytes(f.read(4), byteorder='little')
            
            # Read each compressed color frame
            for _ in range(num_frames):
                # Read size of current frame data
                frame_size = int.from_bytes(f.read(4), byteorder='little')
                # Read frame data
                frame_data = f.read(frame_size)
                compressed_color.append(frame_data)
        
        print(f"Loaded {len(compressed_color)} compressed color frames")
        return compressed_color


def main():
    """Main function to run YouTube video compression."""
    parser = argparse.ArgumentParser(description='Compress YouTube videos with Bloom filter compression')
    
    parser.add_argument('url', type=str, help='YouTube video URL')
    parser.add_argument('--resolution', type=str, default='360p', 
                       help='Video resolution to download (default: 360p)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target frames per second (default: 30)')
    parser.add_argument('--max-frames', type=int, default=0, 
                       help='Maximum number of frames to process (0 for all frames, default: 0)')
    parser.add_argument('--scale', type=float, default=0.5, 
                       help='Scale factor for resizing frames (default: 0.5)')
    parser.add_argument('--binarize', action='store_true', 
                       help='Binarize frames for better compression')
    parser.add_argument('--threshold', type=int, default=127, 
                       help='Threshold for binarization (0-255, default: 127)')
    parser.add_argument('--output-dir', type=str, default='youtube_compression_results', 
                       help='Output directory for results (default: youtube_compression_results)')
    parser.add_argument('--experiment-name', type=str, 
                       help='Name for the experiment (default: based on video title)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Number of frames to process at once (0 for all at once, default: 0)')
    parser.add_argument('--preserve-color', action='store_true',
                       help='Preserve and restore color information')
    parser.add_argument('--chroma-downscale', type=float, default=0.5,
                       help='Downscaling factor for chrominance in color preservation (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize compressor
    compressor = YouTubeBloomCompressor()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Download YouTube video
        video_path = compressor.download_video(args.url, args.resolution)
        
        # Get video title for experiment name if not provided
        if not args.experiment_name:
            video_basename = os.path.basename(video_path)
            args.experiment_name = os.path.splitext(video_basename)[0]
        
        # Check if color preservation is requested
        if args.preserve_color:
            print("Running with color preservation...")
            result = compressor.process_youtube_video_with_color(
                args.url,
                args.output_dir,
                resolution=args.resolution,
                fps=args.fps,
                max_frames=args.max_frames,
                scale=args.scale,
                binarize=args.binarize,
                threshold=args.threshold,
                chroma_downscale=args.chroma_downscale,
                batch_size=args.batch_size,
                experiment_name=args.experiment_name
            )
            
            # Print summary
            if 'error' not in result:
                print("\n=== Color Compression Summary ===")
                print(f"YouTube Video: {args.url}")
                print(f"Resolution: {args.resolution}")
                print(f"Frames: {result['frame_count']} (max: {'all' if args.max_frames <= 0 else args.max_frames})")
                print(f"Compression Ratio: {result['compression_ratio']:.6f}")
                print(f"Space Savings: {(1 - result['compression_ratio']) * 100:.2f}%")
                print(f"Compression Time: {result['compression_time']:.2f} seconds")
                print(f"Decompression Time: {result['decompression_time']:.2f} seconds")
                print(f"Color Video Path: {result['color_video_path']}")
                print(f"\nResults saved to: {args.output_dir}")
                
                # Print playback instructions for the user
                if os.path.exists(result['color_video_path']):
                    print("\nTo play the video on your system, use:")
                    print(f"  open \"{result['color_video_path']}\"  # macOS")
                    print(f"  xdg-open \"{result['color_video_path']}\"  # Linux")
                    print(f"  start \"{result['color_video_path']}\"  # Windows")
                
                # Check for HTML viewer as fallback
                frames_dir = os.path.splitext(result['color_video_path'])[0] + "_frames"
                html_path = os.path.join(frames_dir, "viewer.html")
                if os.path.exists(html_path):
                    print("\nHTML viewer available as fallback:")
                    print(f"  open \"{html_path}\"  # macOS")
                    print(f"  xdg-open \"{html_path}\"  # Linux")
                    print(f"  start \"{html_path}\"  # Windows")
            else:
                print(f"Error: {result['error']}")
        
        else:
            # Standard grayscale processing
            # Preprocess video
            frames = compressor.preprocess_video(
                video_path, args.fps, args.max_frames, args.scale, args.binarize, args.threshold)
            
            if not frames:
                print("Error: No frames were extracted from the video.")
                return
            
            # Compress video
            compressed_frames, metadata, performance = compressor.compress_video(
                frames, args.output_dir, args.experiment_name, args.batch_size)
            
            # Decompress video
            decompressed_frames, decompress_time = compressor.decompress_video(
                compressed_frames, metadata, args.output_dir, args.experiment_name)
            
            # Verify lossless compression
            is_lossless, avg_difference = compressor.verify_lossless(
                frames, decompressed_frames, args.output_dir, args.experiment_name)
            
            # Print summary
            print("\n=== Compression Summary ===")
            print(f"YouTube Video: {args.url}")
            print(f"Resolution: {args.resolution}")
            print(f"Frames: {len(frames)} (max: {'all' if args.max_frames <= 0 else args.max_frames})")
            print(f"Frame Size: {frames[0].shape[1]}x{frames[0].shape[0]}")
            print(f"Original Size: {performance['original_size']:,} bytes")
            print(f"Compressed Size: {performance['compressed_size']:,} bytes")
            print(f"Compression Ratio: {performance['compression_ratio']:.6f}")
            print(f"Space Savings: {(1 - performance['compression_ratio']) * 100:.2f}%")
            print(f"Compression Time: {performance['compression_time']:.2f} seconds")
            print(f"Decompression Time: {decompress_time:.2f} seconds")
            print(f"Lossless: {is_lossless}")
            if not is_lossless:
                print(f"Average Frame Difference: {avg_difference:.6f}")
            print(f"\nResults saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 