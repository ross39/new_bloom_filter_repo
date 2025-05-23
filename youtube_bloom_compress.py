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
from typing import List, Dict, Tuple, Optional

# Import the existing compressor
from video_delta_compress import VideoDeltaCompressor


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