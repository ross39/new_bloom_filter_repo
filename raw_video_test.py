#!/usr/bin/env python3
"""
Scientific Raw Video Compression Test

This script provides a rigorous scientific test of the rational Bloom filter
compression system on real-world raw video. It evaluates compression ratio,
lossless reconstruction, processing time, and memory usage.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import psutil
import json
import subprocess
from improved_video_compressor import ImprovedVideoCompressor

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bytes):
            return str(obj)
        return super().default(obj)

class RawVideoTest:
    """Scientific test harness for raw video compression."""
    
    def __init__(self, output_dir="raw_video_results"):
        """Initialize the test environment."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.log_file = self.output_dir / "test_log.txt"
        
        # Initialize metrics storage
        self.metrics = {
            "test_parameters": {},
            "compression_metrics": {},
            "decompression_metrics": {},
            "quality_metrics": {},
            "system_metrics": {}
        }

    def log(self, message):
        """Log a message to both console and log file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def extract_raw_frames(self, video_path, max_frames=0, frame_step=1, start_frame=0, 
                      width=None, height=None, yuv_format='I420'):
        """
        Extract raw frames from a video file.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (0 = all)
            frame_step: Extract every Nth frame
            start_frame: Starting frame index
            width: Frame width (required for raw YUV files)
            height: Frame height (required for raw YUV files)
            yuv_format: YUV format for raw files (default: I420/YUV420P)
        
        Returns:
            List of extracted frames and video metadata
        """
        self.log(f"Extracting raw frames from: {video_path}")
        video_path = str(video_path)
        
        # Check if this is a raw YUV file
        if video_path.lower().endswith('.yuv'):
            if width is None or height is None:
                self.log("Error: Width and height must be specified for raw YUV files")
                self.log("For the Xiph test files, use: width=352, height=288")
                raise ValueError("Width and height required for YUV files")
            
            return self._extract_from_raw_yuv(
                video_path, width, height, yuv_format,
                max_frames, frame_step, start_frame
            )
        
        # Use OpenCV for other video formats
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.log(f"Starting at frame {start_frame}")
        
        # Calculate frames to extract
        if max_frames > 0:
            frames_to_extract = min(max_frames, (total_frames - start_frame) // frame_step)
        else:
            frames_to_extract = (total_frames - start_frame) // frame_step
        
        self.log(f"Extracting {frames_to_extract} frames (step={frame_step})")
        
        # Extract frames
        frames = []
        frame_idx = start_frame
        frames_extracted = 0
        
        while frames_extracted < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx - start_frame) % frame_step == 0:
                # Don't convert to BGR - keep raw data
                frames.append(frame)
                frames_extracted += 1
                
                if frames_extracted % 100 == 0:
                    self.log(f"Extracted {frames_extracted}/{frames_to_extract} frames...")
            
            frame_idx += 1
        
        cap.release()
        
        self.log(f"Extracted {len(frames)} frames total")
        
        # Save video metadata
        metadata = {
            "original_video": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "extracted_frames": len(frames),
            "frame_step": frame_step,
            "start_frame": start_frame
        }
        
        return frames, metadata

    def _extract_from_raw_yuv(self, video_path, width, height, yuv_format='I420',
                           max_frames=0, frame_step=1, start_frame=0):
        """
        Extract frames from a raw YUV file.
        
        Args:
            video_path: Path to the YUV file
            width: Frame width
            height: Frame height
            yuv_format: YUV format (I420, YV12, etc.)
            max_frames: Maximum frames to extract
            frame_step: Extract every Nth frame
            start_frame: Starting frame index
            
        Returns:
            List of frames and metadata
        """
        self.log(f"Extracting from raw YUV file: {width}x{height}, format={yuv_format}")
        
        # Calculate frame size based on format
        if yuv_format in ['I420', 'YV12', 'yuv420p']:
            # 4:2:0 format - Y gets full resolution, U and V are quarter size each
            frame_size = width * height + (width // 2) * (height // 2) * 2
        elif yuv_format in ['YUV422', 'yuv422p']:
            # 4:2:2 format - Y gets full resolution, U and V are half width
            frame_size = width * height + width * (height // 2)
        elif yuv_format in ['YUV444', 'yuv444p']:
            # 4:4:4 format - All channels at full resolution
            frame_size = width * height * 3
        else:
            raise ValueError(f"Unsupported YUV format: {yuv_format}")
        
        # Open the file
        with open(video_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)     # Back to start
            
            # Calculate total frames in file
            total_frames = file_size // frame_size
            self.log(f"File contains approximately {total_frames} frames")
            
            # Skip to start frame
            if start_frame > 0:
                f.seek(start_frame * frame_size)
                self.log(f"Starting at frame {start_frame}")
            
            # Calculate frames to extract
            if max_frames > 0:
                frames_to_extract = min(max_frames, (total_frames - start_frame) // frame_step)
            else:
                frames_to_extract = (total_frames - start_frame) // frame_step
            
            self.log(f"Extracting {frames_to_extract} frames (step={frame_step})")
            
            # Extract frames
            frames = []
            for i in range(frames_to_extract):
                # Read raw frame data
                frame_data = f.read(frame_size)
                if not frame_data or len(frame_data) < frame_size:
                    break
                
                # Skip frames according to frame_step
                if i % frame_step != 0:
                    continue
                
                # Convert YUV to BGR for OpenCV compatibility
                if yuv_format in ['I420', 'YV12', 'yuv420p']:
                    # Create Y, U, V arrays
                    Y = np.frombuffer(frame_data[:width*height], dtype=np.uint8).reshape((height, width))
                    
                    if yuv_format == 'YV12':  # YV12 has V before U
                        V_size = (width // 2) * (height // 2)
                        V = np.frombuffer(frame_data[width*height:width*height+V_size], 
                                         dtype=np.uint8).reshape((height // 2, width // 2))
                        U = np.frombuffer(frame_data[width*height+V_size:], 
                                         dtype=np.uint8).reshape((height // 2, width // 2))
                    else:  # I420/YUV420P has U before V
                        U_size = (width // 2) * (height // 2)
                        U = np.frombuffer(frame_data[width*height:width*height+U_size], 
                                         dtype=np.uint8).reshape((height // 2, width // 2))
                        V = np.frombuffer(frame_data[width*height+U_size:], 
                                         dtype=np.uint8).reshape((height // 2, width // 2))
                    
                    # Use INTER_NEAREST to avoid interpolation artifacts in chroma upsampling
                    # This improves numerical stability when going YUV->BGR->YUV
                    U_upsampled = cv2.resize(U, (width, height), interpolation=cv2.INTER_NEAREST)
                    V_upsampled = cv2.resize(V, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Merge to YUV
                    yuv = cv2.merge([Y, U_upsampled, V_upsampled])
                    
                    # Convert to BGR using the correct colorspace conversion
                    # This ensures exact consistency when converting back
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    frames.append(frame)
                else:
                    # For other formats, you'd need to implement appropriate conversion
                    raise ValueError(f"Conversion from {yuv_format} not implemented")
                
                if len(frames) % 10 == 0:
                    self.log(f"Extracted {len(frames)} frames...")
                
                # Skip to next frame position based on frame_step
                if frame_step > 1 and i < frames_to_extract - 1:
                    f.seek((frame_step - 1) * frame_size, 1)  # 1 = relative to current position
        
        self.log(f"Extracted {len(frames)} frames total from YUV file")
        
        # Create metadata
        metadata = {
            "original_video": str(video_path),
            "width": width,
            "height": height,
            "fps": 30.0,  # Assume 30fps for raw files
            "total_frames": total_frames,
            "extracted_frames": len(frames),
            "frame_step": frame_step,
            "start_frame": start_frame,
            "format": yuv_format
        }
        
        return frames, metadata
    
    def compress_video(self, frames, test_name, params=None):
        """
        Compress video frames with accurate metrics collection.
        
        Args:
            frames: List of video frames
            test_name: Name of the test
            params: Compression parameters
            
        Returns:
            Path to compressed file and compression metrics
        """
        # Default parameters if none provided
        if params is None:
            params = {
                "noise_tolerance": 5.0,
                "keyframe_interval": 30,
                "min_diff_threshold": 2.0,
                "max_diff_threshold": 20.0,
                "bloom_threshold_modifier": 0.9,
                "batch_size": 30
            }
        
        self.log(f"\nCompressing {len(frames)} frames with parameters:")
        for k, v in params.items():
            self.log(f"  {k}: {v}")
        
        # Initialize compressor with parameters
        compressor = ImprovedVideoCompressor(
            noise_tolerance=params["noise_tolerance"],
            keyframe_interval=params["keyframe_interval"],
            min_diff_threshold=params["min_diff_threshold"],
            max_diff_threshold=params["max_diff_threshold"],
            bloom_threshold_modifier=params["bloom_threshold_modifier"],
            batch_size=params["batch_size"],
            verbose=True
        )
        
        # Memory usage before compression
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Compress and time
        compressed_path = self.results_dir / f"{test_name}_compressed.bfvc"
        start_time = time.time()
        compression_stats = compressor.compress_video(frames, str(compressed_path))
        compression_time = time.time() - start_time
        
        # Memory usage after compression
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Calculate metrics
        original_size = sum(frame.nbytes for frame in frames)
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = compressed_size / original_size
        
        # Store metrics
        metrics = {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time_seconds": compression_time,
            "frames_per_second": len(frames) / compression_time,
            "memory_usage_before_mb": mem_before,
            "memory_usage_after_mb": mem_after,
            "memory_increase_mb": mem_after - mem_before,
            "keyframes": compression_stats.get("keyframes", 0),
            "keyframe_ratio": compression_stats.get("keyframe_ratio", 0)
        }
        
        self.log(f"Compression results:")
        self.log(f"  Original size: {original_size/1024/1024:.2f} MB")
        self.log(f"  Compressed size: {compressed_size/1024/1024:.2f} MB")
        self.log(f"  Compression ratio: {compression_ratio:.4f} ({compression_ratio*100:.1f}%)")
        self.log(f"  Space savings: {(1-compression_ratio)*100:.1f}%")
        self.log(f"  Compression time: {compression_time:.2f}s")
        self.log(f"  Frames per second: {len(frames)/compression_time:.2f}")
        self.log(f"  Memory usage increase: {metrics['memory_increase_mb']:.2f} MB")
        
        # Save sample frames
        self._save_sample_frames(frames, self.samples_dir / f"{test_name}_original")
        
        return compressed_path, metrics
    
    def decompress_video(self, compressed_path, test_name, original_frames=None):
        """
        Decompress video with metrics collection.
        
        Args:
            compressed_path: Path to compressed video
            test_name: Name of the test
            original_frames: Optional original frames for verification
            
        Returns:
            Decompressed frames and metrics
        """
        self.log(f"\nDecompressing video: {compressed_path}")
        
        # Initialize compressor
        compressor = ImprovedVideoCompressor(verbose=True)
        
        # Memory and time measurement
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        start_time = time.time()
        
        # Decompress
        decompressed_frames = compressor.decompress_video(str(compressed_path))
        
        decompression_time = time.time() - start_time
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Calculate metrics
        metrics = {
            "decompression_time_seconds": decompression_time,
            "frames_per_second": len(decompressed_frames) / decompression_time,
            "memory_usage_before_mb": mem_before,
            "memory_usage_after_mb": mem_after,
            "memory_increase_mb": mem_after - mem_before,
        }
        
        self.log(f"Decompression results:")
        self.log(f"  Frames decompressed: {len(decompressed_frames)}")
        self.log(f"  Decompression time: {decompression_time:.2f}s")
        self.log(f"  Frames per second: {metrics['frames_per_second']:.2f}")
        self.log(f"  Memory usage increase: {metrics['memory_increase_mb']:.2f} MB")
        
        # Save sample frames
        self._save_sample_frames(decompressed_frames, self.samples_dir / f"{test_name}_decompressed")
        
        # Verify lossless if original frames provided
        if original_frames is not None:
            verification = compressor.verify_lossless(original_frames, decompressed_frames)
            
            # Add verification metrics
            metrics.update({
                "lossless": verification["lossless"],
                "avg_difference": verification["avg_difference"],
                "max_difference": verification["max_difference"],
                "max_diff_frame": verification["max_diff_frame"]
            })
            
            self.log(f"Verification results:")
            self.log(f"  Lossless: {verification['lossless']}")
            self.log(f"  Average difference: {verification['avg_difference']:.6f}")
            
            if not verification['lossless']:
                self.log(f"  Maximum difference: {verification['max_difference']:.6f} (frame {verification['max_diff_frame']})")
                
                # Calculate PSNR for quality assessment
                psnr_values = []
                for orig, decomp in zip(original_frames, decompressed_frames):
                    mse = np.mean((orig.astype(np.float32) - decomp.astype(np.float32)) ** 2)
                    if mse > 0:
                        psnr = 10 * np.log10((255 ** 2) / mse)
                    else:
                        psnr = float('inf')
                    psnr_values.append(psnr)
                
                avg_psnr = np.mean([p for p in psnr_values if p != float('inf')])
                metrics["psnr"] = avg_psnr
                self.log(f"  PSNR: {avg_psnr:.2f} dB")
                
                # Save difference visualization for a sample frame
                if verification['max_diff_frame'] < len(original_frames):
                    self._save_difference_visualization(
                        original_frames[verification['max_diff_frame']],
                        decompressed_frames[verification['max_diff_frame']],
                        self.samples_dir / f"{test_name}_frame_{verification['max_diff_frame']}_diff.png"
                    )
        
        return decompressed_frames, metrics
    
    def compare_with_standard_codecs(self, frames, test_name, codecs=None):
        """
        Compare with standard video codecs.
        
        Args:
            frames: Video frames
            test_name: Test name
            codecs: List of codecs to test
            
        Returns:
            Comparison metrics
        """
        if codecs is None:
            codecs = [
                {"name": "H.264", "fourcc": "X264", "ext": "mp4"},
                {"name": "H.265", "fourcc": "HEVC", "ext": "mp4"},
                {"name": "FFV1", "fourcc": "FFV1", "ext": "avi"},
                {"name": "Lossless JPEG2000", "fourcc": "MJPG", "ext": "avi"}
            ]
            
        self.log("\nComparing with standard codecs:")
        
        original_size = sum(frame.nbytes for frame in frames)
        results = {}
        
        for codec in codecs:
            self.log(f"\nTesting codec: {codec['name']}")
            output_path = self.results_dir / f"{test_name}_{codec['name'].replace('.', '')}.{codec['ext']}"
            
            # Save video with this codec
            self._save_with_codec(frames, str(output_path), codec["fourcc"])
            
            # Calculate compression ratio
            codec_size = os.path.getsize(output_path)
            codec_ratio = codec_size / original_size
            
            self.log(f"  Size: {codec_size/1024/1024:.2f} MB")
            self.log(f"  Ratio: {codec_ratio:.4f} ({codec_ratio*100:.1f}%)")
            
            results[codec["name"]] = {
                "size_bytes": codec_size,
                "compression_ratio": codec_ratio
            }
        
        return results
        
    def run_full_test(self, video_path, test_name, params=None, max_frames=0, 
                    frame_step=1, start_frame=0, compare_codecs=True, width=None, height=None, yuv_format='I420'):
        """
        Run a complete test on a video.
        
        Args:
            video_path: Path to the raw video
            test_name: Name for this test
            params: Compression parameters
            max_frames: Maximum frames to process
            frame_step: Process every nth frame
            start_frame: Starting frame
            compare_codecs: Whether to compare with standard codecs
            width: Frame width (required for raw YUV files)
            height: Frame height (required for raw YUV files)
            yuv_format: YUV format for raw files
            
        Returns:
            Test metrics
        """
        self.log(f"\n{'='*80}")
        self.log(f"STARTING TEST: {test_name}")
        self.log(f"{'='*80}")
        self.log(f"Video path: {video_path}")
        
        # Store test parameters
        self.metrics["test_parameters"] = {
            "test_name": test_name,
            "video_path": str(video_path),
            "max_frames": max_frames,
            "frame_step": frame_step,
            "start_frame": start_frame,
            "compression_params": params,
            "width": width,
            "height": height,
            "yuv_format": yuv_format
        }
        
        # Extract frames
        frames, video_metadata = self.extract_raw_frames(
            video_path, max_frames, frame_step, start_frame, width, height, yuv_format)
        self.metrics["video_metadata"] = video_metadata
        
        # Compress
        compressed_path, compression_metrics = self.compress_video(frames, test_name, params)
        self.metrics["compression_metrics"] = compression_metrics
        
        # Decompress and verify
        decompressed_frames, decompression_metrics = self.decompress_video(
            compressed_path, test_name, frames)
        self.metrics["decompression_metrics"] = decompression_metrics
        
        # Compare with standard codecs
        if compare_codecs:
            codec_results = self.compare_with_standard_codecs(frames, test_name)
            self.metrics["codec_comparison"] = codec_results
        
        # Save complete metrics
        self._save_metrics(test_name)
        
        # Generate plots
        self._generate_plots(test_name)
        
        self.log(f"\n{'='*80}")
        self.log(f"TEST COMPLETE: {test_name}")
        self.log(f"{'='*80}")
        
        return self.metrics
    
    def _save_sample_frames(self, frames, output_dir, num_samples=5):
        """Save sample frames from the video."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Choose sample indices
        if len(frames) <= num_samples:
            indices = list(range(len(frames)))
        else:
            step = len(frames) // num_samples
            indices = [i * step for i in range(num_samples)]
            if indices[-1] != len(frames) - 1:
                indices.append(len(frames) - 1)
        
        # Save frames
        for i, idx in enumerate(indices):
            frame = frames[idx]
            # Save BGR image
            cv2.imwrite(str(output_dir / f"frame_{idx:04d}.png"), frame)
    
    def _save_difference_visualization(self, original, decompressed, output_path):
        """Save visualization of differences between original and decompressed."""
        # Calculate absolute difference
        diff = np.abs(original.astype(np.float32) - decompressed.astype(np.float32))
        
        # Scale for visualization (multiply by 10 to make small differences visible)
        diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Original
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')
        
        # Decompressed
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(decompressed, cv2.COLOR_BGR2RGB))
        plt.title("Decompressed")
        plt.axis('off')
        
        # Difference (heat map)
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Difference (scaled 10x)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
    
    def _save_with_codec(self, frames, output_path, fourcc):
        """Save frames using a specific codec."""
        if not frames:
            return
        
        # Get dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        out = cv2.VideoWriter(output_path, fourcc_code, 30, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def _save_metrics(self, test_name):
        """Save metrics to JSON file."""
        metrics_path = self.results_dir / f"{test_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, cls=NumpyEncoder)
    
    def _generate_plots(self, test_name):
        """Generate plots summarizing test results."""
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Compression ratio comparison
        if "codec_comparison" in self.metrics:
            self._plot_codec_comparison(test_name, plot_dir)
    
    def _plot_codec_comparison(self, test_name, plot_dir):
        """Plot comparison with standard codecs."""
        # Extract data
        bloom_ratio = self.metrics["compression_metrics"]["compression_ratio"]
        codec_ratios = {
            name: data["compression_ratio"] 
            for name, data in self.metrics["codec_comparison"].items()
        }
        
        # All ratios
        all_codecs = {"Bloom Filter": bloom_ratio, **codec_ratios}
        
        # Sort by ratio
        sorted_codecs = {k: v for k, v in sorted(all_codecs.items(), key=lambda item: item[1])}
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Bar chart
        codecs = list(sorted_codecs.keys())
        ratios = [sorted_codecs[c] * 100 for c in codecs]
        
        # Color the Bloom Filter bar differently
        colors = ['#2C7BB6' if c != "Bloom Filter" else '#D7191C' for c in codecs]
        
        plt.bar(codecs, ratios, color=colors)
        plt.ylabel('Compression Ratio (%)')
        plt.title('Compression Ratio Comparison')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(ratios):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(str(plot_dir / f"{test_name}_codec_comparison.png"), dpi=150)
        plt.close()

def main():
    """Run the raw video test."""
    parser = argparse.ArgumentParser(description="Scientific Raw Video Compression Test")
    
    parser.add_argument("video_path", type=str, 
                      help="Path to the raw video file to test")
    parser.add_argument("--test-name", type=str, default="raw_video_test",
                      help="Name for this test run")
    parser.add_argument("--max-frames", type=int, default=0,
                      help="Maximum number of frames to process (0=all)")
    parser.add_argument("--frame-step", type=int, default=1,
                      help="Process every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0,
                      help="Starting frame index")
    parser.add_argument("--width", type=int, default=None,
                      help="Frame width (required for raw YUV files)")
    parser.add_argument("--height", type=int, default=None,
                      help="Frame height (required for raw YUV files)")
    parser.add_argument("--yuv-format", type=str, default="I420",
                      choices=["I420", "YV12", "YUV422", "YUV444"],
                      help="YUV format for raw files")
    parser.add_argument("--noise-tolerance", type=float, default=5.0,
                      help="Noise tolerance level")
    parser.add_argument("--keyframe-interval", type=int, default=30,
                      help="Maximum frames between keyframes")
    parser.add_argument("--min-diff", type=float, default=2.0,
                      help="Minimum threshold for pixel differences")
    parser.add_argument("--max-diff", type=float, default=20.0,
                      help="Maximum threshold for pixel differences")
    parser.add_argument("--bloom-modifier", type=float, default=0.9,
                      help="Modifier for Bloom filter threshold")
    parser.add_argument("--batch-size", type=int, default=30,
                      help="Number of frames to process in each batch")
    parser.add_argument("--output-dir", type=str, default="raw_video_results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create test runner
    tester = RawVideoTest(output_dir=args.output_dir)
    
    # Set compression parameters
    params = {
        "noise_tolerance": args.noise_tolerance,
        "keyframe_interval": args.keyframe_interval,
        "min_diff_threshold": args.min_diff,
        "max_diff_threshold": args.max_diff,
        "bloom_threshold_modifier": args.bloom_modifier,
        "batch_size": args.batch_size
    }
    
    # Run the test
    tester.run_full_test(
        video_path=args.video_path,
        test_name=args.test_name,
        params=params,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        width=args.width,
        height=args.height,
        yuv_format=args.yuv_format
    )

if __name__ == "__main__":
    main() 