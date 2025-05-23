#!/usr/bin/env python3
"""
Simple test script for the YouTube Bloom Filter compression.
This will download and compress a short public domain video from YouTube.
If downloading fails, it will fall back to generating a synthetic video.
"""

import os
import time
import argparse
import numpy as np
from youtube_bloom_compress import YouTubeBloomCompressor
from video_delta_compress import generate_synthetic_video_with_pattern

# Default test video (NASA official channel - short Earth view, public domain)
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=q37DTIJEfhY"


def generate_fallback_video(pattern="circle", 
                           frame_count=100, 
                           width=320, 
                           height=240,
                           output_dir="test_output"):
    """Generate a synthetic video as fallback if YouTube download fails."""
    print(f"\nGenerating synthetic {pattern} video as fallback...")
    frames = generate_synthetic_video_with_pattern(
        pattern, frame_count, width, height
    )
    
    # Save the first frame for reference
    os.makedirs(output_dir, exist_ok=True)
    synthetic_path = os.path.join(output_dir, "synthetic_video")
    os.makedirs(synthetic_path, exist_ok=True)
    
    # Return the frames directly
    print(f"Generated {len(frames)} synthetic frames")
    return frames


def run_test(url: str = DEFAULT_VIDEO_URL, 
             resolution: str = "360p",
             max_frames: int = 0,
             binarize: bool = True,
             output_dir: str = "test_output",
             batch_size: int = 0):
    """
    Run a test of the YouTube video compression.
    
    Args:
        url: YouTube video URL
        resolution: Video resolution
        max_frames: Maximum frames to process (0 for all frames)
        binarize: Whether to binarize frames
        output_dir: Output directory
        batch_size: Number of frames to process at once (0 for all at once)
    """
    print(f"=== YouTube Bloom Filter Compression Test ===")
    print(f"Video URL: {url}")
    print(f"Resolution: {resolution}")
    print(f"Max Frames: {'all' if max_frames <= 0 else max_frames}")
    print(f"Binarize: {binarize}")
    print(f"Batch Size: {'all at once' if batch_size <= 0 else batch_size}")
    print(f"Output Directory: {output_dir}")
    print("=" * 45)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize compressor
    compressor = YouTubeBloomCompressor(temp_dir=os.path.join(output_dir, "temp"))
    
    # Start timing
    total_start_time = time.time()
    
    try:
        # Step 1: Get video frames (from YouTube or synthetic)
        frames = None
        try:
            # First attempt: Download YouTube video
            print("\nStep 1: Downloading video...")
            video_path = compressor.download_video(url, resolution)
            
            # Step 2: Preprocess video
            print("\nStep 2: Preprocessing video...")
            frames = compressor.preprocess_video(
                video_path, 
                fps=30, 
                max_frames=max_frames,
                scale=0.5,
                binarize=binarize
            )
        except Exception as e:
            print(f"\nYouTube download failed: {e}")
            print("Falling back to synthetic video...")
            # Fallback to synthetic video - limit frames if processing all
            frames_for_synthetic = 100 if max_frames <= 0 else max_frames
            frames = generate_fallback_video(
                pattern="circle", 
                frame_count=frames_for_synthetic,
                output_dir=output_dir
            )
        
        if not frames or len(frames) == 0:
            raise ValueError("No frames were obtained from the video source")
            
        print(f"Working with {len(frames)} frames")
        
        # Step 3: Compress video
        print("\nStep 3: Compressing video...")
        compressed_frames, metadata, performance = compressor.compress_video(
            frames, output_dir, "test_compression", batch_size
        )
        
        # Step 4: Decompress video
        print("\nStep 4: Decompressing video...")
        decompressed_frames, decompress_time = compressor.decompress_video(
            compressed_frames, metadata, output_dir, "test_compression"
        )
        
        # Step 5: Verify lossless
        print("\nStep 5: Verifying lossless compression...")
        is_lossless, avg_difference = compressor.verify_lossless(
            frames, decompressed_frames, output_dir, "test_compression"
        )
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Print final summary
        print("\n" + "=" * 30)
        print("Test Completed Successfully!")
        print("=" * 30)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Original size: {performance['original_size']:,} bytes")
        print(f"Compressed size: {performance['compressed_size']:,} bytes")
        print(f"Compression ratio: {performance['compression_ratio']:.6f}")
        print(f"Space savings: {(1 - performance['compression_ratio']) * 100:.2f}%")
        print(f"Lossless reconstruction: {is_lossless}")
        if not is_lossless:
            print(f"Average frame difference: {avg_difference:.6f}")
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Test YouTube Bloom filter compression')
    parser.add_argument('--url', type=str, default=DEFAULT_VIDEO_URL,
                        help=f'YouTube video URL (default: {DEFAULT_VIDEO_URL})')
    parser.add_argument('--resolution', type=str, default='360p',
                        help='Video resolution (default: 360p)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Maximum frames to process (0 for all frames, default: 0)')
    parser.add_argument('--no-binarize', action='store_true',
                        help='Disable binarization of frames')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory (default: test_output)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Number of frames to process at once (0 for all at once, default: 0)')
    
    args = parser.parse_args()
    
    run_test(
        url=args.url,
        resolution=args.resolution,
        max_frames=args.max_frames,
        binarize=not args.no_binarize,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 