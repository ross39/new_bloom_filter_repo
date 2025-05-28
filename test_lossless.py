#!/usr/bin/env python3
"""
Direct test of lossless reconstruction in the Improved Video Compressor.
This script focuses on verifying that the video compressor can achieve
true lossless reconstruction when processing raw video data.
"""

import os
import cv2
import numpy as np
from improved_video_compressor import ImprovedVideoCompressor
import time

def convert_frames_to_yuv(frames):
    """
    Convert BGR frames to YUV for direct YUV processing.
    
    Args:
        frames: List of BGR frames
        
    Returns:
        List of YUV frames with YUV planes stored
    """
    yuv_frames = []
    
    for frame in frames:
        # Convert BGR to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Create attribute dictionary
        yuv.yuv_info = {
            'format': 'YUV444',
            'y_plane': yuv[:, :, 0].copy(),
            'u_plane': yuv[:, :, 1].copy(),
            'v_plane': yuv[:, :, 2].copy()
        }
        
        yuv_frames.append(yuv)
    
    return yuv_frames

def test_lossless_reconstruction(video_path, max_frames=30, color_space="BGR"):
    """
    Test lossless reconstruction on a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to test
        color_space: Color space to use ("BGR" or "YUV")
    """
    print(f"Testing lossless reconstruction on: {video_path}")
    print(f"Max frames: {max_frames}")
    print(f"Color space: {color_space}")
    
    # Create compressor with direct YUV processing enabled
    compressor = ImprovedVideoCompressor(
        use_direct_yuv=(color_space == "YUV"),
        verbose=True
    )
    
    # Extract frames directly (no color space conversion)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video dimensions: {width}x{height} @ {fps} FPS")
    
    # Extract frames
    frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store as is - no conversion
        frames.append(frame)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    # Convert to YUV if requested
    if color_space == "YUV":
        print("Converting frames to YUV...")
        try:
            frames = convert_frames_to_yuv(frames)
            print("Conversion complete")
        except AttributeError:
            print("Error: Unable to set yuv_info attribute on numpy array")
            print("Trying another approach with direct YUV planes...")
            
            # Alternative approach: store Y, U, V planes separately
            yuv_planes = []
            for frame in frames:
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                # Store planes as a tuple
                yuv_planes.append((
                    yuv[:, :, 0].copy(),  # Y plane
                    yuv[:, :, 1].copy(),  # U plane
                    yuv[:, :, 2].copy()   # V plane
                ))
            
            # Keep original YUV arrays without attribute
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) for frame in frames]
            # Store planes separately
            frames_yuv_planes = yuv_planes
    
    # Create temporary directory
    temp_dir = "temp_lossless_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Compress the frames
    print("\nCompressing frames...")
    compressed_path = os.path.join(temp_dir, f"test_compressed_{color_space}.bfvc")
    start_time = time.time()
    compression_stats = compressor.compress_video(frames, compressed_path, input_color_space=color_space)
    compression_time = time.time() - start_time
    
    print(f"Compression time: {compression_time:.2f} seconds")
    print(f"Compression ratio: {compression_stats['compression_ratio']:.4f}")
    
    # Decompress the frames
    print("\nDecompressing frames...")
    start_time = time.time()
    decompressed_frames = compressor.decompress_video(compressed_path)
    decompression_time = time.time() - start_time
    
    print(f"Decompression time: {decompression_time:.2f} seconds")
    
    # Verify lossless reconstruction
    print("\nVerifying lossless reconstruction...")
    verification = compressor.verify_lossless(frames, decompressed_frames)
    
    print(f"Lossless: {verification['lossless']}")
    print(f"Exact lossless: {verification.get('exact_lossless', False)}")
    print(f"Average difference: {verification['avg_difference']}")
    
    if verification['lossless']:
        print("SUCCESS: Lossless reconstruction verified")
    else:
        print(f"FAILED: Reconstruction not lossless (avg diff: {verification['avg_difference']})")
        print(f"Maximum difference: {verification['max_difference']} (frame {verification['max_diff_frame']})")
        
        # Save the frames with maximum difference for inspection
        max_diff_frame = verification['max_diff_frame']
        if max_diff_frame < len(frames):
            # Convert to BGR for saving if needed
            orig_save = frames[max_diff_frame]
            decomp_save = decompressed_frames[max_diff_frame]
            
            if color_space == "YUV":
                orig_save = cv2.cvtColor(orig_save, cv2.COLOR_YUV2BGR)
                decomp_save = cv2.cvtColor(decomp_save, cv2.COLOR_YUV2BGR)
                
            orig_path = os.path.join(temp_dir, f"original_frame_{max_diff_frame}_{color_space}.png")
            decomp_path = os.path.join(temp_dir, f"decompressed_frame_{max_diff_frame}_{color_space}.png")
            
            cv2.imwrite(orig_path, orig_save)
            cv2.imwrite(decomp_path, decomp_save)
            
            print(f"Saved frames with maximum difference to {temp_dir}/")
            
            # Also create a difference visualization
            if color_space == "YUV":
                # For YUV, convert to RGB for visualization
                orig_rgb = cv2.cvtColor(orig_save, cv2.COLOR_BGR2RGB)
                decomp_rgb = cv2.cvtColor(decomp_save, cv2.COLOR_BGR2RGB)
            else:
                # For BGR, convert to RGB for visualization
                orig_rgb = cv2.cvtColor(frames[max_diff_frame], cv2.COLOR_BGR2RGB)
                decomp_rgb = cv2.cvtColor(decompressed_frames[max_diff_frame], cv2.COLOR_BGR2RGB)
            
            # Calculate absolute difference
            diff = np.abs(orig_rgb.astype(np.float32) - decomp_rgb.astype(np.float32))
            
            # Scale for visualization
            diff_scaled = np.clip(diff * 10, 0, 255).astype(np.uint8)
            
            # Save difference image
            diff_path = os.path.join(temp_dir, f"diff_frame_{max_diff_frame}_{color_space}.png")
            cv2.imwrite(diff_path, cv2.cvtColor(diff_scaled, cv2.COLOR_RGB2BGR))
    
    # Additional detailed analysis
    print("\nPerforming detailed analysis on channels...")
    analyze_channel_differences(frames, decompressed_frames, color_space)
    
    return verification['lossless']

def analyze_channel_differences(original_frames, decompressed_frames, color_space="BGR"):
    """
    Analyze differences between original and decompressed frames by channel.
    
    Args:
        original_frames: List of original frames
        decompressed_frames: List of decompressed frames
        color_space: Color space of the frames
    """
    if len(original_frames) != len(decompressed_frames):
        print("Error: Frame count mismatch")
        return
    
    # Only analyze a few frames for detailed output
    num_frames_to_analyze = min(5, len(original_frames))
    
    for i in range(num_frames_to_analyze):
        orig = original_frames[i]
        decomp = decompressed_frames[i]
        
        if orig.shape != decomp.shape:
            print(f"Error: Frame {i} shape mismatch")
            continue
        
        # Calculate differences for each channel
        diffs_by_channel = []
        
        for c in range(orig.shape[2]):
            orig_channel = orig[:, :, c].astype(float)
            decomp_channel = decomp[:, :, c].astype(float)
            
            diff = np.abs(orig_channel - decomp_channel)
            avg_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            diffs_by_channel.append({
                'channel': c,
                'avg_diff': avg_diff,
                'max_diff': max_diff,
                'num_nonzero': np.count_nonzero(diff)
            })
        
        # Print results for this frame
        print(f"\nFrame {i} channel analysis:")
        for c_diff in diffs_by_channel:
            if color_space == "BGR":
                channel_name = "B" if c_diff['channel'] == 0 else "G" if c_diff['channel'] == 1 else "R"
            else:  # YUV
                channel_name = "Y" if c_diff['channel'] == 0 else "U" if c_diff['channel'] == 1 else "V"
                
            print(f"  Channel {channel_name}: avg={c_diff['avg_diff']:.6f}, max={c_diff['max_diff']:.6f}, non-zero pixels={c_diff['num_nonzero']}")
        
        # Calculate combined difference
        frame_diff = np.mean(np.abs(orig.astype(float) - decomp.astype(float)))
        print(f"  Overall difference: {frame_diff:.6f}")

if __name__ == "__main__":
    import sys
    
    # Use the first command-line argument as the video path, or default to the akiyo test video
    video_path = sys.argv[1] if len(sys.argv) > 1 else "raw_videos/downloads/akiyo_cif.y4m"
    
    # Get max frames from second argument, or default to 30
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Test with BGR color space
    print("\n===== Testing with BGR color space =====\n")
    test_lossless_reconstruction(video_path, max_frames, "BGR")
    
    # Test with YUV color space
    print("\n===== Testing with YUV color space =====\n")
    test_lossless_reconstruction(video_path, max_frames, "YUV") 