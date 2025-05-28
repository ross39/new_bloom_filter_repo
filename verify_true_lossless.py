#!/usr/bin/env python3
"""
True Lossless Verification Test Script

This script performs rigorous testing of the lossless compression capabilities
of the rational Bloom filter video compression system, ensuring bit-exact
reconstruction with zero tolerance for any rounding errors.
"""

import os
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from improved_video_compressor import ImprovedVideoCompressor

def test_true_lossless(video_path, max_frames=30, color_spaces=None,
                      keyframe_interval=10, save_diagnostics=True,
                      output_dir="true_lossless_results"):
    """
    Test for true bit-exact lossless reconstruction across different color spaces.
    
    Args:
        video_path: Path to test video
        max_frames: Maximum frames to test
        color_spaces: List of color spaces to test ("BGR", "RGB", "YUV")
        keyframe_interval: Interval between keyframes for compression
        save_diagnostics: Whether to save diagnostic information
        output_dir: Directory to save results
    
    Returns:
        Dictionary with test results
    """
    # Default color spaces if none provided
    if color_spaces is None:
        color_spaces = ["BGR", "YUV"]
    
    # Prepare output directory
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video frames once
    frames = extract_frames(video_path, max_frames)
    if not frames:
        print(f"Error: Failed to extract frames from {video_path}")
        return {"success": False, "error": "Failed to extract frames"}
    
    print(f"Testing with {len(frames)} frames from {video_path}")
    print(f"Frame dimensions: {frames[0].shape}")
    
    # Record overall results
    results = {
        "video_path": str(video_path),
        "frames_tested": len(frames),
        "frame_dimensions": frames[0].shape,
        "color_space_results": {}
    }
    
    # Test each color space
    for cs in color_spaces:
        print(f"\n{'='*80}")
        print(f"Testing {cs} color space")
        print(f"{'='*80}")
        
        # Convert frames to the target color space
        cs_frames = convert_to_color_space(frames, cs)
        
        # Run the compression test
        cs_result = test_color_space(
            cs_frames, 
            color_space=cs,
            keyframe_interval=keyframe_interval,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir / cs
        )
        
        # Store results
        results["color_space_results"][cs] = cs_result
    
    # Calculate overall success
    all_success = all(r.get("success", False) for r in results["color_space_results"].values())
    results["overall_success"] = all_success
    
    # Print summary
    print("\nOverall Results Summary:")
    print(f"  Video: {video_path}")
    print(f"  Frames tested: {len(frames)}")
    for cs, result in results["color_space_results"].items():
        status = "SUCCESS" if result.get("success", False) else "FAILED"
        print(f"  {cs}: {status}")
        if not result.get("success", False):
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nFinal result: {'SUCCESS' if all_success else 'FAILED'}")
    return results

def extract_frames(video_path, max_frames):
    """Extract frames from a video file."""
    print(f"Extracting frames from {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {width}x{height}, {fps} FPS, {total_frames} total frames")
    
    # Adjust max_frames if needed
    if max_frames <= 0 or max_frames > total_frames:
        max_frames = total_frames
    
    # Extract frames
    frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())  # Make a copy to ensure we have a clean frame
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames

def convert_to_color_space(frames, color_space):
    """Convert frames to the specified color space."""
    if not frames:
        return []
    
    # Return original frames for BGR (OpenCV default)
    if color_space == "BGR":
        return [f.copy() for f in frames]  # Return copies to avoid modifying originals
    
    converted_frames = []
    
    for frame in frames:
        if color_space == "RGB":
            # Convert BGR to RGB
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_space == "YUV":
            # Convert BGR to YUV
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Store YUV planes for perfect reconstruction
            # We can't add attributes to numpy arrays, so we'll use a structured array
            converted = add_yuv_info_to_frame(converted)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
        
        converted_frames.append(converted)
    
    return converted_frames

def add_yuv_info_to_frame(yuv_frame):
    """
    Add YUV plane information to a frame.
    
    Since we can't add arbitrary attributes to numpy arrays directly,
    we create a wrapper class to hold both the frame data and YUV info.
    """
    class YUVFrame:
        def __init__(self, frame):
            self.data = frame
            self.yuv_info = {
                'format': 'YUV444',
                'y_plane': frame[:, :, 0].copy(),
                'u_plane': frame[:, :, 1].copy(),
                'v_plane': frame[:, :, 2].copy()
            }
            self.shape = frame.shape
            self.dtype = frame.dtype
            self.nbytes = frame.nbytes
        
        def __array__(self):
            return self.data
        
        def copy(self):
            return YUVFrame(self.data.copy())
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
            
        def tobytes(self):
            """Return the raw bytes of the frame data."""
            return self.data.tobytes()
            
        def astype(self, dtype):
            """Convert the frame data to the specified type."""
            return self.data.astype(dtype)
            
        # Add compatibility methods for numpy array interface
        def __repr__(self):
            return f"YUVFrame(shape={self.shape}, dtype={self.dtype})"
            
        def flatten(self):
            return self.data.flatten()
            
        def reshape(self, *args, **kwargs):
            return self.data.reshape(*args, **kwargs)
            
        @property
        def size(self):
            return self.data.size
            
        @property
        def T(self):
            return self.data.T
    
    return YUVFrame(yuv_frame)

def test_color_space(frames, color_space, keyframe_interval=10, 
                   save_diagnostics=True, output_dir=None):
    """
    Test lossless compression and reconstruction in a specific color space.
    
    Args:
        frames: List of frames in the specified color space
        color_space: Color space being tested
        keyframe_interval: Interval between keyframes
        save_diagnostics: Whether to save diagnostic information
        output_dir: Directory to save results
    
    Returns:
        Dictionary with test results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize compressor with appropriate settings
    compressor = ImprovedVideoCompressor(
        use_direct_yuv=(color_space == "YUV"),
        keyframe_interval=keyframe_interval,
        noise_tolerance=0.0,  # Minimum noise tolerance
        min_diff_threshold=0.0,  # Catch any differences
        max_diff_threshold=10.0,
        bloom_threshold_modifier=1.0,
        verbose=True
    )
    
    # First, test with a single frame to verify we have no serialization issues
    print(f"Testing single frame compression in {color_space} color space...")
    single_frame_path = os.path.join(output_dir, f"test_single_frame_{color_space}.bfvc") if output_dir else None
    
    try:
        # Try with a single frame first
        single_frame = frames[0]
        if isinstance(single_frame, np.ndarray):
            # Regular numpy array
            single_frame_test = [single_frame.copy()]
        else:
            # Custom frame class
            single_frame_test = [frames[0].copy()]
            
        compressor.compress_video(
            single_frame_test,
            single_frame_path,
            input_color_space=color_space
        )
        print("Single frame test successful")
    except Exception as e:
        return {
            "success": False,
            "error": f"Single frame test failed: {str(e)}"
        }
    
    # Now test with all frames
    print(f"Compressing {len(frames)} frames in {color_space} color space...")
    compressed_path = os.path.join(output_dir, f"compressed_{color_space}.bfvc") if output_dir else None
    
    try:
        start_time = time.time()
        compression_stats = compressor.compress_video(
            frames, 
            compressed_path,
            input_color_space=color_space
        )
        compression_time = time.time() - start_time
        
        # Decompress
        print(f"Decompressing video...")
        start_time = time.time()
        decompressed_frames = compressor.decompress_video(compressed_path)
        decompression_time = time.time() - start_time
        
        # Verify true lossless reconstruction
        print(f"Verifying bit-exact reconstruction...")
        verification = compressor.verify_lossless(frames, decompressed_frames)
        
        # Detailed bit-level verification
        bit_exact_verification = verify_bit_exact(frames, decompressed_frames, 
                                               color_space=color_space,
                                               save_diagnostics=save_diagnostics,
                                               output_dir=output_dir)
        
        # Combine results
        result = {
            "success": verification["exact_lossless"] and bit_exact_verification["success"],
            "compression_ratio": compression_stats["overall_ratio"],
            "compression_time": compression_time,
            "decompression_time": decompression_time,
            "frames_per_second_compress": len(frames) / compression_time,
            "frames_per_second_decompress": len(frames) / decompression_time,
            "verification_result": verification,
            "bit_exact_verification": bit_exact_verification
        }
        
        # Print summary
        print(f"\n{color_space} Results:")
        print(f"  Compression ratio: {compression_stats['overall_ratio']:.4f}")
        print(f"  Compression time: {compression_time:.2f}s ({result['frames_per_second_compress']:.2f} FPS)")
        print(f"  Decompression time: {decompression_time:.2f}s ({result['frames_per_second_decompress']:.2f} FPS)")
        print(f"  Exact lossless: {verification['exact_lossless']}")
        print(f"  Exact frame matches: {verification['exact_frame_matches']}/{len(frames)}")
        
        if not verification["exact_lossless"]:
            print(f"  Average difference: {verification['avg_difference']}")
            print(f"  Maximum difference: {verification['max_difference']} (frame {verification['max_diff_frame']})")
        
        return result
    
    except Exception as e:
        print(f"Error in {color_space} test: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def verify_bit_exact(original_frames, decompressed_frames, color_space="BGR",
                    save_diagnostics=True, output_dir=None):
    """
    Perform manual bit-exact verification between original and decompressed frames.
    
    This function compares every single byte to ensure perfect reconstruction.
    
    Args:
        original_frames: Original video frames
        decompressed_frames: Decompressed video frames
        color_space: Color space of the frames
        save_diagnostics: Whether to save diagnostic information
        output_dir: Directory to save diagnostics
    
    Returns:
        Dictionary with verification results
    """
    print("Performing bit-exact verification...")
    
    if len(original_frames) != len(decompressed_frames):
        return {
            "success": False,
            "error": f"Frame count mismatch: {len(original_frames)} vs {len(decompressed_frames)}"
        }
    
    # Track differences
    exact_matches = 0
    diff_frames = []
    diff_details = []
    
    for i, (orig, decomp) in enumerate(zip(original_frames, decompressed_frames)):
        try:
            # Handle wrapped YUV frames
            if hasattr(orig, 'data') and hasattr(decomp, 'data'):
                orig_data = orig.data
                decomp_data = decomp.data
            else:
                orig_data = orig
                decomp_data = decomp
            
            # Check if frames have the same shape
            if orig_data.shape != decomp_data.shape:
                diff_frames.append(i)
                diff_details.append({
                    "frame": i,
                    "error": f"Shape mismatch: {orig_data.shape} vs {decomp_data.shape}"
                })
                continue
            
            # Direct byte-level comparison
            if np.array_equal(orig_data, decomp_data):
                exact_matches += 1
            else:
                diff_frames.append(i)
                
                # Find differences
                try:
                    diff = np.abs(orig_data.astype(np.int16) - decomp_data.astype(np.int16))
                    diff_indices = np.where(diff > 0)
                    
                    # Collect the first few differences for analysis
                    diff_examples = []
                    if len(diff_indices[0]) > 0:
                        for idx in range(min(10, len(diff_indices[0]))):
                            coords = tuple(axis[idx] for axis in diff_indices)
                            orig_val = int(orig_data[coords])
                            decomp_val = int(decomp_data[coords])
                            diff_val = int(diff[coords])
                            
                            diff_examples.append({
                                "coordinates": str(coords),
                                "original_value": orig_val,
                                "decompressed_value": decomp_val,
                                "difference": diff_val
                            })
                    
                    diff_details.append({
                        "frame": i,
                        "differences_found": len(diff_indices[0]),
                        "examples": diff_examples
                    })
                except Exception as e:
                    diff_details.append({
                        "frame": i,
                        "error": f"Error calculating differences: {str(e)}"
                    })
                
                # Save problem frames if requested
                if save_diagnostics and output_dir:
                    try:
                        # Ensure we're saving in a standard format
                        if color_space == "YUV":
                            if hasattr(orig, 'data'):
                                orig_save = cv2.cvtColor(orig.data, cv2.COLOR_YUV2BGR)
                                decomp_save = cv2.cvtColor(decomp.data, cv2.COLOR_YUV2BGR)
                            else:
                                orig_save = cv2.cvtColor(orig, cv2.COLOR_YUV2BGR)
                                decomp_save = cv2.cvtColor(decomp, cv2.COLOR_YUV2BGR)
                        elif color_space == "RGB":
                            orig_save = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
                            decomp_save = cv2.cvtColor(decomp, cv2.COLOR_RGB2BGR)
                        else:
                            orig_save = orig.copy()
                            decomp_save = decomp.copy()
                        
                        # Create a difference visualization (if possible)
                        if 'diff' in locals():
                            diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(output_dir, f"frame_{i}_diff.png"), diff_vis)
                        
                        # Save the images
                        cv2.imwrite(os.path.join(output_dir, f"frame_{i}_original.png"), orig_save)
                        cv2.imwrite(os.path.join(output_dir, f"frame_{i}_decompressed.png"), decomp_save)
                    except Exception as e:
                        print(f"Error saving diagnostic images for frame {i}: {str(e)}")
        except Exception as e:
            diff_frames.append(i)
            diff_details.append({
                "frame": i,
                "error": f"Error processing frame: {str(e)}"
            })
    
    # Compile results
    success = (exact_matches == len(original_frames))
    
    result = {
        "success": success,
        "frames_compared": len(original_frames),
        "exact_matches": exact_matches,
        "different_frames": len(diff_frames),
        "different_frame_indices": diff_frames,
        "diff_details": diff_details
    }
    
    # Print summary
    print(f"Bit-exact verification: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Exact frame matches: {exact_matches}/{len(original_frames)}")
    
    if not success:
        print(f"  Frames with differences: {len(diff_frames)}")
        for detail in diff_details[:3]:  # Show first 3 problem frames
            frame_idx = detail.get("frame", "unknown")
            if "error" in detail:
                print(f"  Frame {frame_idx}: Error - {detail['error']}")
            else:
                print(f"  Frame {frame_idx}: {detail.get('differences_found', 0)} differences")
                for ex in detail.get('examples', [])[:3]:  # Show first 3 examples per frame
                    coords = ex.get("coordinates", "unknown")
                    print(f"    Pos {coords}: orig={ex.get('original_value')}, "
                          f"decomp={ex.get('decompressed_value')}, diff={ex.get('difference')}")
        
        if len(diff_details) > 3:
            print(f"  ... and {len(diff_details) - 3} more frames with differences")
    
    return result

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Verify true lossless video compression with bit-exact reconstruction"
    )
    
    parser.add_argument("video_path", type=str, 
                      help="Path to the test video file")
    parser.add_argument("--max-frames", type=int, default=30,
                      help="Maximum number of frames to test")
    parser.add_argument("--color-spaces", type=str, nargs="+", 
                      choices=["BGR", "RGB", "YUV"], default=["BGR", "YUV"],
                      help="Color spaces to test")
    parser.add_argument("--keyframe-interval", type=int, default=10,
                      help="Interval between keyframes")
    parser.add_argument("--output-dir", type=str, default="true_lossless_results",
                      help="Directory to save results")
    parser.add_argument("--no-diagnostics", action="store_true",
                      help="Disable saving diagnostic information")
    
    args = parser.parse_args()
    
    test_true_lossless(
        video_path=args.video_path,
        max_frames=args.max_frames,
        color_spaces=args.color_spaces,
        keyframe_interval=args.keyframe_interval,
        save_diagnostics=not args.no_diagnostics,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 