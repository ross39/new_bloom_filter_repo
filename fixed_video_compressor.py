#!/usr/bin/env python3
"""
Simplified ImprovedVideoCompressor for true lossless video compression
"""

import os
import cv2
import numpy as np
import zlib
import struct
import io
import time
from typing import List, Dict, Tuple, Optional

class FixedVideoCompressor:
    """
    True Lossless Video Compression System
    
    This class provides a mathematically lossless video compression system that guarantees
    bit-exact reconstruction of the original video frames with zero tolerance for errors.
    """
    
    def __init__(self, verbose=True):
        """Initialize the compressor."""
        self.verbose = verbose
        
    def compress_frame(self, frame: np.ndarray) -> bytes:
        """Compress a single frame with bit-exact preservation."""
        # Direct compression with no preprocessing
        frame_bytes = frame.tobytes()
        compressed_frame = zlib.compress(frame_bytes, level=9)
        
        # Create buffer
        buffer = io.BytesIO()
        
        # Store frame info
        buffer.write(struct.pack('<III', frame.shape[0], frame.shape[1], frame.dtype.itemsize))
        
        # Store compressed data
        buffer.write(struct.pack('<I', len(compressed_frame)))
        buffer.write(compressed_frame)
        
        # Record if this is a special YUV frame
        has_yuv_info = hasattr(frame, 'yuv_info')
        buffer.write(struct.pack('<B', 1 if has_yuv_info else 0))
        
        if has_yuv_info:
            # Store YUV format
            yuv_format = frame.yuv_info.get('format', 'YUV444').encode('utf-8')
            buffer.write(struct.pack('<H', len(yuv_format)))
            buffer.write(yuv_format)
            
            # Store Y plane
            y_plane = frame.yuv_info['y_plane'].tobytes()
            y_compressed = zlib.compress(y_plane, level=9)
            buffer.write(struct.pack('<I', len(y_compressed)))
            buffer.write(y_compressed)
            buffer.write(struct.pack('<II', *frame.yuv_info['y_plane'].shape))
            
            # Store U plane
            u_plane = frame.yuv_info['u_plane'].tobytes()
            u_compressed = zlib.compress(u_plane, level=9)
            buffer.write(struct.pack('<I', len(u_compressed)))
            buffer.write(u_compressed)
            buffer.write(struct.pack('<II', *frame.yuv_info['u_plane'].shape))
            
            # Store V plane
            v_plane = frame.yuv_info['v_plane'].tobytes()
            v_compressed = zlib.compress(v_plane, level=9)
            buffer.write(struct.pack('<I', len(v_compressed)))
            buffer.write(v_compressed)
            buffer.write(struct.pack('<II', *frame.yuv_info['v_plane'].shape))
        
        return buffer.getvalue()
    
    def decompress_frame(self, compressed_data: bytes) -> np.ndarray:
        """Decompress a single frame with bit-exact precision."""
        buffer = io.BytesIO(compressed_data)
        
        # Read shape and data type
        height, width, dtype_size = struct.unpack('<III', buffer.read(12))
        
        # Read compressed data
        compressed_size = struct.unpack('<I', buffer.read(4))[0]
        compressed_frame = buffer.read(compressed_size)
        
        # Decompress
        frame_data = zlib.decompress(compressed_frame)
        
        # Convert to numpy array with exact dtype
        if dtype_size == 1:
            dtype = np.uint8
        elif dtype_size == 2:
            dtype = np.uint16
        else:
            dtype = np.float32
        
        # Determine if this is a color frame by checking the data size
        data_size = len(frame_data)
        expected_gray_size = height * width * dtype_size
        
        if data_size > expected_gray_size and data_size % expected_gray_size == 0:
            # Color frame - calculate number of channels
            channels = data_size // expected_gray_size
            frame = np.frombuffer(frame_data, dtype=dtype).reshape((height, width, channels))
        else:
            # Grayscale frame
            frame = np.frombuffer(frame_data, dtype=dtype).reshape((height, width))
        
        # Check for YUV info
        try:
            has_yuv_info = struct.unpack('<B', buffer.read(1))[0] == 1
        except:
            has_yuv_info = False
        
        if has_yuv_info:
            # Create YUV frame wrapper
            class YUVFrame:
                def __init__(self, data):
                    self.data = data
                    self.shape = data.shape
                    self.dtype = data.dtype
                    self.yuv_info = {}
                    self.nbytes = data.nbytes
                    
                def __array__(self):
                    return self.data
                    
                def copy(self):
                    new_frame = YUVFrame(self.data.copy())
                    new_frame.yuv_info = {k: v.copy() for k, v in self.yuv_info.items()}
                    return new_frame
                    
                def __getitem__(self, key):
                    return self.data[key]
                    
                def __setitem__(self, key, value):
                    self.data[key] = value
                    
                def tobytes(self):
                    return self.data.tobytes()
            
            # Create frame wrapper
            yuv_frame = YUVFrame(frame)
            
            # Read YUV format
            yuv_format_len = struct.unpack('<H', buffer.read(2))[0]
            yuv_format = buffer.read(yuv_format_len).decode('utf-8')
            
            # Read Y plane
            y_compressed_size = struct.unpack('<I', buffer.read(4))[0]
            y_compressed = buffer.read(y_compressed_size)
            y_height, y_width = struct.unpack('<II', buffer.read(8))
            y_data = zlib.decompress(y_compressed)
            y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape((y_height, y_width))
            
            # Read U plane
            u_compressed_size = struct.unpack('<I', buffer.read(4))[0]
            u_compressed = buffer.read(u_compressed_size)
            u_height, u_width = struct.unpack('<II', buffer.read(8))
            u_data = zlib.decompress(u_compressed)
            u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape((u_height, u_width))
            
            # Read V plane
            v_compressed_size = struct.unpack('<I', buffer.read(4))[0]
            v_compressed = buffer.read(v_compressed_size)
            v_height, v_width = struct.unpack('<II', buffer.read(8))
            v_data = zlib.decompress(v_compressed)
            v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape((v_height, v_width))
            
            # Set YUV info
            yuv_frame.yuv_info = {
                'format': yuv_format,
                'y_plane': y_plane,
                'u_plane': u_plane,
                'v_plane': v_plane
            }
            
            return yuv_frame
        
        return frame
    
    def compress_video(self, frames: List[np.ndarray]) -> List[bytes]:
        """Compress a sequence of frames with bit-exact preservation."""
        if self.verbose:
            print(f"Compressing {len(frames)} frames")
        
        compressed_frames = []
        
        for i, frame in enumerate(frames):
            # Compress each frame directly
            compressed_data = self.compress_frame(frame)
            compressed_frames.append(compressed_data)
            
            if self.verbose and (i+1) % 10 == 0:
                print(f"Compressed {i+1}/{len(frames)} frames")
        
        return compressed_frames
    
    def decompress_video(self, compressed_frames: List[bytes]) -> List[np.ndarray]:
        """Decompress a sequence of frames with bit-exact precision."""
        if self.verbose:
            print(f"Decompressing {len(compressed_frames)} frames")
        
        decompressed_frames = []
        
        for i, compressed_data in enumerate(compressed_frames):
            # Decompress each frame
            frame = self.decompress_frame(compressed_data)
            decompressed_frames.append(frame)
            
            if self.verbose and (i+1) % 10 == 0:
                print(f"Decompressed {i+1}/{len(compressed_frames)} frames")
        
        return decompressed_frames
    
    def verify_lossless(self, original_frames: List[np.ndarray], 
                      decompressed_frames: List[np.ndarray]) -> Dict:
        """
        Verify that decompression is truly lossless with bit-exact reconstruction.
        """
        if len(original_frames) != len(decompressed_frames):
            return {
                'lossless': False,
                'reason': f"Frame count mismatch: {len(original_frames)} vs {len(decompressed_frames)}",
                'avg_difference': float('inf')
            }
        
        # Track frame-by-frame differences
        exact_matches = 0
        diff_frames = []
        max_diff = 0
        max_diff_frame = -1
        
        for i, (orig, decomp) in enumerate(zip(original_frames, decompressed_frames)):
            # Handle YUV frames
            if hasattr(orig, 'data'):
                orig_data = orig.data
            else:
                orig_data = orig
                
            if hasattr(decomp, 'data'):
                decomp_data = decomp.data
            else:
                decomp_data = decomp
            
            # Check for exact byte-for-byte equality
            if np.array_equal(orig_data, decomp_data):
                exact_matches += 1
                frame_diff = 0.0
            else:
                # Not an exact match - compute difference
                diff = np.abs(orig_data.astype(np.float32) - decomp_data.astype(np.float32))
                frame_diff = np.mean(diff)
                diff_frames.append(i)
                
                if frame_diff > max_diff:
                    max_diff = frame_diff
                    max_diff_frame = i
        
        # Calculate overall metrics
        avg_diff = 0.0 if len(diff_frames) == 0 else max_diff  # Worst-case difference
        is_lossless = exact_matches == len(original_frames)
        
        # Prepare result
        result = {
            'lossless': is_lossless,
            'exact_lossless': is_lossless,
            'avg_difference': avg_diff,
            'max_difference': max_diff,
            'max_diff_frame': max_diff_frame,
            'exact_frame_matches': exact_matches,
            'total_frames': len(original_frames),
            'diff_frames': diff_frames
        }
        
        if self.verbose:
            print(f"Lossless verification: {'SUCCESS' if is_lossless else 'FAILED'}")
            print(f"Exact frame matches: {exact_matches}/{len(original_frames)}")
            
            if not is_lossless:
                print(f"Frames with differences: {len(diff_frames)}")
                print(f"Maximum difference: {max_diff} (frame {max_diff_frame})")
        
        return result
    
    def add_yuv_info_to_frame(self, yuv_frame):
        """Add YUV plane information to a frame."""
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
                return self.data.tobytes()
                
            def astype(self, dtype):
                return self.data.astype(dtype)
                
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

def test_lossless():
    """Test the lossless compression system."""
    # Create test image
    print("Creating test image...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (25, 25), (75, 75), (0, 255, 0), -1)
    cv2.circle(test_image, (50, 50), 25, (0, 0, 255), -1)
    
    # Create compressor
    compressor = FixedVideoCompressor(verbose=True)
    
    # Test with single frame
    print("\nTesting with single frame...")
    test_frames = [test_image.copy()]
    
    # Compress
    compressed_frames = compressor.compress_video(test_frames)
    
    # Decompress
    decompressed_frames = compressor.decompress_video(compressed_frames)
    
    # Verify
    result = compressor.verify_lossless(test_frames, decompressed_frames)
    
    print(f"\nSingle frame test result: {'SUCCESS' if result['lossless'] else 'FAILED'}")
    
    # Test with multiple frames
    print("\nTesting with multiple frames...")
    test_frames = []
    for i in range(5):
        frame = test_image.copy()
        # Add some variation
        cv2.putText(frame, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        test_frames.append(frame)
    
    # Compress
    compressed_frames = compressor.compress_video(test_frames)
    
    # Decompress
    decompressed_frames = compressor.decompress_video(compressed_frames)
    
    # Verify
    result = compressor.verify_lossless(test_frames, decompressed_frames)
    
    print(f"\nMultiple frame test result: {'SUCCESS' if result['lossless'] else 'FAILED'}")
    
    # Test with YUV frames
    print("\nTesting with YUV frames...")
    yuv_frames = []
    for frame in test_frames:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv_with_info = compressor.add_yuv_info_to_frame(yuv)
        yuv_frames.append(yuv_with_info)
    
    # Compress
    compressed_frames = compressor.compress_video(yuv_frames)
    
    # Decompress
    decompressed_frames = compressor.decompress_video(compressed_frames)
    
    # Verify
    result = compressor.verify_lossless(yuv_frames, decompressed_frames)
    
    print(f"\nYUV frame test result: {'SUCCESS' if result['lossless'] else 'FAILED'}")
    
    print("\nAll tests complete")

if __name__ == "__main__":
    test_lossless() 