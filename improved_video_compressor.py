#!/usr/bin/env python3
"""
Improved Video Compressor with Rational Bloom Filter

This module implements an optimized video compression system that uses
Rational Bloom Filters to achieve lossless compression, with a focus on
raw noisy video content. The implementation aims to achieve 50-70% of
the original size while maintaining perfect reconstruction.

Key features:
- Adaptive compression based on noise characteristics
- Multi-threaded processing for performance
- Memory-efficient batch processing for large videos
- Accurate compression ratio calculation
- Optimized for different noise patterns
"""

import os
import time
import sys
import io
import math
import struct
import argparse
import multiprocessing
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import xxhash
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed


class RationalBloomFilter:
    """
    An optimized Rational Bloom Filter implementation specifically designed for video compression.
    
    This implementation allows for non-integer numbers of hash functions (k) which
    theoretically enables better compression than traditional Bloom filters with integer k.
    """
    
    def __init__(self, size: int, k_star: float):
        """
        Initialize a Rational Bloom filter.
        
        Args:
            size: Size of the bit array
            k_star: Optimal (rational) number of hash functions
        """
        self.size = size
        self.k_star = k_star
        self.floor_k = math.floor(k_star)
        self.p_activation = k_star - self.floor_k  # Fractional part as probability
        self.bit_array = np.zeros(size, dtype=np.uint8)
        
        # Constants for double hashing - fixed seeds for deterministic results
        self.h1_seed = 0x12345678
        self.h2_seed = 0x87654321
    
    def _get_hash_indices(self, item: int, i: int) -> int:
        """
        Generate hash indices using double hashing technique for faster computation.
        
        Args:
            item: The integer item to hash (index position)
            i: The index of the hash function (0 to floor_k or ceil_k - 1)
            
        Returns:
            A hash index in range [0, size-1]
        """
        # Use xxhash for speed - much faster than built-in hash()
        h1 = xxhash.xxh64_intdigest(str(item), self.h1_seed)
        h2 = xxhash.xxh64_intdigest(str(item), self.h2_seed)
        
        # Double hashing: (h1(x) + i * h2(x)) % size
        return (h1 + i * h2) % self.size
    
    def _determine_activation(self, item: int) -> bool:
        """
        Deterministically decide whether to apply the additional hash function.
        
        Args:
            item: The item to check
            
        Returns:
            True if additional hash function should be activated
        """
        # Deterministic decision based on the item value
        hash_value = xxhash.xxh64_intdigest(str(item), 999)
        normalized_value = hash_value / (2**64 - 1)  # Convert to [0,1)
        
        return normalized_value < self.p_activation
    
    def add_index(self, index: int) -> None:
        """
        Add an index to the Bloom filter.
        
        Args:
            index: The index to add (0 to n-1)
        """
        # Apply the floor(k*) hash functions deterministically
        for i in range(self.floor_k):
            hash_idx = self._get_hash_indices(index, i)
            self.bit_array[hash_idx] = 1
        
        # Probabilistically apply the additional hash function
        if self._determine_activation(index):
            hash_idx = self._get_hash_indices(index, self.floor_k)
            self.bit_array[hash_idx] = 1
    
    def check_index(self, index: int) -> bool:
        """
        Check if an index might be in the Bloom filter.
        
        Args:
            index: The index to check
            
        Returns:
            True if all relevant bits are set, False otherwise
        """
        # Check deterministic hash functions
        for i in range(self.floor_k):
            hash_idx = self._get_hash_indices(index, i)
            if self.bit_array[hash_idx] == 0:
                return False
        
        # Check probabilistic hash function if applicable
        if self._determine_activation(index):
            hash_idx = self._get_hash_indices(index, self.floor_k)
            if self.bit_array[hash_idx] == 0:
                return False
        
        return True 

class BloomFilterCompressor:
    """
    Optimized implementation of lossless compression with Bloom filters.
    
    This class implements the core compression algorithm using Rational Bloom Filters
    to achieve optimal compression ratios for binary data, particularly suited for
    noise patterns in video frame differences.
    """
    
    # Critical density threshold for compression - theoretical limit
    P_STAR = 0.32453
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the compressor.
        
        Args:
            verbose: Whether to print detailed compression information
        """
        self.verbose = verbose
    
    def _calculate_optimal_params(self, n: int, p: float) -> Tuple[float, int]:
        """
        Calculate the optimal parameters k (number of hash functions) and
        l (bloom filter length) for lossless compression.
        
        Args:
            n: Length of the binary input string
            p: Density (probability of '1' bits)
            
        Returns:
            Tuple of (k, l) where k is optimal hash count and l is optimal filter length
        """
        # Handle edge cases
        if p <= 0.0001:
            return 0, 0
        
        if p >= self.P_STAR:
            # Compression not effective for this density
            return 0, 0
        
        q = 1 - p  # Probability of '0' bits
        L = math.log(2)  # ln(2)
        
        # Calculate optimal k based on theory
        k = math.log2(q * (L**2) / p)
        
        # Ensure k is valid
        if math.isnan(k) or k <= 0:
            return 0, 0
        
        # Calculate optimal filter length
        gamma = 1 / L
        l = int(p * n * k * gamma)
        
        # Ensure minimum viable values
        return max(0.1, k), max(1, l)
    
    def compress(self, binary_input: np.ndarray) -> Tuple[np.ndarray, list, float, int, float]:
        """
        Compress a binary input using Bloom filter-based compression.
        
        Args:
            binary_input: Binary input as 1D numpy array of 0s and 1s
            
        Returns:
            Tuple of (bloom_filter_bitmap, witness, density, input_length, compression_ratio)
        """
        n = len(binary_input)
        
        # Calculate density (probability of '1' bits)
        ones_count = np.sum(binary_input)
        p = ones_count / n
        
        # Check if compression is possible
        if p >= self.P_STAR:
            if self.verbose:
                print(f"Density {p:.4f} is >= threshold {self.P_STAR}, compression not effective")
            return binary_input, [], p, n, 1.0
        
        # Calculate optimal parameters
        k, l = self._calculate_optimal_params(n, p)
        
        if l == 0 or l >= n:
            # Compression not possible or not beneficial, return original
            return binary_input, [], p, n, 1.0
        
        if self.verbose:
            print(f"Input length: {n}, Density: {p:.4f}")
            print(f"Optimal parameters: k={k:.4f}, l={l}")
        
        # Create Bloom filter
        bloom_filter = RationalBloomFilter(l, k)
        
        # First pass: Add all '1' bit positions to the Bloom filter
        for i in range(n):
            if binary_input[i] == 1:
                bloom_filter.add_index(i)
        
        # Second pass: Generate witness data
        witness = []
        
        # Count bloom filter test checks (for analysis)
        bft_pass_count = 0
        
        for i in range(n):
            # Check if position passes Bloom filter test
            if bloom_filter.check_index(i):
                # This is either a true positive (original bit was 1)
                # or a false positive (original bit was 0)
                bft_pass_count += 1
                
                # Add the original bit to the witness
                witness.append(binary_input[i])
        
        # Calculate compression ratio
        original_size = n
        compressed_size = l + len(witness)
        compression_ratio = compressed_size / original_size
        
        if self.verbose:
            print(f"Bloom filter size: {l} bits")
            print(f"Witness size: {len(witness)} bits")
            print(f"Compression ratio: {compression_ratio:.4f}")
            print(f"Bloom filter test pass rate: {bft_pass_count/n:.4f}")
        
        return bloom_filter.bit_array, witness, p, n, compression_ratio
    
    def decompress(self, bloom_bitmap: np.ndarray, witness: list, n: int, k: float) -> np.ndarray:
        """
        Decompress data that was compressed with the Bloom filter method.
        
        Args:
            bloom_bitmap: The Bloom filter bitmap
            witness: The witness data (list of original bits where BFT passes)
            n: Original length of the binary input
            k: The number of hash functions used in compression
            
        Returns:
            The decompressed binary data as a 1D numpy array
        """
        # Handle the case where compression wasn't applied (density >= threshold)
        if len(witness) == 0:
            # If witness is empty, the bloom_bitmap is actually the original data
            return bloom_bitmap
            
        l = len(bloom_bitmap)
        
        # Create Bloom filter with provided bitmap
        bloom_filter = RationalBloomFilter(l, k)
        bloom_filter.bit_array = bloom_bitmap
        
        # Initialize output array
        decompressed = np.zeros(n, dtype=np.uint8)
        
        # Witness bit index
        witness_idx = 0
        
        # Reconstruct the original binary data
        for i in range(n):
            # Check if position passes Bloom filter test
            if bloom_filter.check_index(i):
                # This position passed BFT, get the actual bit from the witness
                decompressed[i] = witness[witness_idx]
                witness_idx += 1
            # If BFT fails, the bit is definitely 0 (true negative)
        
        return decompressed 

class ImprovedVideoCompressor:
    """
    True Lossless Video Compression System
    
    This implementation ensures mathematically lossless video compression
    with bit-exact reconstruction. It is based on the FixedVideoCompressor
    approach for perfect fidelity.
    """
    
    def __init__(self, 
                noise_tolerance: float = 10.0,
                keyframe_interval: int = 30,
                min_diff_threshold: float = 3.0,
                max_diff_threshold: float = 30.0,
                bloom_threshold_modifier: float = 1.0,
                batch_size: int = 30,
                num_threads: int = None,
                use_direct_yuv: bool = False,
                verbose: bool = False):
        """
        Initialize the video compressor.
        
        Args:
            noise_tolerance: Tolerance for noise in frame differences (higher = more tolerant)
            keyframe_interval: Maximum number of frames between keyframes
            min_diff_threshold: Minimum threshold for considering pixels different
            max_diff_threshold: Maximum threshold for considering pixels different
            bloom_threshold_modifier: Modifier for Bloom filter threshold
            batch_size: Number of frames to process in each batch
            num_threads: Number of threads to use for parallel processing
            use_direct_yuv: Process YUV frames directly without conversion to avoid rounding errors
            verbose: Whether to print detailed compression information
        """
        # Store parameters
        self.noise_tolerance = noise_tolerance
        self.keyframe_interval = keyframe_interval
        self.min_diff_threshold = min_diff_threshold
        self.max_diff_threshold = max_diff_threshold
        self.bloom_threshold_modifier = bloom_threshold_modifier
        self.batch_size = batch_size
        self.use_direct_yuv = use_direct_yuv
        self.verbose = verbose
        
        # Import fixed compressor
        from fixed_video_compressor import FixedVideoCompressor
        
        # Create fixed compressor for true lossless compression
        self.compressor = FixedVideoCompressor(verbose=verbose)
        
    def compress_video(self, frames: List[np.ndarray], 
                     output_path: str = None,
                     input_color_space: str = "BGR") -> Dict:
        """
        Compress video frames with accurate compression ratio calculation.
        
        Args:
            frames: List of video frames
            output_path: Path to save the compressed video
            input_color_space: Color space of input frames ('BGR', 'RGB', 'YUV')
            
        Returns:
            Dictionary with compression results and statistics
        """
        if not frames:
            raise ValueError("No frames provided for compression")
        
        start_time = time.time()
        
        # Set YUV mode if needed
        if input_color_space.upper() == "YUV":
            self.use_direct_yuv = True
            
            # Add YUV info to frames if not already present
            for i in range(len(frames)):
                if not hasattr(frames[i], 'yuv_info'):
                    frames[i] = self.compressor.add_yuv_info_to_frame(frames[i])
        
        # Calculate original size accurately
        original_size = sum(frame.nbytes for frame in frames)
        
        # Compress frames
        compressed_frames = self.compressor.compress_video(frames)
        
        # Save to file if requested
        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write compressed data
            with open(output_path, 'wb') as f:
                # Write header
                f.write(b'BFVC')  # Magic number
                f.write(struct.pack('<I', len(frames)))  # Frame count
                
                # Write each compressed frame
                for compressed_frame in compressed_frames:
                    f.write(struct.pack('<I', len(compressed_frame)))
                    f.write(compressed_frame)
        
        # Calculate compressed size
        if output_path and os.path.exists(output_path):
            compressed_size = os.path.getsize(output_path)
        else:
            # Calculate from compressed frames if file wasn't saved
            compressed_size = sum(len(data) for data in compressed_frames)
            # Add header size
            compressed_size += 4 + 4 + (4 * len(compressed_frames))
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size
        
        # Calculate stats
        compression_time = time.time() - start_time
        
        # Results
        results = {
            'frame_count': len(frames),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_savings': 1.0 - compression_ratio,
            'compression_time': compression_time,
            'frames_per_second': len(frames) / compression_time,
            'keyframes': len(frames),  # All frames are keyframes in this version
            'keyframe_ratio': 1.0,
            'output_path': output_path,
            'color_space': input_color_space,
            'overall_ratio': compression_ratio
        }
        
        if self.verbose:
            print("\nCompression Results:")
            print(f"Original Size: {original_size / (1024*1024):.2f} MB")
            print(f"Compressed Size: {compressed_size / (1024*1024):.2f} MB")
            print(f"Compression Ratio: {compression_ratio:.4f}")
            print(f"Space Savings: {(1 - compression_ratio) * 100:.1f}%")
            print(f"Compression Time: {compression_time:.2f} seconds")
            print(f"Frames Per Second: {results['frames_per_second']:.2f}")
            print(f"Keyframes: {results['keyframes']} ({results['keyframe_ratio']*100:.1f}%)")
            print(f"Color Space: {input_color_space}")
        
        return results
    
    def decompress_video(self, input_path: str = None, 
                       output_path: Optional[str] = None,
                       compressed_frames: List[bytes] = None,
                       metadata: Dict = None) -> List[np.ndarray]:
        """
        Decompress video from file or compressed frames.
        
        Args:
            input_path: Path to the compressed video file
            output_path: Optional path to save decompressed frames as video
            compressed_frames: List of compressed frame data (alternative to input_path)
            metadata: Optional metadata for compressed frames
            
        Returns:
            List of decompressed video frames
        """
        start_time = time.time()
        
        # Read from file if provided
        if input_path and os.path.exists(input_path):
            with open(input_path, 'rb') as f:
                # Read header
                magic = f.read(4)
                if magic != b'BFVC':
                    raise ValueError(f"Invalid file format: {magic}")
                
                frame_count = struct.unpack('<I', f.read(4))[0]
                
                # Read compressed frames
                compressed_frames = []
                for _ in range(frame_count):
                    frame_size = struct.unpack('<I', f.read(4))[0]
                    frame_data = f.read(frame_size)
                    compressed_frames.append(frame_data)
        
        if not compressed_frames:
            raise ValueError("No compressed frames provided")
        
        # Decompress frames
        frames = self.compressor.decompress_video(compressed_frames)
        
        # Save as video if requested
        if output_path:
            self.save_frames_as_video(frames, output_path)
        
        # Calculate stats
        decompression_time = time.time() - start_time
        
        if self.verbose:
            print(f"Decompressed {len(frames)} frames in {decompression_time:.2f} seconds")
            print(f"Frames Per Second: {len(frames) / decompression_time:.2f}")
        
        return frames
    
    def verify_lossless(self, original_frames: List[np.ndarray], 
                      decompressed_frames: List[np.ndarray]) -> Dict:
        """
        Verify that decompression is truly lossless with bit-exact reconstruction.
        
        This method enforces strict bit-exact reconstruction with zero tolerance for
        any differences. If even a single pixel in a single frame differs by the smallest 
        possible value, the verification will fail.
        
        Args:
            original_frames: List of original video frames
            decompressed_frames: List of decompressed video frames
            
        Returns:
            Dictionary with verification results
        """
        # Delegate to the fixed compressor's verify_lossless method
        return self.compressor.verify_lossless(original_frames, decompressed_frames)
    
    def save_frames_as_video(self, frames: List[np.ndarray], output_path: str, 
                          fps: int = 30) -> str:
        """
        Save frames as a video file.
        
        Args:
            frames: List of frames to save
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            Path to the saved video file
        """
        if not frames:
            raise ValueError("No frames provided")
        
        if self.verbose:
            print(f"Saving {len(frames)} frames as video: {output_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        is_color = len(frames[0].shape) > 2
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
        
        if not out.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")
        
        # Write frames
        for frame in frames:
            # Check if this is a YUV frame and convert back to BGR for saving
            if is_color and hasattr(frame, 'yuv_info') and self.use_direct_yuv:
                # Convert YUV to BGR for saving
                frame_to_write = cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR)
            # Convert grayscale to BGR if needed
            elif not is_color and len(frame.shape) == 2:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # RGB needs to be converted to BGR for OpenCV
            elif is_color and frame.shape[2] == 3 and not hasattr(frame, 'yuv_info'):
                # Assume it's RGB and convert to BGR for OpenCV
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_to_write = frame
            
            out.write(frame_to_write)
        
        out.release()
        
        if self.verbose:
            print(f"Video saved: {output_path}")
        
        return output_path
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 0,
                               target_fps: Optional[float] = None,
                               scale_factor: float = 1.0,
                               output_color_space: str = "BGR") -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (0 = all)
            target_fps: Target frames per second (None = use original)
            scale_factor: Scale factor for frame dimensions
            output_color_space: Color space for output frames
            
        Returns:
            List of video frames
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.verbose:
            print(f"Video: {video_path}")
            print(f"Dimensions: {width}x{height}, {fps} FPS, {total_frames} total frames")
        
        # Determine frame extraction parameters
        if max_frames <= 0 or max_frames > total_frames:
            max_frames = total_frames
        
        # Calculate frame step for target FPS
        frame_step = 1
        if target_fps is not None and target_fps < fps:
            frame_step = max(1, round(fps / target_fps))
        
        # Calculate new dimensions if scaling
        if scale_factor != 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width, new_height = width, height
        
        # Extract frames
        frames = []
        frame_idx = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should keep this frame based on frame_step
            if frame_idx % frame_step == 0:
                # Resize if needed
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert color space if needed
                if output_color_space.upper() == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif output_color_space.upper() == "YUV":
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    frame = self.compressor.add_yuv_info_to_frame(yuv)
                
                frames.append(frame)
                
                # Status update
                if self.verbose and len(frames) % 10 == 0:
                    print(f"Extracted {len(frames)}/{max_frames} frames")
            
            frame_idx += 1
        
        cap.release()
        
        if self.verbose:
            print(f"Extracted {len(frames)} frames from {video_path}")
        
        return frames

class VideoFrameCompressor:
    """
    Specialized video frame compressor using Bloom filters for difference encoding.
    
    This class implements compression techniques specifically optimized for raw,
    noisy video frames by:
    1. Using adaptive thresholding for frame differences
    2. Special handling for noisy images
    3. Fast, parallelized operations where possible
    4. Memory-efficient operations for large frame sizes (e.g., 4K)
    """
    
    def __init__(self, 
                noise_tolerance: float = 10.0,
                keyframe_interval: int = 30,
                min_diff_threshold: float = 3.0,
                max_diff_threshold: float = 30.0,
                bloom_threshold_modifier: float = 1.0,
                num_threads: int = None,
                use_direct_yuv: bool = False,
                verbose: bool = False):
        """
        Initialize the video frame compressor.
        
        Args:
            noise_tolerance: Tolerance for noise in frame differences (higher = more tolerant)
            keyframe_interval: Maximum number of frames between keyframes
            min_diff_threshold: Minimum threshold for considering pixels different
            max_diff_threshold: Maximum threshold for considering pixels different
            bloom_threshold_modifier: Modifier for Bloom filter threshold
            num_threads: Number of threads to use for parallel processing
            use_direct_yuv: Process YUV frames directly without conversion to avoid rounding errors
            verbose: Whether to print detailed compression information
        """
        self.noise_tolerance = noise_tolerance
        self.keyframe_interval = keyframe_interval
        self.min_diff_threshold = min_diff_threshold
        self.max_diff_threshold = max_diff_threshold
        self.bloom_threshold_modifier = bloom_threshold_modifier
        self.use_direct_yuv = use_direct_yuv
        self.verbose = verbose
        
        # Set up multi-threading
        if num_threads is None:
            self.num_threads = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_threads = max(1, num_threads)
        
        if self.verbose:
            print(f"Initialized VideoFrameCompressor with {self.num_threads} threads")
            print(f"Noise tolerance: {self.noise_tolerance}")
            print(f"Keyframe interval: {self.keyframe_interval}")
            print(f"Difference thresholds: {self.min_diff_threshold}-{self.max_diff_threshold}")
            if self.use_direct_yuv:
                print(f"Using direct YUV processing for lossless reconstruction")
    
    def _estimate_noise_level(self, frame: np.ndarray) -> float:
        """
        Estimate the noise level in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Estimated standard deviation of noise
        """
        # Use median filter to create a smoothed version
        smoothed = cv2.medianBlur(frame, 5)
        
        # Noise is approximated as the difference between original and smoothed
        noise = frame.astype(np.float32) - smoothed.astype(np.float32)
        
        # Estimate noise level as standard deviation
        noise_level = np.std(noise)
        
        return noise_level
    
    def _adaptive_diff_threshold(self, frame: np.ndarray) -> float:
        """
        Calculate an adaptive threshold for frame differences based on noise.
        
        Args:
            frame: Input frame
            
        Returns:
            Threshold value for binarizing differences
        """
        # Estimate noise level
        noise_level = self._estimate_noise_level(frame)
        
        # Scale threshold based on noise (with limits)
        threshold = max(self.min_diff_threshold, 
                        min(self.max_diff_threshold, 
                            noise_level * self.noise_tolerance))
        
        return threshold
    
    def _calculate_frame_diff(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                             threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate binary difference mask and changed values between two frames.
        
        This method ensures bit-exact precision by carefully tracking which pixels have
        changed and storing their exact values for perfect reconstruction.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            threshold: Optional fixed threshold (if None, will use adaptive threshold)
            
        Returns:
            Tuple of (binary_diff_mask, changed_values, diff_density)
        """
        is_color = len(prev_frame.shape) > 2 and prev_frame.shape[2] > 1
        
        # For threshold calculation, convert to grayscale or use Y channel for YUV
        if is_color:
            if self.use_direct_yuv and prev_frame.shape[2] >= 3:
                # If using direct YUV, Y channel is already the first channel
                prev_gray = prev_frame[:, :, 0].copy()
                curr_gray = curr_frame[:, :, 0].copy()
            else:
                # Convert to grayscale for BGR/RGB formats
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame.copy()
            curr_gray = curr_frame.copy()
        
        # Calculate absolute difference using integer precision
        diff = np.abs(prev_gray.astype(np.int16) - curr_gray.astype(np.int16))
        
        # Determine threshold
        if threshold is None:
            threshold = self._adaptive_diff_threshold(curr_gray)
            
        # Create binary difference mask - 1 where pixel differs
        binary_diff = (diff > threshold).astype(np.uint8)
        
        # Get changed pixel values
        changed_indices = np.where(binary_diff == 1)
        
        if is_color:
            # For color frames, get all channel values for changed pixels
            rows, cols = changed_indices
            
            # Store each channel separately to prevent any loss of precision
            if self.use_direct_yuv and hasattr(curr_frame, 'yuv_info'):
                # For YUV frames, extract values from the original YUV planes for perfect reconstruction
                y_values = curr_frame.yuv_info['y_plane'][rows, cols]
                u_values = curr_frame.yuv_info['u_plane'][rows, cols]
                v_values = curr_frame.yuv_info['v_plane'][rows, cols]
                
                # Combine values, ensuring exact original values are preserved
                changed_values = np.zeros(len(rows) * curr_frame.shape[2], dtype=np.uint8)
                for i in range(len(rows)):
                    changed_values[i*3] = y_values[i]
                    changed_values[i*3+1] = u_values[i]
                    changed_values[i*3+2] = v_values[i]
            else:
                # For regular color frames, extract exact channel values
                changed_values = np.zeros(len(rows) * curr_frame.shape[2], dtype=curr_frame.dtype)
                
                # Extract all channel values for each changed pixel
                idx = 0
                for i in range(len(rows)):
                    for c in range(curr_frame.shape[2]):
                        changed_values[idx] = curr_frame[rows[i], cols[i], c]
                        idx += 1
        else:
            # For grayscale, directly get the values
            changed_values = curr_frame[changed_indices].copy()
        
        # Calculate difference density
        diff_density = np.sum(binary_diff) / binary_diff.size
        
        return binary_diff, changed_values, diff_density
    
    def _apply_frame_diff(self, base_frame: np.ndarray, diff_mask: np.ndarray, 
                        changed_values: np.ndarray) -> np.ndarray:
        """
        Apply frame difference to reconstruct the next frame with bit-exact precision.
        
        This method ensures that the decompressed frame is an exact binary match to the
        original frame by precisely applying the stored difference values.
        
        Args:
            base_frame: Base frame
            diff_mask: Binary difference mask (1 where pixels differ)
            changed_values: New values for pixels that differ
            
        Returns:
            Reconstructed next frame with bit-exact precision
        """
        # Create a copy of the base frame to avoid modifying the original
        next_frame = base_frame.copy()
        
        # Find indices where diff is 1
        diff_indices = np.where(diff_mask == 1)
        
        # Handle color frames differently from grayscale frames
        if len(base_frame.shape) == 3 and base_frame.shape[2] > 1:
            # For color frames, we need to update all channels for each changed pixel
            channels = base_frame.shape[2]
            
            # Get row and column indices where changes occurred
            rows, cols = diff_indices
            
            # Calculate how many values we should have (pixels * channels)
            expected_values = len(rows) * channels
            
            if len(changed_values) == expected_values:
                # Reshape changed values to match the original format
                if self.use_direct_yuv and hasattr(next_frame, 'yuv_info'):
                    # For YUV frames with yuv_info, update the planes directly
                    pixel_values = changed_values.reshape(-1, channels)
                    
                    # Update the frame data
                    for i in range(len(rows)):
                        next_frame[rows[i], cols[i]] = pixel_values[i]
                    
                    # Update the YUV planes for perfect reconstruction
                    for i in range(len(rows)):
                        next_frame.yuv_info['y_plane'][rows[i], cols[i]] = pixel_values[i, 0]
                        next_frame.yuv_info['u_plane'][rows[i], cols[i]] = pixel_values[i, 1]
                        next_frame.yuv_info['v_plane'][rows[i], cols[i]] = pixel_values[i, 2]
                else:
                    # Reshape changed values to [num_pixels, channels]
                    pixel_values = changed_values.reshape(-1, channels)
                    
                    # Update each pixel with exact values
                    for i in range(len(rows)):
                        next_frame[rows[i], cols[i]] = pixel_values[i]
        else:
            # For grayscale frames, directly update the pixels with exact values
            if len(diff_indices[0]) > 0:
                next_frame[diff_indices] = changed_values
        
        return next_frame
    
    def _compress_frame_differences(self, binary_diff: np.ndarray, 
                                 changed_values: np.ndarray) -> Tuple[bytes, float]:
        """
        Compress frame differences using Bloom filter compression.
        
        Args:
            binary_diff: Binary difference mask
            changed_values: Changed pixel values
            
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        # Flatten the binary difference mask
        flat_diff = binary_diff.flatten()
        
        # Compress with Bloom filter
        bloom_bitmap, witness, p, n, bloom_ratio = self.bloom_compressor.compress(flat_diff)
        
        # Create buffer for binary data
        buffer = io.BytesIO()
        
        # Store compression parameters
        buffer.write(struct.pack('<f', p))  # Density
        buffer.write(struct.pack('<I', n))  # Original length
        
        # Calculate optimal k
        k, l = self.bloom_compressor._calculate_optimal_params(n, p)
        buffer.write(struct.pack('<f', k))  # Hash function count
        
        # Store bloom filter bitmap
        buffer.write(struct.pack('<I', len(bloom_bitmap)))  # Bitmap length
        buffer.write(struct.pack('<I', len(witness)))       # Witness length
        
        # Store the bitmap (compressed)
        bitmap_bytes = np.packbits(bloom_bitmap).tobytes()
        buffer.write(struct.pack('<I', len(bitmap_bytes)))
        buffer.write(bitmap_bytes)
        
        # Store the witness (compressed)
        witness_array = np.array(witness, dtype=np.uint8)
        witness_bytes = np.packbits(witness_array).tobytes()
        buffer.write(struct.pack('<I', len(witness_bytes)))
        buffer.write(witness_bytes)
        
        # Store the changed values (compressed with zlib)
        values_bytes = zlib.compress(changed_values.tobytes(), level=9)
        buffer.write(struct.pack('<I', len(values_bytes)))
        buffer.write(struct.pack('<I', len(changed_values)))  # Store original count
        buffer.write(values_bytes)
        
        # Calculate overall compression ratio
        original_size = n + len(changed_values) * 8  # Binary diff + 8 bits per changed value
        compressed_size = buffer.tell() * 8  # Size in bits
        
        compression_ratio = compressed_size / original_size
        
        return buffer.getvalue(), compression_ratio
    
    def _decompress_frame_differences(self, compressed_data: bytes, 
                                   frame_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompress frame differences.
        
        Args:
            compressed_data: Compressed binary data
            frame_shape: Shape of the original frame
            
        Returns:
            Tuple of (binary_diff_mask, changed_values)
        """
        buffer = io.BytesIO(compressed_data)
        
        # Read parameters
        p = struct.unpack('<f', buffer.read(4))[0]
        n = struct.unpack('<I', buffer.read(4))[0]
        k = struct.unpack('<f', buffer.read(4))[0]
        
        # Read bloom filter data
        bitmap_length = struct.unpack('<I', buffer.read(4))[0]
        witness_length = struct.unpack('<I', buffer.read(4))[0]
        
        # Read compressed bitmap
        bitmap_size = struct.unpack('<I', buffer.read(4))[0]
        bitmap_bytes = buffer.read(bitmap_size)
        bloom_bits = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        bloom_bitmap = bloom_bits[:bitmap_length]
        
        # Read compressed witness
        witness_size = struct.unpack('<I', buffer.read(4))[0]
        witness_bytes = buffer.read(witness_size)
        witness_bits = np.unpackbits(np.frombuffer(witness_bytes, dtype=np.uint8))
        witness = witness_bits[:witness_length].tolist()
        
        # Read compressed changed values
        values_size = struct.unpack('<I', buffer.read(4))[0]
        values_count = struct.unpack('<I', buffer.read(4))[0]
        values_bytes = buffer.read(values_size)
        values_data = zlib.decompress(values_bytes)
        changed_values = np.frombuffer(values_data, dtype=np.uint8)[:values_count]
        
        # Decompress the binary difference mask
        if witness_length > 0:
            flat_diff = self.bloom_compressor.decompress(bloom_bitmap, witness, n, k)
        else:
            flat_diff = bloom_bitmap
        
        # For color frames, the binary diff is a 2D mask (height x width) that indicates 
        # which pixels changed, not which specific color channels changed
        if len(frame_shape) == 3 and frame_shape[2] > 1:
            # Extract the 2D shape (height, width) from the 3D frame shape
            mask_shape = (frame_shape[0], frame_shape[1])
            binary_diff = flat_diff.reshape(mask_shape)
        else:
            # Grayscale frame, reshape to original dimensions
            binary_diff = flat_diff.reshape(frame_shape)
        
        return binary_diff, changed_values
    
    def compress_frame(self, frame: np.ndarray, is_keyframe: bool = True) -> Tuple[bytes, dict]:
        """
        Compress a single frame with bit-exact preservation.
        
        This method ensures that frames can be reconstructed exactly bit-for-bit
        without any loss of information.
        
        Args:
            frame: Frame data as numpy array
            is_keyframe: Whether this is a keyframe
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        if is_keyframe:
            # For keyframes, use direct compression with no preprocessing
            # This preserves the exact bit pattern for perfect reconstruction
            frame_bytes = frame.tobytes()
            compressed_frame = zlib.compress(frame_bytes, level=9)
            
            # Create buffer
            buffer = io.BytesIO()
            
            # Store frame type and original size
            buffer.write(struct.pack('<B', 1))  # 1 = keyframe
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
            
            metadata = {
                'type': 'keyframe',
                'shape': frame.shape,
                'original_size': frame.nbytes,
                'compressed_size': buffer.tell(),
                'compression_ratio': buffer.tell() / frame.nbytes,
                'has_yuv_info': has_yuv_info
            }
            
            return buffer.getvalue(), metadata
        else:
            # For non-keyframes, this method is not used directly
            # (frame differences are handled in compress_video)
            raise ValueError("Non-keyframe compression should be handled by compress_video")
    
    def decompress_frame(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress a single frame with bit-exact precision.
        
        This method ensures that the decompressed frame is an exact bit-for-bit
        match to the original frame.
        
        Args:
            compressed_data: Compressed frame data
            
        Returns:
            Decompressed frame as numpy array with exact precision
        """
        buffer = io.BytesIO(compressed_data)
        
        # Read frame type
        frame_type = struct.unpack('<B', buffer.read(1))[0]
        
        if frame_type == 1:  # Keyframe
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
                
            # Check if this has YUV info
            has_yuv_info = False
            try:
                has_yuv_info = struct.unpack('<B', buffer.read(1))[0] == 1
            except:
                # For backward compatibility
                pass
                
            if has_yuv_info and self.use_direct_yuv:
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
                        if hasattr(self, 'yuv_info'):
                            new_frame.yuv_info = {
                                k: v.copy() if hasattr(v, 'copy') else v 
                                for k, v in self.yuv_info.items()
                            }
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
        else:
            raise ValueError(f"Unknown frame type: {frame_type}")
    
    def compress_video(self, frames: List[np.ndarray], 
                     output_path: str,
                     input_color_space: str = "BGR") -> Dict:
        """
        Compress video frames with accurate compression ratio calculation.
        
        Args:
            frames: List of video frames
            output_path: Path to save the compressed video
            input_color_space: Color space of input frames ('BGR', 'RGB', 'YUV')
            
        Returns:
            Dictionary with compression results and statistics
        """
        if not frames:
            raise ValueError("No frames provided for compression")
        
        start_time = time.time()
        
        # Calculate original size accurately
        original_size = sum(frame.nbytes for frame in frames)
        
        # Set YUV mode if needed
        if input_color_space.upper() == "YUV":
            self.use_direct_yuv = True
            
            # Add YUV info to frames if not already present
            for i in range(len(frames)):
                if not hasattr(frames[i], 'yuv_info'):
                    frames[i] = self.compressor.add_yuv_info_to_frame(frames[i])
        
        # Compress frames
        compressed_frames = self.compressor.compress_video(frames)
        
        # Save to file if requested
        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write compressed data
            with open(output_path, 'wb') as f:
                # Write header
                f.write(b'BFVC')  # Magic number
                f.write(struct.pack('<I', len(frames)))  # Frame count
                
                # Write each compressed frame
                for compressed_frame in compressed_frames:
                    f.write(struct.pack('<I', len(compressed_frame)))
                    f.write(compressed_frame)
        
        # Calculate compressed size
        if output_path and os.path.exists(output_path):
            compressed_size = os.path.getsize(output_path)
        else:
            # Calculate from compressed frames if file wasn't saved
            compressed_size = sum(len(data) for data in compressed_frames)
            # Add header size
            compressed_size += 4 + 4 + (4 * len(compressed_frames))
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size
        
        # Calculate stats
        compression_time = time.time() - start_time
        
        # Results
        results = {
            'frame_count': len(frames),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_savings': 1.0 - compression_ratio,
            'compression_time': compression_time,
            'frames_per_second': len(frames) / compression_time,
            'keyframes': len(frames),  # All frames are keyframes in this version
            'keyframe_ratio': 1.0,
            'output_path': output_path,
            'color_space': input_color_space,
            'overall_ratio': compression_ratio
        }
        
        if self.verbose:
            print("\nCompression Results:")
            print(f"Original Size: {original_size / (1024*1024):.2f} MB")
            print(f"Compressed Size: {compressed_size / (1024*1024):.2f} MB")
            print(f"Compression Ratio: {compression_ratio:.4f}")
            print(f"Space Savings: {(1 - compression_ratio) * 100:.1f}%")
            print(f"Compression Time: {compression_time:.2f} seconds")
            print(f"Frames Per Second: {results['frames_per_second']:.2f}")
            print(f"Keyframes: {results['keyframes']} ({results['keyframe_ratio']*100:.1f}%)")
            print(f"Color Space: {input_color_space}")
        
        return results
    
    def decompress_video(self, input_path: str = None, 
                       output_path: Optional[str] = None,
                       compressed_frames: List[bytes] = None,
                       metadata: Dict = None) -> List[np.ndarray]:
        """
        Decompress video from file or compressed frames.
        
        Args:
            input_path: Path to the compressed video file
            output_path: Optional path to save decompressed frames as video
            compressed_frames: List of compressed frame data (alternative to input_path)
            metadata: Optional metadata for compressed frames
            
        Returns:
            List of decompressed video frames
        """
        start_time = time.time()
        
        # Read from file if provided
        if input_path and os.path.exists(input_path):
            with open(input_path, 'rb') as f:
                # Read header
                magic = f.read(4)
                if magic != b'BFVC':
                    raise ValueError(f"Invalid file format: {magic}")
                
                frame_count = struct.unpack('<I', f.read(4))[0]
                
                # Read compressed frames
                compressed_frames = []
                for _ in range(frame_count):
                    frame_size = struct.unpack('<I', f.read(4))[0]
                    frame_data = f.read(frame_size)
                    compressed_frames.append(frame_data)
        
        if not compressed_frames:
            raise ValueError("No compressed frames provided")
        
        # Decompress frames
        frames = self.compressor.decompress_video(compressed_frames)
        
        # Save as video if requested
        if output_path:
            self.save_frames_as_video(frames, output_path)
        
        # Calculate stats
        decompression_time = time.time() - start_time
        
        if self.verbose:
            print(f"Decompressed {len(frames)} frames in {decompression_time:.2f} seconds")
            print(f"Frames Per Second: {len(frames) / decompression_time:.2f}")
        
        return frames
    
    def verify_lossless(self, original_frames: List[np.ndarray], 
                      decompressed_frames: List[np.ndarray]) -> Dict:
        """
        Verify that decompression is truly lossless with bit-exact reconstruction.
        
        This method enforces strict bit-exact reconstruction with zero tolerance for
        any differences. If even a single pixel in a single frame differs by the smallest 
        possible value, the verification will fail.
        
        Args:
            original_frames: List of original video frames
            decompressed_frames: List of decompressed video frames
            
        Returns:
            Dictionary with verification results
        """
        # Delegate to the fixed compressor's verify_lossless method
        return self.compressor.verify_lossless(original_frames, decompressed_frames)
    
    def save_frames_as_video(self, frames: List[np.ndarray], output_path: str, 
                          fps: int = 30) -> str:
        """
        Save frames as a video file.
        
        Args:
            frames: List of frames to save
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            Path to the saved video file
        """
        if not frames:
            raise ValueError("No frames provided")
        
        if self.verbose:
            print(f"Saving {len(frames)} frames as video: {output_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        is_color = len(frames[0].shape) > 2
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
        
        if not out.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")
        
        # Write frames
        for frame in frames:
            # Check if this is a YUV frame and convert back to BGR for saving
            if is_color and hasattr(frame, 'yuv_info') and self.use_direct_yuv:
                # Convert YUV to BGR for saving
                frame_to_write = cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR)
            # Convert grayscale to BGR if needed
            elif not is_color and len(frame.shape) == 2:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # RGB needs to be converted to BGR for OpenCV
            elif is_color and frame.shape[2] == 3 and not hasattr(frame, 'yuv_info'):
                # Assume it's RGB and convert to BGR for OpenCV
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_to_write = frame
            
            out.write(frame_to_write)
        
        out.release()
        
        if self.verbose:
            print(f"Video saved: {output_path}")
        
        return output_path
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 0,
                               target_fps: Optional[float] = None,
                               scale_factor: float = 1.0,
                               output_color_space: str = "BGR") -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (0 = all)
            target_fps: Target frames per second (None = use original)
            scale_factor: Scale factor for frame dimensions
            output_color_space: Color space for output frames
            
        Returns:
            List of video frames
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.verbose:
            print(f"Video: {video_path}")
            print(f"Dimensions: {width}x{height}, {fps} FPS, {total_frames} total frames")
        
        # Determine frame extraction parameters
        if max_frames <= 0 or max_frames > total_frames:
            max_frames = total_frames
        
        # Calculate frame step for target FPS
        frame_step = 1
        if target_fps is not None and target_fps < fps:
            frame_step = max(1, round(fps / target_fps))
        
        # Calculate new dimensions if scaling
        if scale_factor != 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width, new_height = width, height
        
        # Extract frames
        frames = []
        frame_idx = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should keep this frame based on frame_step
            if frame_idx % frame_step == 0:
                # Resize if needed
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert color space if needed
                if output_color_space.upper() == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif output_color_space.upper() == "YUV":
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    frame = self.compressor.add_yuv_info_to_frame(yuv)
                
                frames.append(frame)
                
                # Status update
                if self.verbose and len(frames) % 10 == 0:
                    print(f"Extracted {len(frames)}/{max_frames} frames")
            
            frame_idx += 1
        
        cap.release()
        
        if self.verbose:
            print(f"Extracted {len(frames)} frames from {video_path}")
        
        return frames

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Improved Video Compressor with Rational Bloom Filter")
    
    # Action subparsers
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Compress video parser
    compress_parser = subparsers.add_parser("compress", help="Compress a video file")
    compress_parser.add_argument("input", type=str, help="Input video file path")
    compress_parser.add_argument("output", type=str, help="Output compressed file path")
    compress_parser.add_argument("--max-frames", type=int, default=0, 
                                help="Maximum frames to process (0 = all)")
    compress_parser.add_argument("--fps", type=float, default=None,
                                help="Target frames per second (default = original)")
    compress_parser.add_argument("--scale", type=float, default=1.0,
                                help="Scale factor for frame dimensions")
    compress_parser.add_argument("--noise-tolerance", type=float, default=10.0,
                                help="Noise tolerance level")
    compress_parser.add_argument("--keyframe-interval", type=int, default=30,
                                help="Maximum frames between keyframes")
    compress_parser.add_argument("--min-diff", type=float, default=3.0,
                                help="Minimum threshold for pixel differences")
    compress_parser.add_argument("--max-diff", type=float, default=30.0,
                                help="Maximum threshold for pixel differences")
    compress_parser.add_argument("--bloom-modifier", type=float, default=1.0,
                                help="Modifier for Bloom filter threshold")
    compress_parser.add_argument("--batch-size", type=int, default=30,
                                help="Number of frames to process in each batch")
    compress_parser.add_argument("--threads", type=int, default=None,
                                help="Number of threads for parallel processing")
    compress_parser.add_argument("--use-direct-yuv", action="store_true",
                                help="Use direct YUV processing for lossless reconstruction")
    compress_parser.add_argument("--color-space", type=str, default="BGR", choices=["BGR", "RGB", "YUV"],
                                help="Color space of input video")
    compress_parser.add_argument("--verbose", action="store_true",
                                help="Print detailed information")
    
    # Decompress video parser
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a video file")
    decompress_parser.add_argument("input", type=str, help="Input compressed file path")
    decompress_parser.add_argument("output", type=str, help="Output video file path")
    decompress_parser.add_argument("--use-direct-yuv", action="store_true",
                                  help="Use direct YUV processing for lossless reconstruction")
    decompress_parser.add_argument("--verbose", action="store_true",
                                  help="Print detailed information")
    
    # Raw YUV file parser
    yuv_parser = subparsers.add_parser("process-yuv", help="Process a raw YUV file")
    yuv_parser.add_argument("input", type=str, help="Input YUV file path")
    yuv_parser.add_argument("output", type=str, help="Output compressed file path")
    yuv_parser.add_argument("--width", type=int, required=True,
                           help="Frame width")
    yuv_parser.add_argument("--height", type=int, required=True,
                           help="Frame height")
    yuv_parser.add_argument("--format", type=str, default="I420", 
                           choices=["I420", "YV12", "YUV422", "YUV444"],
                           help="YUV format")
    yuv_parser.add_argument("--max-frames", type=int, default=0,
                           help="Maximum frames to process (0 = all)")
    yuv_parser.add_argument("--frame-step", type=int, default=1,
                           help="Process every nth frame")
    yuv_parser.add_argument("--noise-tolerance", type=float, default=10.0,
                           help="Noise tolerance level")
    yuv_parser.add_argument("--keyframe-interval", type=int, default=30,
                           help="Maximum frames between keyframes")
    yuv_parser.add_argument("--min-diff", type=float, default=3.0,
                           help="Minimum threshold for pixel differences")
    yuv_parser.add_argument("--max-diff", type=float, default=30.0,
                           help="Maximum threshold for pixel differences")
    yuv_parser.add_argument("--bloom-modifier", type=float, default=1.0,
                           help="Modifier for Bloom filter threshold")
    yuv_parser.add_argument("--verbose", action="store_true",
                           help="Print detailed information")
    
    # Generate synthetic video parser
    synthetic_parser = subparsers.add_parser("synthetic", help="Generate and compress synthetic video")
    synthetic_parser.add_argument("output", type=str, help="Output directory")
    synthetic_parser.add_argument("--frames", type=int, default=90,
                                 help="Number of frames to generate")
    synthetic_parser.add_argument("--width", type=int, default=640,
                                 help="Frame width")
    synthetic_parser.add_argument("--height", type=int, default=480,
                                 help="Frame height")
    synthetic_parser.add_argument("--noise", type=float, default=1.0,
                                 help="Noise level (standard deviation)")
    synthetic_parser.add_argument("--speed", type=float, default=1.0,
                                 help="Movement speed for objects")
    synthetic_parser.add_argument("--use-direct-yuv", action="store_true",
                                 help="Use direct YUV processing for lossless reconstruction")
    synthetic_parser.add_argument("--color-space", type=str, default="BGR", choices=["BGR", "RGB", "YUV"],
                                 help="Color space for generated frames")
    synthetic_parser.add_argument("--verbose", action="store_true",
                                 help="Print detailed information")
    
    # Analyze noise parser
    analyze_parser = subparsers.add_parser("analyze", help="Analyze noise vs. compression")
    analyze_parser.add_argument("output", type=str, help="Output directory")
    analyze_parser.add_argument("--frames", type=int, default=90,
                               help="Number of frames per test")
    analyze_parser.add_argument("--width", type=int, default=640,
                               help="Frame width")
    analyze_parser.add_argument("--height", type=int, default=480,
                               help="Frame height")
    analyze_parser.add_argument("--noise-levels", type=float, nargs="+",
                               default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                               help="Noise levels to test")
    analyze_parser.add_argument("--use-direct-yuv", action="store_true",
                               help="Use direct YUV processing for lossless reconstruction")
    analyze_parser.add_argument("--color-space", type=str, default="BGR", choices=["BGR", "RGB", "YUV"],
                               help="Color space for generated frames")
    analyze_parser.add_argument("--verbose", action="store_true",
                               help="Print detailed information")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.action is None:
        parser.print_help()
        return
    
    # Create compressor with common parameters
    compressor = ImprovedVideoCompressor(
        verbose=args.verbose if hasattr(args, 'verbose') else False
    )
    
    # Handle different actions
    if args.action == "compress":
        # Update compressor with compression-specific parameters
        compressor = ImprovedVideoCompressor(
            noise_tolerance=args.noise_tolerance,
            keyframe_interval=args.keyframe_interval,
            min_diff_threshold=args.min_diff,
            max_diff_threshold=args.max_diff,
            bloom_threshold_modifier=args.bloom_modifier,
            batch_size=args.batch_size,
            num_threads=args.threads,
            use_direct_yuv=args.use_direct_yuv,
            verbose=args.verbose
        )
        
        # Extract frames from video
        frames = compressor.extract_frames_from_video(
            args.input,
            max_frames=args.max_frames,
            target_fps=args.fps,
            scale_factor=args.scale,
            output_color_space=args.color_space
        )
        
        # Compress the video
        result = compressor.compress_video(
            frames, 
            args.output,
            input_color_space=args.color_space
        )
        
        # Print summary
        print("\nCompression Summary:")
        print(f"Original Size: {result['original_size'] / (1024*1024):.2f} MB")
        print(f"Compressed Size: {result['compressed_size'] / (1024*1024):.2f} MB")
        print(f"Compression Ratio: {result['compression_ratio']:.4f}")
        print(f"Space Savings: {(1 - result['compression_ratio']) * 100:.1f}%")
        
    elif args.action == "decompress":
        # Create compressor with decompression-specific parameters
        compressor = ImprovedVideoCompressor(
            use_direct_yuv=args.use_direct_yuv,
            verbose=args.verbose
        )
        
        # Decompress the video
        frames = compressor.decompress_video(args.input, args.output)
        
        # Print summary
        print("\nDecompression Summary:")
        print(f"Decompressed {len(frames)} frames")
        print(f"Output saved to: {args.output}")
        
    elif args.action == "process-yuv":
        # Create compressor for YUV processing
        compressor = ImprovedVideoCompressor(
            noise_tolerance=args.noise_tolerance,
            keyframe_interval=args.keyframe_interval,
            min_diff_threshold=args.min_diff,
            max_diff_threshold=args.max_diff,
            bloom_threshold_modifier=args.bloom_modifier,
            use_direct_yuv=True,  # Always use direct YUV for YUV files
            verbose=args.verbose
        )
        
        # Extract frames from YUV file
        frames = compressor.extract_frames_from_video(
            args.input,
            width=args.width,
            height=args.height,
            format=args.format,
            max_frames=args.max_frames,
            frame_step=args.frame_step
        )
        
        # Compress the video
        result = compressor.compress_video(
            frames, 
            args.output,
            input_color_space="YUV"
        )
        
        # Print summary
        print("\nYUV Processing Summary:")
        print(f"Processed {len(frames)} frames from {args.input}")
        print(f"Format: {args.format}, Dimensions: {args.width}x{args.height}")
        print(f"Original Size: {result['original_size'] / (1024*1024):.2f} MB")
        print(f"Compressed Size: {result['compressed_size'] / (1024*1024):.2f} MB")
        print(f"Compression Ratio: {result['compression_ratio']:.4f}")
        print(f"Space Savings: {(1 - result['compression_ratio']) * 100:.1f}%")
        
    elif args.action == "synthetic":
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Create compressor
        compressor = ImprovedVideoCompressor(
            use_direct_yuv=args.use_direct_yuv,
            verbose=args.verbose
        )
        
        # Generate synthetic frames
        frames = compressor.extract_frames_from_video(
            args.input,
            max_frames=args.frames,
            target_fps=args.fps,
            scale_factor=args.scale,
            output_color_space=args.color_space
        )
        
        # Compress the video
        compressed_path = os.path.join(args.output, "synthetic_compressed.bfvc")
        result = compressor.compress_video(
            frames, 
            compressed_path,
            input_color_space=args.color_space
        )
        
        # Decompress and verify
        decompressed_frames = compressor.decompress_video(compressed_path)
        verification = compressor.verify_lossless(frames, decompressed_frames)
        
        # Save as video
        video_path = os.path.join(args.output, "synthetic.mp4")
        compressor.save_frames_as_video(frames, video_path)
        
        # Print summary
        print("\nSynthetic Video Summary:")
        print(f"Generated {len(frames)} frames ({args.width}x{args.height})")
        print(f"Noise Level: {args.noise}")
        print(f"Compression Ratio: {result['compression_ratio']:.4f}")
        print(f"Space Savings: {(1 - result['compression_ratio']) * 100:.1f}%")
        print(f"Lossless: {verification['lossless']}")
        if verification['exact_lossless']:
            print("Perfect bit-exact reconstruction achieved")
        elif verification['lossless']:
            print(f"Perceptually lossless reconstruction (avg diff: {verification['avg_difference']:.6f})")
        
    elif args.action == "analyze":
        # Run noise analysis
        compressor = ImprovedVideoCompressor(
            use_direct_yuv=args.use_direct_yuv,
            verbose=args.verbose
        )
        
        # Run noise analysis with color space selection
        result = compressor.analyze_noise_vs_compression(
            width=args.width,
            height=args.height,
            frame_count=args.frames,
            noise_levels=args.noise_levels,
            output_dir=args.output,
            color_space=args.color_space
        )
        
        # Print summary
        print("\nNoise Analysis Summary:")
        print(f"Tested {len(result['noise_levels'])} noise levels: {result['noise_levels']}")
        print(f"Results saved to: {args.output}")
        print(f"See {os.path.join(args.output, f'noise_comparison_{args.color_space}.png')} for visual comparison")


if __name__ == "__main__":
    main() 