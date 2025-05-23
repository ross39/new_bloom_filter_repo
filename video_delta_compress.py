import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import gzip
from bloom_compress import BloomFilterCompressor
from typing import List, Tuple, Dict
import imageio
import io
import struct


class VideoDeltaCompressor:
    """
    Compress video by applying Bloom filter compression to frame differences.
    """
    
    def __init__(self):
        """Initialize the compressor with a Bloom filter compressor."""
        self.bloom_compressor = BloomFilterCompressor()
        
    def _calculate_frame_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculate binary difference between two frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            Binary difference (1 where pixels differ, 0 where they're the same)
        """
        # Ensure frames are in the same format
        if len(frame1.shape) == 3 and frame1.shape[2] == 3:
            # Convert color frames to grayscale for simplicity
            frame1_gray = np.mean(frame1, axis=2).astype(np.uint8)
            frame2_gray = np.mean(frame2, axis=2).astype(np.uint8)
        else:
            frame1_gray = frame1
            frame2_gray = frame2
        
        # Calculate absolute difference
        diff = np.abs(frame1_gray.astype(np.int16) - frame2_gray.astype(np.int16))
        
        # Binarize the difference (1 where pixels differ significantly)
        threshold = 10  # Adjust based on noise tolerance needed
        binary_diff = (diff > threshold).astype(np.uint8)
        
        return binary_diff
    
    def _apply_frame_diff(self, base_frame: np.ndarray, diff: np.ndarray, 
                         changed_values: np.ndarray) -> np.ndarray:
        """
        Apply frame difference to reconstruct the next frame.
        
        Args:
            base_frame: Base frame
            diff: Binary difference mask (1 where pixels differ)
            changed_values: New values for pixels that differ
            
        Returns:
            Reconstructed next frame
        """
        # Create a copy of the base frame
        next_frame = base_frame.copy()
        
        # Find indices where diff is 1
        diff_indices = np.where(diff == 1)
        
        # Update those pixels with the new values
        for i in range(len(diff_indices[0])):
            y, x = diff_indices[0][i], diff_indices[1][i]
            next_frame[y, x] = changed_values[i]
            
        return next_frame
    
    def compress_video(self, frames: List[np.ndarray], 
                      output_path: str = None) -> Tuple[List[bytes], Dict]:
        """
        Compress a video using delta encoding with Bloom filter compression.
        
        Args:
            frames: List of video frames as numpy arrays
            output_path: Optional path to save compressed data
            
        Returns:
            Tuple of (compressed_frames, metadata)
        """
        if not frames:
            return [], {}
            
        # Store the first frame as-is (using simple compression)
        first_frame = frames[0]
        first_frame_bytes = self._compress_frame(first_frame)
        
        compressed_frames = [first_frame_bytes]
        metadata = {
            'frame_count': len(frames),
            'frame_shape': frames[0].shape,
            'keyframes': [0],  # Track which frames are keyframes (not delta)
            'compression_ratios': [],
            'densities': []
        }
        
        prev_frame = first_frame
        total_orig_size = 0
        total_comp_size = 0
        
        print(f"Compressing {len(frames)} frames")
        print("-" * 40)
        print("Frame | Diff Density | Bloom Ratio | Gzip Ratio")
        print("------|-------------|-------------|----------")
        
        # Process each subsequent frame
        for i in range(1, len(frames)):
            curr_frame = frames[i]
            
            # Calculate the binary difference mask
            diff_mask = self._calculate_frame_diff(prev_frame, curr_frame)
            
            # Calculate the density of differences
            diff_density = np.sum(diff_mask) / diff_mask.size
            metadata['densities'].append(diff_density)
            
            # Store changed pixel values
            changed_indices = np.where(diff_mask == 1)
            changed_values = curr_frame[changed_indices]
            
            # Flatten the diff mask for compression
            flat_diff = diff_mask.flatten()
            
            # Compress with Bloom filter
            start_time = time.time()
            bloom_bitmap, witness, p, n, bloom_ratio = self.bloom_compressor.compress(flat_diff)
            
            # Compress the same data with gzip for comparison
            gzip_compressed = gzip.compress(np.packbits(flat_diff).tobytes())
            gzip_ratio = len(gzip_compressed) / (flat_diff.size / 8)
            
            print(f"{i:5} | {diff_density:.6f} | {bloom_ratio:.6f} | {gzip_ratio:.6f}")
            
            # If compression isn't effective, store as keyframe
            if bloom_ratio >= 0.9:  # Threshold can be adjusted
                frame_bytes = self._compress_frame(curr_frame)
                compressed_frames.append(frame_bytes)
                metadata['keyframes'].append(i)
            else:
                # Pack the diff, bloom filter, witness, and changed values
                k, _ = self.bloom_compressor._calculate_optimal_params(n, p)
                frame_data = self._pack_delta_frame(
                    bloom_bitmap, witness, p, n, k, changed_values)
                compressed_frames.append(frame_data)
                
            metadata['compression_ratios'].append(bloom_ratio)
            
            # Update for next iteration
            prev_frame = curr_frame
            
            # Update total sizes (in bits)
            orig_frame_bits = curr_frame.nbytes * 8
            comp_frame_bits = len(compressed_frames[-1]) * 8
            total_orig_size += orig_frame_bits
            total_comp_size += comp_frame_bits
        
        # Calculate overall compression ratio
        overall_ratio = total_comp_size / total_orig_size
        metadata['overall_ratio'] = overall_ratio
        
        print("-" * 40)
        print(f"Overall compression ratio: {overall_ratio:.6f}")
        
        # Save compressed data if requested
        if output_path:
            with open(output_path, 'wb') as f:
                # Write header with metadata
                f.write(struct.pack('!I', metadata['frame_count']))
                
                height, width = metadata['frame_shape'][:2]
                f.write(struct.pack('!II', height, width))
                
                # Write keyframes count and indices
                f.write(struct.pack('!I', len(metadata['keyframes'])))
                for kf in metadata['keyframes']:
                    f.write(struct.pack('!I', kf))
                
                # Write each compressed frame
                for frame_data in compressed_frames:
                    f.write(struct.pack('!I', len(frame_data)))
                    f.write(frame_data)
        
        return compressed_frames, metadata
    
    def decompress_video(self, compressed_data: List[bytes], metadata: Dict = None, 
                        input_path: str = None, output_path: str = None) -> List[np.ndarray]:
        """
        Decompress a video using delta encoding with Bloom filter decompression.
        
        Args:
            compressed_data: List of compressed frame data
            metadata: Dictionary with video metadata
            input_path: Optional path to read compressed data from
            output_path: Optional path to save decompressed video
            
        Returns:
            List of decompressed video frames
        """
        # If input path is provided, read compressed data from file
        if input_path:
            compressed_data, metadata = self._read_compressed_video(input_path)
        
        if not compressed_data:
            return []
        
        # Get frame dimensions
        frame_shape = metadata['frame_shape']
        frame_count = metadata['frame_count']
        keyframes = metadata['keyframes']
        
        decompressed_frames = []
        frame_index_map = {}  # Maps decompressed index to original frame index
        
        # Decompress the first frame (always a keyframe)
        first_frame = self._decompress_frame(compressed_data[0], frame_shape)
        decompressed_frames.append(first_frame)
        frame_index_map[0] = 0
        
        prev_frame = first_frame
        keyframe_idx = 1  # Skip the first keyframe (already processed)
        
        print(f"Decompressing {len(compressed_data)} frames")
        print("-" * 40)
        
        # Process each subsequent frame
        for i in range(1, len(compressed_data)):
            # Check if this is a keyframe
            if keyframe_idx < len(keyframes) and i == keyframes[keyframe_idx]:
                # This is a keyframe, decompress directly
                curr_frame = self._decompress_frame(compressed_data[i], frame_shape)
                keyframe_idx += 1
            else:
                # This is a delta frame, apply diff to previous frame
                diff_data, changed_values = self._unpack_delta_frame(compressed_data[i])
                
                # Reshape diff to frame dimensions
                diff_mask = diff_data.reshape(frame_shape[:2])
                
                # Apply the diff to get the current frame
                curr_frame = self._apply_frame_diff(prev_frame, diff_mask, changed_values)
            
            decompressed_frames.append(curr_frame)
            frame_index_map[i] = len(decompressed_frames) - 1
            prev_frame = curr_frame
            
            if i % 10 == 0:
                print(f"Decompressed frame {i}/{len(compressed_data)}")
        
        print("-" * 40)
        print(f"Decompressed {len(decompressed_frames)} frames")
        
        # Save as video if output path is provided
        if output_path:
            self._save_video(decompressed_frames, output_path)
        
        return decompressed_frames
    
    def _compress_frame(self, frame: np.ndarray) -> bytes:
        """Compress a single frame using standard compression."""
        # For simplicity, we'll use PIL and save as PNG (which has its own compression)
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def _decompress_frame(self, frame_data: bytes, frame_shape: Tuple) -> np.ndarray:
        """Decompress a single frame."""
        buffer = io.BytesIO(frame_data)
        img = Image.open(buffer)
        return np.array(img)
    
    def _pack_delta_frame(self, bloom_bitmap: np.ndarray, witness: List, 
                         p: float, n: int, k: float, changed_values: np.ndarray) -> bytes:
        """Pack delta frame data into bytes."""
        buffer = io.BytesIO()
        
        # Write header for diff data
        buffer.write(struct.pack('!f', p))  # Density
        buffer.write(struct.pack('!I', n))  # Original length
        buffer.write(struct.pack('!f', k))  # Hash function count
        
        # Write Bloom filter bitmap size
        l = len(bloom_bitmap)
        buffer.write(struct.pack('!I', l))
        
        # Write witness size
        witness_len = len(witness)
        buffer.write(struct.pack('!I', witness_len))
        
        # Pack bloom filter bitmap into bytes
        bloom_bytes = np.packbits(bloom_bitmap)
        buffer.write(bloom_bytes.tobytes())
        
        # Pack witness data into bytes
        witness_array = np.array(witness, dtype=np.uint8)
        witness_bytes = np.packbits(witness_array)
        buffer.write(witness_bytes.tobytes())
        
        # Write changed values
        buffer.write(struct.pack('!I', len(changed_values)))
        buffer.write(changed_values.tobytes())
        
        return buffer.getvalue()
    
    def _unpack_delta_frame(self, frame_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack delta frame data from bytes."""
        buffer = io.BytesIO(frame_data)
        
        # Read header
        p = struct.unpack('!f', buffer.read(4))[0]
        n = struct.unpack('!I', buffer.read(4))[0]
        k = struct.unpack('!f', buffer.read(4))[0]
        
        # Read Bloom filter bitmap size
        l = struct.unpack('!I', buffer.read(4))[0]
        
        # Read witness size
        witness_len = struct.unpack('!I', buffer.read(4))[0]
        
        # Calculate bytes needed for bloom filter
        bloom_bytes_len = (l + 7) // 8  # Ceiling division by 8
        bloom_bytes = buffer.read(bloom_bytes_len)
        bloom_bits = np.unpackbits(np.frombuffer(bloom_bytes, dtype=np.uint8))
        bloom_bitmap = bloom_bits[:l]  # Trim to exact size
        
        # Calculate bytes needed for witness
        witness_bytes_len = (witness_len + 7) // 8  # Ceiling division by 8
        witness_bytes = buffer.read(witness_bytes_len)
        witness_bits = np.unpackbits(np.frombuffer(witness_bytes, dtype=np.uint8))
        witness = witness_bits[:witness_len].tolist()  # Trim to exact size
        
        # Decompress the difference data
        diff_data = self.bloom_compressor.decompress(bloom_bitmap, witness, n, k)
        
        # Read changed values
        changed_values_len = struct.unpack('!I', buffer.read(4))[0]
        changed_values = np.frombuffer(buffer.read(changed_values_len * np.dtype(np.uint8).itemsize), 
                                     dtype=np.uint8)
        
        return diff_data, changed_values
    
    def _read_compressed_video(self, input_path: str) -> Tuple[List[bytes], Dict]:
        """Read compressed video data from file."""
        with open(input_path, 'rb') as f:
            # Read header with metadata
            frame_count = struct.unpack('!I', f.read(4))[0]
            height, width = struct.unpack('!II', f.read(8))
            
            keyframes_count = struct.unpack('!I', f.read(4))[0]
            keyframes = []
            for _ in range(keyframes_count):
                keyframes.append(struct.unpack('!I', f.read(4))[0])
            
            # Read each compressed frame
            compressed_frames = []
            for _ in range(frame_count):
                frame_len = struct.unpack('!I', f.read(4))[0]
                frame_data = f.read(frame_len)
                compressed_frames.append(frame_data)
            
            metadata = {
                'frame_count': frame_count,
                'frame_shape': (height, width),
                'keyframes': keyframes
            }
            
            return compressed_frames, metadata
    
    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Save frames as a video file."""
        try:
            # First try to use imageio to save as video
            try:
                writer = imageio.get_writer(output_path, fps=fps)
                for frame in frames:
                    writer.append_data(frame.astype(np.uint8))
                writer.close()
                print(f"Video saved to {output_path}")
            except (ImportError, ValueError) as e:
                # If FFMPEG is not available, inform the user
                print(f"Could not save video: {e}")
                print("To save videos, install FFMPEG support: pip install 'imageio[ffmpeg]'")
                
                # Save as individual frames instead
                frames_dir = os.path.splitext(output_path)[0] + "_frames"
                os.makedirs(frames_dir, exist_ok=True)
                
                print(f"Saving individual frames to {frames_dir} instead")
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    Image.fromarray(frame).save(frame_path)
                
                print(f"Saved {len(frames)} frames to {frames_dir}")
        except Exception as e:
            print(f"Error saving video: {e}")
            raise


def generate_synthetic_video(frame_count: int = 60, width: int = 320, height: int = 240) -> List[np.ndarray]:
    """
    Generate a synthetic video with a moving object.
    
    Args:
        frame_count: Number of frames to generate
        width: Frame width
        height: Frame height
        
    Returns:
        List of video frames
    """
    frames = []
    
    # Create a black background
    background = np.zeros((height, width), dtype=np.uint8)
    
    # Parameters for a moving circle
    radius = 30
    x, y = radius + 10, height // 2
    dx, dy = 4, 2
    
    for i in range(frame_count):
        # Create a copy of the background
        frame = background.copy()
        
        # Draw a circle at the current position
        for h in range(height):
            for w in range(width):
                if (h - y) ** 2 + (w - x) ** 2 <= radius ** 2:
                    frame[h, w] = 255
        
        frames.append(frame)
        
        # Update position
        x += dx
        y += dy
        
        # Bounce off edges
        if x <= radius or x >= width - radius:
            dx = -dx
        if y <= radius or y >= height - radius:
            dy = -dy
    
    return frames


def generate_synthetic_video_with_pattern(pattern: str = "circle", 
                                frame_count: int = 60, 
                                width: int = 320, 
                                height: int = 240) -> List[np.ndarray]:
    """
    Generate a synthetic video with different motion patterns.
    
    Args:
        pattern: The motion pattern type ('circle', 'growing', 'random', 'static', 'multiple')
        frame_count: Number of frames to generate
        width: Frame width
        height: Frame height
        
    Returns:
        List of video frames
    """
    frames = []
    
    # Create a black background
    background = np.zeros((height, width), dtype=np.uint8)
    
    if pattern == "circle":
        # Moving circle (bouncing around)
        radius = 30
        x, y = radius + 10, height // 2
        dx, dy = 4, 2
        
        for i in range(frame_count):
            frame = background.copy()
            
            # Draw a circle
            for h in range(height):
                for w in range(width):
                    if (h - y) ** 2 + (w - x) ** 2 <= radius ** 2:
                        frame[h, w] = 255
            
            frames.append(frame)
            
            # Update position
            x += dx
            y += dy
            
            # Bounce off edges
            if x <= radius or x >= width - radius:
                dx = -dx
            if y <= radius or y >= height - radius:
                dy = -dy
                
    elif pattern == "growing":
        # Circle that grows and shrinks
        x, y = width // 2, height // 2
        min_radius, max_radius = 10, 80
        radius = min_radius
        growing = True
        
        for i in range(frame_count):
            frame = background.copy()
            
            # Draw a circle with current radius
            for h in range(height):
                for w in range(width):
                    if (h - y) ** 2 + (w - x) ** 2 <= radius ** 2:
                        frame[h, w] = 255
            
            frames.append(frame)
            
            # Update radius
            if growing:
                radius += 2
                if radius >= max_radius:
                    growing = False
            else:
                radius -= 2
                if radius <= min_radius:
                    growing = True
                    
    elif pattern == "random":
        # Random noise with slowly changing density
        density = 0.05  # Starting density
        change_rate = 0.005  # Rate of change
        increasing = True
        
        for i in range(frame_count):
            # Generate random noise with current density
            random_data = np.random.random((height, width))
            frame = (random_data < density).astype(np.uint8) * 255
            frames.append(frame)
            
            # Update density
            if increasing:
                density += change_rate
                if density >= 0.3:
                    increasing = False
            else:
                density -= change_rate
                if density <= 0.05:
                    increasing = True
                    
    elif pattern == "static":
        # Static image that doesn't change
        # Create a checkerboard pattern
        cell_size = 20
        for i in range(frame_count):
            frame = background.copy()
            for h in range(0, height, cell_size):
                for w in range(0, width, cell_size):
                    if (h // cell_size + w // cell_size) % 2 == 0:
                        h_end = min(h + cell_size, height)
                        w_end = min(w + cell_size, width)
                        frame[h:h_end, w:w_end] = 255
            frames.append(frame)
            
    elif pattern == "multiple":
        # Multiple objects moving independently
        num_objects = 5
        objects = []
        
        # Initialize objects with different positions, sizes, and velocities
        for _ in range(num_objects):
            radius = np.random.randint(10, 40)
            x = np.random.randint(radius, width - radius)
            y = np.random.randint(radius, height - radius)
            dx = np.random.randint(2, 6) * (1 if np.random.random() > 0.5 else -1)
            dy = np.random.randint(2, 6) * (1 if np.random.random() > 0.5 else -1)
            objects.append({
                'radius': radius,
                'x': x, 'y': y,
                'dx': dx, 'dy': dy
            })
        
        for i in range(frame_count):
            frame = background.copy()
            
            # Draw and update each object
            for obj in objects:
                # Draw circle
                for h in range(height):
                    for w in range(width):
                        if (h - obj['y']) ** 2 + (w - obj['x']) ** 2 <= obj['radius'] ** 2:
                            frame[h, w] = 255
                
                # Update position
                obj['x'] += obj['dx']
                obj['y'] += obj['dy']
                
                # Bounce off edges
                if obj['x'] <= obj['radius'] or obj['x'] >= width - obj['radius']:
                    obj['dx'] = -obj['dx']
                if obj['y'] <= obj['radius'] or obj['y'] >= height - obj['radius']:
                    obj['dy'] = -obj['dy']
            
            frames.append(frame)
    
    else:
        # Default to simple circle if pattern is not recognized
        print(f"Unrecognized pattern '{pattern}', using circle instead")
        return generate_synthetic_video(frame_count, width, height)
    
    return frames


def run_video_compression_test(save_path: str = "test_videos",
                             frame_count: int = 60,
                             width: int = 320,
                             height: int = 240,
                             pattern: str = "circle"):
    """
    Run a test of the video delta compression using Bloom filters.
    
    Args:
        save_path: Directory to save results
        frame_count: Number of frames in the test video
        width: Frame width
        height: Frame height
        pattern: Motion pattern to use for synthetic video generation
    """
    # Create directory if needed
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Generating synthetic video with '{pattern}' pattern ({frame_count} frames, {width}x{height})")
    
    # Generate synthetic video
    frames = generate_synthetic_video_with_pattern(pattern, frame_count, width, height)
    
    # Save original video or frames
    original_path = os.path.join(save_path, "original.mp4")
    compressor = VideoDeltaCompressor()
    try:
        compressor._save_video(frames, original_path)
    except Exception as e:
        print(f"Warning: Could not save original video: {e}")
        print("Continuing with test...")
    
    # Compress the video
    print("\nCompressing video...")
    compressed_path = os.path.join(save_path, "compressed.bin")
    start_time = time.time()
    compressed_frames, metadata = compressor.compress_video(frames, output_path=compressed_path)
    compress_time = time.time() - start_time
    
    # Calculate the size of the frames directly (in case video saving failed)
    frame_size_bytes = sum(frame.nbytes for frame in frames)
    
    # Calculate compression statistics
    try:
        original_size = os.path.getsize(original_path)
    except:
        # If original video file doesn't exist, use raw frame size
        original_size = frame_size_bytes
        
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = compressed_size / original_size
    
    # Decompress the video
    print("\nDecompressing video...")
    decompressed_path = os.path.join(save_path, "decompressed.mp4")
    start_time = time.time()
    decompressed_frames = compressor.decompress_video(
        compressed_frames, metadata, output_path=decompressed_path)
    decompress_time = time.time() - start_time
    
    # Verify reconstruction
    is_lossless = True
    frame_diffs = []
    for i in range(len(frames)):
        # Calculate mean absolute error
        mae = np.mean(np.abs(frames[i].astype(np.float32) - decompressed_frames[i].astype(np.float32)))
        frame_diffs.append(mae)
        if mae > 0:
            is_lossless = False
    
    # Print results
    print("\nVideo Compression Results:")
    print("-" * 40)
    print(f"Frame count: {frame_count}")
    print(f"Frame dimensions: {width}x{height}")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.6f}")
    print(f"Bytes per frame (compressed): {compressed_size/frame_count:,.1f}")
    print(f"Compression time: {compress_time:.2f} seconds")
    print(f"Decompression time: {decompress_time:.2f} seconds")
    print(f"Lossless reconstruction: {is_lossless}")
    
    if not is_lossless:
        print(f"Average frame difference (MAE): {np.mean(frame_diffs):.4f}")
    
    # Plot compression statistics
    plt.figure(figsize=(12, 10))
    
    # Plot frame-by-frame density and compression ratio
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(metadata['densities']) + 1), metadata['densities'], 'o-')
    plt.axhline(y=BloomFilterCompressor.P_STAR, color='r', linestyle='--', 
                label=f'Threshold ({BloomFilterCompressor.P_STAR:.4f})')
    plt.xlabel('Frame')
    plt.ylabel('Difference Density')
    plt.title('Frame-by-Frame Difference Density')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(metadata['compression_ratios']) + 1), metadata['compression_ratios'], 'o-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.xlabel('Frame')
    plt.ylabel('Compression Ratio')
    plt.title('Frame-by-Frame Compression Ratio')
    plt.grid(True)
    plt.legend()
    
    # Show original vs decompressed frame comparison
    frame_idx = min(30, len(frames) - 1)  # Choose a mid-way frame
    
    plt.subplot(2, 2, 3)
    plt.imshow(frames[frame_idx], cmap='gray')
    plt.title(f"Original Frame {frame_idx}")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(decompressed_frames[frame_idx], cmap='gray')
    plt.title(f"Decompressed Frame {frame_idx}")
    plt.axis('off')
    
    # Save key frames as images for inspection
    frames_sample_dir = os.path.join(save_path, "sample_frames")
    os.makedirs(frames_sample_dir, exist_ok=True)
    
    sample_indices = [0, 15, 30, 45] if frame_count >= 45 else list(range(0, frame_count, max(1, frame_count // 4)))
    for idx in sample_indices:
        if idx < len(frames):
            # Save original
            orig_path = os.path.join(frames_sample_dir, f"original_frame_{idx:02d}.png")
            Image.fromarray(frames[idx]).save(orig_path)
            
            # Save decompressed
            decomp_path = os.path.join(frames_sample_dir, f"decompressed_frame_{idx:02d}.png")
            Image.fromarray(decompressed_frames[idx]).save(decomp_path)
    
    # Save difference visualization for a sample frame
    if not is_lossless and frame_idx < len(frames):
        diff = np.abs(frames[frame_idx].astype(np.float32) - decompressed_frames[frame_idx].astype(np.float32))
        diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else np.zeros_like(frames[frame_idx])
        diff_path = os.path.join(frames_sample_dir, f"diff_frame_{frame_idx:02d}.png")
        Image.fromarray(diff_norm).save(diff_path)
    
    plt.tight_layout()
    plots_path = os.path.join(save_path, "video_compression_results.png")
    plt.savefig(plots_path)
    
    print(f"\nResults saved in: {save_path}")
    print(f"- Compressed data: {compressed_path}")
    print(f"- Sample frames: {frames_sample_dir}")
    print(f"- Compression statistics plot: {plots_path}")
    
    # Return key metrics
    return {
        "frame_count": frame_count,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "is_lossless": is_lossless
    }


def run_video_pattern_experiments():
    """Run experiments on different video motion patterns."""
    patterns = [
        "circle",      # Single circle moving around (low frame differences)
        "growing",     # Growing/shrinking circle (medium frame differences)
        "multiple",    # Multiple objects (medium-high frame differences)
        "random",      # Random noise with changing density (high frame differences)
        "static"       # Static image (minimal frame differences)
    ]
    
    results = []
    
    print("========================================================")
    print("RUNNING EXPERIMENTS WITH DIFFERENT VIDEO MOTION PATTERNS")
    print("========================================================")
    
    for pattern in patterns:
        print(f"\n\n{'=' * 30}")
        print(f"TESTING PATTERN: {pattern.upper()}")
        print(f"{'=' * 30}")
        
        save_path = f"test_videos_{pattern}"
        result = run_video_compression_test(
            save_path=save_path,
            frame_count=60,
            width=320,
            height=240,
            pattern=pattern
        )
        
        result['pattern'] = pattern
        results.append(result)
    
    # Print summary comparison
    print("\n\n")
    print("=" * 60)
    print("SUMMARY OF RESULTS FOR DIFFERENT MOTION PATTERNS")
    print("=" * 60)
    print("Pattern    | Compression Ratio | Bytes/Frame | Lossless")
    print("-" * 60)
    
    for r in results:
        print(f"{r['pattern']:10} | {r['compression_ratio']:.6f} | {r['compressed_size']/r['frame_count']:8,.1f} | {r['is_lossless']}")
    
    # Plot comparison of compression ratios
    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    index = np.arange(len(patterns))
    
    compression_ratios = [r['compression_ratio'] for r in results]
    
    plt.bar(index, compression_ratios, bar_width)
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.xlabel('Motion Pattern')
    plt.ylabel('Compression Ratio')
    plt.title('Bloom Filter Compression Performance for Different Motion Patterns')
    plt.xticks(index, patterns)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/pattern_comparison.png")
    print(f"\nComparison plot saved to: test_results/pattern_comparison.png")


if __name__ == "__main__":
    # Run test on a single video
    # run_video_compression_test()
    
    # Run experiments with different video motion patterns
    run_video_pattern_experiments() 