import xxhash
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional, Union
import io
import struct
from pathlib import Path
import time


class BloomFilterCompressor:
    """
    Implementation of lossless compression with Bloom filters as described in 
    "Lossless Compression with Bloom Filters" paper.
    
    This implementation uses Rational Bloom Filters to allow for non-integer number
    of hash functions (k).
    """
    
    # Critical density threshold for compression
    P_STAR = 0.32453
    
    def __init__(self):
        """Initialize the compressor with default parameters."""
        pass
        
    @staticmethod
    def _calculate_optimal_params(n: int, p: float) -> Tuple[float, int]:
        """
        Calculate the optimal parameters k (number of hash functions) and
        l (bloom filter length) for lossless compression.
        
        Args:
            n: Length of the binary input string
            p: Density (probability of '1' bits) in the input
            
        Returns:
            Tuple of (k, l) where k is optimal hash count and l is optimal filter length
        """
        if p >= BloomFilterCompressor.P_STAR:
            # Compression not effective for this density
            return 0, 0
        
        q = 1 - p  # Probability of '0' bits
        L = math.log(2)  # ln(2)
        
        # Calculate optimal k 
        k = math.log2(q * (L**2) / p)
        
        # Calculate optimal filter length
        gamma = 1 / L
        l = int(p * n * k * gamma)
        
        return max(0.1, k), max(1, l)  # Ensure k and l are positive
    
    @staticmethod
    def _binarize_image(image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Convert an image to a binary representation.
        
        Args:
            image: Input image as numpy array
            threshold: Threshold value for binarization (0-255)
            
        Returns:
            Binary representation of the image as 1D numpy array of 0s and 1s
        """
        # If image has multiple channels, convert to grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            # Simple grayscale conversion (average of RGB)
            image = np.mean(image, axis=2).astype(np.uint8)
        
        # Binarize the image
        binary_image = (image > threshold).astype(np.uint8)
        
        # Flatten to 1D array
        return binary_image.flatten()
    
    class RationalBloomFilter:
        """
        Rational Bloom filter implementation specifically for compression.
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
            
            # Constants for double hashing
            self.h1_seed = 0
            self.h2_seed = 1
        
        def _get_hash_indices(self, item: int, i: int) -> int:
            """
            Generate hash indices using double hashing technique.
            
            Args:
                item: The integer item to hash (index position)
                i: The index of the hash function (0 to floor_k or ceil_k - 1)
                
            Returns:
                A hash index in range [0, size-1]
            """
            # Use item as a seed for xxhash
            h1 = xxhash.xxh64(str(item), seed=self.h1_seed).intdigest()
            h2 = xxhash.xxh64(str(item), seed=self.h2_seed).intdigest()
            
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
            hash_value = xxhash.xxh64(str(item), seed=999).intdigest()
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
            print(f"Density {p:.4f} is >= threshold {self.P_STAR}, compression not effective")
            return binary_input, [], p, n, 1.0
        
        # Calculate optimal parameters
        k, l = self._calculate_optimal_params(n, p)
        
        if l == 0:
            # Compression not possible, return original
            return binary_input, [], p, n, 1.0
        
        print(f"Input length: {n}, Density: {p:.4f}")
        print(f"Optimal parameters: k={k:.4f}, l={l}")
        
        # Create Bloom filter
        bloom_filter = self.RationalBloomFilter(l, k)
        
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
        bloom_filter = self.RationalBloomFilter(l, k)
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
    
    def compress_image(self, image_path: str, threshold: int = 127, 
                      output_path: Optional[str] = None) -> Tuple[bytes, float]:
        """
        Compress an image using Bloom filter compression.
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binarization
            output_path: Optional path to save the compressed data
            
        Returns:
            Tuple of (compressed_data_bytes, compression_ratio)
        """
        # Load and binarize image
        img = np.array(Image.open(image_path))
        binary_data = self._binarize_image(img, threshold)
        
        # Store original image dimensions
        original_shape = img.shape
        
        # Compress the binary data
        bloom_bitmap, witness, p, n, compression_ratio = self.compress(binary_data)
        
        # Calculate optimal k for the given density
        k, _ = self._calculate_optimal_params(n, p)
        
        # Pack the compressed data
        compressed_data = self._pack_compressed_data(
            bloom_bitmap, witness, p, n, k, original_shape)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
        
        return compressed_data, compression_ratio
    
    def decompress_image(self, compressed_data: bytes, 
                        output_path: Optional[str] = None) -> np.ndarray:
        """
        Decompress an image that was compressed with Bloom filter compression.
        
        Args:
            compressed_data: The compressed data bytes
            output_path: Optional path to save the decompressed image
            
        Returns:
            The decompressed image as a numpy array
        """
        # Unpack the compressed data
        bloom_bitmap, witness, p, n, k, original_shape = self._unpack_compressed_data(compressed_data)
        
        # Decompress the binary data
        decompressed_binary = self.decompress(bloom_bitmap, witness, n, k)
        
        # Reshape to original image dimensions
        if len(original_shape) > 2:
            # Handle grayscale conversion
            height, width = original_shape[:2]
        else:
            height, width = original_shape
            
        decompressed_image = decompressed_binary.reshape((height, width)) * 255
        
        # Convert to PIL Image and save if requested
        if output_path:
            Image.fromarray(decompressed_image.astype(np.uint8)).save(output_path)
        
        return decompressed_image
    
    def _pack_compressed_data(self, bloom_bitmap: np.ndarray, witness: list, 
                             p: float, n: int, k: float, 
                             original_shape: Tuple) -> bytes:
        """Pack the compressed data into a binary format for storage."""
        buffer = io.BytesIO()
        
        # Write header
        buffer.write(struct.pack('!f', p))  # Density
        buffer.write(struct.pack('!I', n))  # Original length
        buffer.write(struct.pack('!f', k))  # Hash function count
        
        # Write shape information
        shape_len = len(original_shape)
        buffer.write(struct.pack('!B', shape_len))
        for dim in original_shape:
            buffer.write(struct.pack('!I', dim))
        
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
        
        return buffer.getvalue()
    
    def _unpack_compressed_data(self, data: bytes) -> Tuple:
        """Unpack the compressed data from binary format."""
        buffer = io.BytesIO(data)
        
        # Read header
        p = struct.unpack('!f', buffer.read(4))[0]
        n = struct.unpack('!I', buffer.read(4))[0]
        k = struct.unpack('!f', buffer.read(4))[0]
        
        # Read shape information
        shape_len = struct.unpack('!B', buffer.read(1))[0]
        original_shape = []
        for _ in range(shape_len):
            original_shape.append(struct.unpack('!I', buffer.read(4))[0])
        original_shape = tuple(original_shape)
        
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
        
        return bloom_bitmap, witness, p, n, k, original_shape


def run_compression_tests():
    """Run tests for the Bloom filter compression algorithm."""
    compressor = BloomFilterCompressor()
    
    # Test 1: Synthetic binary data
    print("Test 1: Synthetic binary data")
    print("============================")
    
    # Create synthetic data with controlled density
    n = 100000  # Size of binary vector
    for p in [0.1, 0.2, 0.3, 0.4]:
        print(f"\nDensity p = {p}")
        binary_data = np.random.choice([0, 1], size=n, p=[1-p, p])
        
        # Compress
        start_time = time.time()
        bloom_bitmap, witness, density, input_length, ratio = compressor.compress(binary_data)
        compress_time = time.time() - start_time
        
        # Calculate optimal parameters for decompression
        k, _ = compressor._calculate_optimal_params(n, density)
        
        # Decompress
        start_time = time.time()
        decompressed = compressor.decompress(bloom_bitmap, witness, input_length, k)
        decompress_time = time.time() - start_time
        
        # Verify correctness
        is_lossless = np.array_equal(binary_data, decompressed)
        print(f"Lossless reconstruction: {is_lossless}")
        print(f"Compression ratio: {ratio:.4f}")
        print(f"Compression time: {compress_time:.4f}s")
        print(f"Decompression time: {decompress_time:.4f}s")
        
        # Print explanation if density is above threshold
        if density >= compressor.P_STAR:
            print(f"Note: Density {density:.4f} is above threshold {compressor.P_STAR:.4f}")
            print("No actual compression was performed (ratio should be 1.0)")
    
    # Test 2: Image compression
    try:
        # Create a synthetic image
        print("\nTest 2: Image compression")
        print("========================")
        
        # Create a simple 100x100 binary image
        width, height = 100, 100
        test_image = np.zeros((height, width), dtype=np.uint8)
        
        # Add some patterns to make it interesting
        test_image[25:75, 25:75] = 255  # Square
        test_image[40:60, 40:60] = 0    # Inner square
        
        # Save the test image
        Image.fromarray(test_image).save("test_image.png")
        
        # Binarize and check density before attempting compression
        binary_data = compressor._binarize_image(test_image, threshold=127)
        density = np.sum(binary_data) / len(binary_data)
        print(f"Image density: {density:.4f}")
        
        if density >= compressor.P_STAR:
            print(f"Note: Image density {density:.4f} is above threshold {compressor.P_STAR:.4f}")
            print("Compression may not be effective")
        
        # Compress the image
        print("\nCompressing test image...")
        compressed_data, ratio = compressor.compress_image("test_image.png", threshold=127, 
                                                          output_path="test_image.bloom")
        
        # Decompress the image
        print("\nDecompressing test image...")
        decompressed_image = compressor.decompress_image(compressed_data, 
                                                        output_path="test_image_decompressed.png")
        
        # Calculate PSNR or other image quality metrics
        # Since it's a binary image and lossless compression, we just check for exact equality
        original_binary = compressor._binarize_image(test_image, threshold=127)
        decompressed_binary = decompressed_image.flatten() / 255
        
        is_lossless = np.array_equal(original_binary, decompressed_binary)
        print(f"Lossless reconstruction: {is_lossless}")
        print(f"Compression ratio: {ratio:.4f}")

        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(decompressed_image, cmap='gray')
        plt.title("Decompressed Image")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("bloom_compression_results.png")
        plt.close()
        
        print("Results saved to bloom_compression_results.png")
        
    except Exception as e:
        print(f"Error in image compression test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_compression_tests() 