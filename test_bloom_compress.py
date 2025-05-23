import numpy as np
from bloom_compress import BloomFilterCompressor
import gzip
import time
import matplotlib.pyplot as plt
from PIL import Image
import os


def test_synthetic_data_compression():
    """Test compression performance on synthetic binary data."""
    compressor = BloomFilterCompressor()
    
    # Test various input densities
    densities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    n = 100000  # Size of binary vector
    
    bloom_ratios = []
    gzip_ratios = []
    lossless_results = []
    
    print("Testing compression on synthetic data")
    print("====================================")
    print(f"Data length: {n} bits")
    print("\nDensity | Bloom Ratio | Gzip Ratio | Lossless")
    print("--------|-------------|-----------|----------")
    
    for p in densities:
        # Generate random binary data with specified density
        binary_data = np.random.choice([0, 1], size=n, p=[1-p, p]).astype(np.uint8)
        
        # Compress with Bloom filter
        bloom_bitmap, witness, density, input_length, bloom_ratio = compressor.compress(binary_data)
        
        # Calculate parameters for decompression
        k, _ = compressor._calculate_optimal_params(n, density)
        
        # Decompress
        decompressed = compressor.decompress(bloom_bitmap, witness, input_length, k)
        
        # Compress with gzip for comparison
        binary_bytes = np.packbits(binary_data).tobytes()
        gzip_compressed = gzip.compress(binary_bytes)
        gzip_ratio = len(gzip_compressed) * 8 / n  # Convert bytes to bits
        
        # Verify lossless reconstruction
        is_lossless = np.array_equal(binary_data, decompressed)
        lossless_results.append(is_lossless)
        
        # Store results
        bloom_ratios.append(bloom_ratio)
        gzip_ratios.append(gzip_ratio)
        
        print(f"{p:.2f}    | {bloom_ratio:.4f}     | {gzip_ratio:.4f}   | {is_lossless}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(densities, bloom_ratios, 'o-', label='Bloom Filter')
    plt.plot(densities, gzip_ratios, 's-', label='Gzip')
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.axvline(x=BloomFilterCompressor.P_STAR, color='g', linestyle='--', 
                label=f'Theoretical Threshold ({BloomFilterCompressor.P_STAR:.4f})')
    
    plt.xlabel('Density (probability of 1s)')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio vs Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('compression_comparison.png')
    
    return densities, bloom_ratios, gzip_ratios, lossless_results


def test_image_compression(threshold=127):
    """Test compression on a sample image."""
    compressor = BloomFilterCompressor()
    
    # Create a test directory
    os.makedirs("test_images", exist_ok=True)
    
    # Generate a test image with different shapes and patterns
    width, height = 200, 200
    test_image = np.zeros((height, width), dtype=np.uint8)
    
    # Add some patterns
    # Outer square
    test_image[25:175, 25:175] = 255
    # Inner square
    test_image[50:150, 50:150] = 0
    # Diagonal line
    for i in range(width):
        if 0 <= i < height:
            test_image[i, i] = 255
    # Circle
    center_x, center_y = width // 2, height // 2
    radius = 60
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                test_image[y, x] = 255
    
    # Save the test image
    test_image_path = "test_images/test_pattern.png"
    Image.fromarray(test_image).save(test_image_path)
    
    # Binarize image and calculate density
    binary_data = compressor._binarize_image(test_image, threshold)
    density = np.sum(binary_data) / len(binary_data)
    
    print("\nImage Compression Test")
    print("=====================")
    print(f"Image size: {width}x{height} pixels")
    print(f"Binary density: {density:.4f}")
    
    # Check if compression is theoretically possible
    if density >= BloomFilterCompressor.P_STAR:
        print(f"Warning: Density {density:.4f} is >= threshold {BloomFilterCompressor.P_STAR}")
        print("Compression may not be effective.")
    
    # Compress with Bloom filter
    start_time = time.time()
    compressed_data, bloom_ratio = compressor.compress_image(
        test_image_path, threshold, output_path="test_images/compressed.bloom")
    bloom_time = time.time() - start_time
    
    # Decompress
    start_time = time.time()
    decompressed_image = compressor.decompress_image(
        compressed_data, output_path="test_images/decompressed.png")
    decompress_time = time.time() - start_time
    
    # Compress with gzip for comparison
    binary_bytes = np.packbits(binary_data).tobytes()
    start_time = time.time()
    gzip_compressed = gzip.compress(binary_bytes)
    gzip_time = time.time() - start_time
    gzip_ratio = len(gzip_compressed) * 8 / len(binary_data)  # Convert bytes to bits
    
    # Verify lossless reconstruction
    decompressed_binary = decompressed_image.flatten() / 255
    is_lossless = np.array_equal(binary_data, decompressed_binary)
    
    # Print results
    print("\nCompression Results:")
    print(f"Original size: {len(binary_data)} bits")
    print(f"Bloom filter size: {len(compressed_data) * 8} bits")
    print(f"Gzip size: {len(gzip_compressed) * 8} bits")
    print(f"\nBloom Filter Compression ratio: {bloom_ratio:.4f}")
    print(f"Gzip Compression ratio: {gzip_ratio:.4f}")
    print(f"\nBloom Filter Compression time: {bloom_time:.4f}s")
    print(f"Bloom Filter Decompression time: {decompress_time:.4f}s")
    print(f"Gzip Compression time: {gzip_time:.4f}s")
    print(f"\nLossless reconstruction: {is_lossless}")
    
    # Display the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Decompressed image
    axes[1].imshow(decompressed_image, cmap='gray')
    axes[1].set_title("Decompressed Image")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_images/compression_results.png")
    plt.close()
    
    return is_lossless, bloom_ratio, gzip_ratio


def main():
    print("BLOOM FILTER COMPRESSION TESTS")
    print("=============================")
    
    # Test 1: Synthetic data with different densities
    densities, bloom_ratios, gzip_ratios, lossless_results = test_synthetic_data_compression()
    
    # Verify all tests were lossless
    all_lossless = all(lossless_results)
    print(f"\nAll synthetic tests lossless: {all_lossless}")
    
    # Test 2: Image compression
    is_lossless, bloom_ratio, gzip_ratio = test_image_compression()
    
    # Summary
    print("\nSUMMARY")
    print("=======")
    print(f"Bloom filter compression lossless: {all_lossless and is_lossless}")
    print(f"Best Bloom compression ratio on synthetic data: {min(bloom_ratios):.4f}")
    
    # Find density with best compression
    best_idx = bloom_ratios.index(min(bloom_ratios))
    best_density = densities[best_idx]
    print(f"Optimal density for Bloom compression: {best_density:.4f}")
    
    print(f"Image compression ratio (Bloom): {bloom_ratio:.4f}")
    print(f"Image compression ratio (Gzip): {gzip_ratio:.4f}")
    
    print("\nResults saved in test_images/ directory and compression_comparison.png")


if __name__ == "__main__":
    main() 