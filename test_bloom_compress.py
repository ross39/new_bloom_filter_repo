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


def test_text_compression():
    """Test compression performance on various types of text data."""
    compressor = BloomFilterCompressor()
    
    # Create a test directory
    os.makedirs("test_texts", exist_ok=True)
    
    # Test with different types of text content
    text_samples = {
        "sparse_binary": "0" * 1000 + "1" * 100,  # Very sparse binary string (mostly 0s)
        "repeated_patterns": "hello world " * 100,  # Repeated pattern
        "english_text": """
            The Rational Bloom filter allows for a non-integer number of hash functions (k*),
            which is achieved by probabilistically applying an additional hash function
            beyond the floor(k*) deterministic hash functions. This approach provides better
            false positive rates compared to standard Bloom filters that must use an integer
            number of hash functions.
        """,
        "lorem_ipsum": """
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
            exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
            irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
            pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
            deserunt mollit anim id est laborum.
        """,
        "unicode_text": """
            Unicode includes symbols like: ☀ ☁ ☂ ☃ ☄ ★ ☆ ☉ ☘ ☺ ☻
            And languages like: 你好 (Chinese), こんにちは (Japanese), 안녕하세요 (Korean)
            Plus emojis: 😀 😁 😂 😃 😄 😅 😆 😇
        """
    }
    
    print("\nText Compression Tests")
    print("=====================")
    print("\nText Type | Length | Bin Density | Bloom Ratio | Gzip Ratio | Lossless")
    print("----------|--------|-------------|-------------|------------|----------")
    
    results = []
    
    for text_type, text in text_samples.items():
        # Save original text
        with open(f"test_texts/{text_type}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Binarize to check density
        binary_data = compressor._binarize_text(text)
        density = np.sum(binary_data) / len(binary_data)
        
        # Compress with Bloom filter
        start_time = time.time()
        compressed_data, bloom_ratio = compressor.compress_text(
            text, output_path=f"test_texts/{text_type}.bloom")
        compress_time = time.time() - start_time
        
        # Decompress
        start_time = time.time()
        decompressed_text = compressor.decompress_text(
            compressed_data, output_path=f"test_texts/{text_type}_decompressed.txt")
        decompress_time = time.time() - start_time
        
        # Verify lossless reconstruction
        is_lossless = (text == decompressed_text)
        
        # Compress with gzip for comparison
        text_bytes = text.encode('utf-8')
        gzip_compressed = gzip.compress(text_bytes)
        gzip_ratio = len(gzip_compressed) / len(text_bytes)
        
        print(f"{text_type[:10]:10} | {len(text):6} | {density:.6f} | {bloom_ratio:.6f} | {gzip_ratio:.6f} | {is_lossless}")
        
        # Store detailed results
        result = {
            'text_type': text_type,
            'length': len(text),
            'bin_length': len(binary_data),
            'density': density,
            'bloom_ratio': bloom_ratio,
            'gzip_ratio': gzip_ratio,
            'is_lossless': is_lossless,
            'compress_time': compress_time,
            'decompress_time': decompress_time
        }
        results.append(result)
        
        # Print detailed stats for this sample
        print(f"\nDetailed results for {text_type}:")
        print(f"  Text length: {len(text)} characters")
        print(f"  Binary length: {len(binary_data)} bits")
        print(f"  Binary density: {density:.6f}")
        print(f"  Bloom filter compression ratio: {bloom_ratio:.6f}")
        print(f"  Gzip compression ratio: {gzip_ratio:.6f}")
        print(f"  Bloom compression time: {compress_time:.6f}s")
        print(f"  Bloom decompression time: {decompress_time:.6f}s")
        print(f"  Lossless reconstruction: {is_lossless}")
        
        # Add analysis notes
        if density >= BloomFilterCompressor.P_STAR:
            print(f"  Note: Density {density:.6f} is above threshold {BloomFilterCompressor.P_STAR}")
            print("  No actual compression with Bloom filter (ratio should be 1.0)")
        elif bloom_ratio < gzip_ratio:
            print("  Bloom filter compression outperformed gzip!")
        else:
            print("  Gzip compression was more efficient in this case")
    
    # Plot comparison
    text_types = [r['text_type'] for r in results]
    bloom_ratios = [r['bloom_ratio'] for r in results]
    gzip_ratios = [r['gzip_ratio'] for r in results]
    densities = [r['density'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # Plot comparison of compression ratios
    plt.subplot(2, 1, 1)
    bar_width = 0.35
    index = np.arange(len(text_types))
    
    plt.bar(index, bloom_ratios, bar_width, label='Bloom Filter')
    plt.bar(index + bar_width, gzip_ratios, bar_width, label='Gzip')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.xlabel('Text Type')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio Comparison')
    plt.xticks(index + bar_width/2, text_types, rotation=45)
    plt.legend()
    
    # Plot density vs compression ratio
    plt.subplot(2, 1, 2)
    plt.scatter(densities, bloom_ratios, label='Bloom Filter', marker='o')
    plt.scatter(densities, gzip_ratios, label='Gzip', marker='s')
    plt.axvline(x=BloomFilterCompressor.P_STAR, color='g', linestyle='--',
                label=f'P* Threshold ({BloomFilterCompressor.P_STAR:.4f})')
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.xlabel('Binary Density')
    plt.ylabel('Compression Ratio')
    plt.title('Density vs Compression Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('text_compression_comparison.png')
    plt.close()
    
    print("\nResults saved to test_texts/ directory and text_compression_comparison.png")
    
    return results


def test_real_world_files():
    """Test compression on real-world text files like source code, XML, JSON, etc."""
    compressor = BloomFilterCompressor()
    
    # Create test directory
    os.makedirs("test_files", exist_ok=True)
    
    # Create sample files of different types
    test_files = {}
    
    # Python source code
    test_files["python_code"] = """
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def calculate_fibonacci(n: int) -> List[int]:
    # Return the first n Fibonacci numbers.
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def plot_fibonacci(n: int, save_path: Optional[str] = None) -> None:
    # Plot the first n Fibonacci numbers.
    fib = calculate_fibonacci(n)
    plt.figure(figsize=(10, 6))
    plt.plot(range(n), fib, 'o-')
    plt.title(f"First {n} Fibonacci Numbers")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    plot_fibonacci(20, "fibonacci.png")
"""
    
    # JSON data
    test_files["json_data"] = """
{
  "name": "Bloom Filter Compression",
  "version": "1.0.0",
  "description": "Lossless compression using Bloom filters",
  "author": "Example Author",
  "license": "MIT",
  "dependencies": {
    "numpy": "^1.20.0",
    "matplotlib": "^3.4.0",
    "xxhash": "^2.0.0"
  },
  "keywords": [
    "bloom-filter",
    "compression",
    "lossless",
    "probabilistic",
    "data-structures"
  ],
  "examples": [
    {"name": "Basic usage", "file": "example1.py"},
    {"name": "Advanced features", "file": "example2.py"},
    {"name": "Benchmarks", "file": "benchmarks.py"}
  ]
}
"""
    
    # XML data
    test_files["xml_data"] = """
<?xml version="1.0" encoding="UTF-8"?>
<root>
    <compression>
        <algorithm name="BloomFilterCompression" type="lossless">
            <parameter name="bit_array_size" value="10000" />
            <parameter name="hash_functions" value="2.5" />
            <parameter name="threshold" value="0.32453" />
        </algorithm>
        <performance>
            <metric name="compression_ratio" value="0.75" />
            <metric name="speed" value="fast" />
            <metric name="memory_usage" value="low" />
        </performance>
        <applications>
            <application>Binary images</application>
            <application>Sparse data</application>
            <application>Text compression</application>
        </applications>
    </compression>
</root>
"""
    
    # HTML data
    test_files["html_data"] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bloom Filter Compression</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #2c3e50; }
        .container { max-width: 800px; margin: 0 auto; }
        .highlight { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
        code { font-family: 'Courier New', monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bloom Filter Compression</h1>
        <p>A novel approach to lossless compression using Bloom filters.</p>
        
        <h2>How It Works</h2>
        <div class="highlight">
            <code>
                # Create compressor<br>
                compressor = BloomFilterCompressor()<br>
                <br>
                # Compress data<br>
                compressed = compressor.compress(binary_data)<br>
                <br>
                # Decompress data<br>
                original = compressor.decompress(compressed)
            </code>
        </div>
        
        <h2>Features</h2>
        <ul>
            <li>Lossless compression</li>
            <li>Optimal for sparse binary data</li>
            <li>Rational Bloom filter implementation</li>
            <li>Text and image compression support</li>
        </ul>
    </div>
</body>
</html>
"""
    
    print("\nReal-World File Compression Tests")
    print("===============================")
    print("\nFile Type | Size (B) | Bin Density | Bloom Ratio | Gzip Ratio | Lossless")
    print("----------|----------|-------------|-------------|------------|----------")
    
    results = []
    
    for file_type, content in test_files.items():
        # Save the file
        file_path = f"test_files/{file_type}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Binarize to check density
        binary_data = compressor._binarize_text(content)
        density = np.sum(binary_data) / len(binary_data)
        
        # Compress with Bloom filter
        compressed_data, bloom_ratio = compressor.compress_text(
            content, output_path=f"test_files/{file_type}.bloom")
        
        # Decompress
        decompressed_text = compressor.decompress_text(
            compressed_data, output_path=f"test_files/{file_type}_decompressed.txt")
        
        # Verify lossless reconstruction
        is_lossless = (content == decompressed_text)
        
        # Compress with gzip for comparison
        text_bytes = content.encode('utf-8')
        gzip_compressed = gzip.compress(text_bytes)
        gzip_ratio = len(gzip_compressed) / len(text_bytes)
        
        print(f"{file_type[:10]:10} | {file_size:8} | {density:.6f} | {bloom_ratio:.6f} | {gzip_ratio:.6f} | {is_lossless}")
        
        results.append({
            'file_type': file_type,
            'size': file_size,
            'density': density,
            'bloom_ratio': bloom_ratio,
            'gzip_ratio': gzip_ratio,
            'is_lossless': is_lossless
        })
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(results))
    
    bloom_ratios = [r['bloom_ratio'] for r in results]
    gzip_ratios = [r['gzip_ratio'] for r in results]
    file_types = [r['file_type'] for r in results]
    
    plt.bar(index, bloom_ratios, bar_width, label='Bloom Filter')
    plt.bar(index + bar_width, gzip_ratios, bar_width, label='Gzip')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Compression')
    plt.xlabel('File Type')
    plt.ylabel('Compression Ratio')
    plt.title('File Compression Ratio Comparison')
    plt.xticks(index + bar_width/2, file_types, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('file_compression_comparison.png')
    plt.close()
    
    print("\nResults saved to test_files/ directory and file_compression_comparison.png")
    
    return results


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
    
    # Test 3: Text compression
    text_results = test_text_compression()
    
    # Test 4: Real-world file compression
    file_results = test_real_world_files()
    
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
    
    # Text compression summary
    best_text_ratio = min([r['bloom_ratio'] for r in text_results])
    best_text_type = next(r['text_type'] for r in text_results if r['bloom_ratio'] == best_text_ratio)
    print(f"Best text compression ratio (Bloom): {best_text_ratio:.4f} ({best_text_type})")
    
    # File compression summary
    best_file_ratio = min([r['bloom_ratio'] for r in file_results])
    best_file_type = next(r['file_type'] for r in file_results if r['bloom_ratio'] == best_file_ratio)
    print(f"Best file compression ratio (Bloom): {best_file_ratio:.4f} ({best_file_type})")
    
    print("\nResults saved in test directories and comparison PNG files")


if __name__ == "__main__":
    main() 