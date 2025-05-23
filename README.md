# Bloom Filter-Based Lossless Compression

This project implements a lossless compression scheme using Bloom filters, based on the paper "Lossless Compression with Bloom Filters". The implementation uses Rational Bloom Filters to optimize compression performance.

## Overview

Bloom filters are typically used for membership testing, but this project demonstrates how they can be leveraged for lossless data compression. The compression approach works particularly well for binary data with low density (proportion of 1's).

### Key Features

- Lossless compression of binary data using Bloom filters
- Supports non-integer number of hash functions through Rational Bloom Filters
- Image binarization and compression
- Compression performance comparison with standard techniques (gzip)
- Complete test suite to verify lossless reconstruction

## How It Works

### Theoretical Background

The compression algorithm is based on the insight that when the density (proportion of 1's) in a binary string is below a threshold (approximately 0.32453), a Bloom filter can be used to encode the positions of 1's more efficiently than storing the original bits.

The compression consists of two components:
1. **Bloom Filter Bitmap**: Records all positions where 1's appear in the original data
2. **Witness Data**: Records the actual bit values for positions that pass the Bloom filter test

### Compression Process

1. Calculate the density (p) of the binary input
2. If p < p* (threshold ≈ 0.32453), compression is possible
3. Calculate optimal parameters (k, l) for the Bloom filter
4. Create a Bloom filter and add all positions of '1' bits
5. Generate witness data by checking all positions against the Bloom filter
6. The compressed output is the combination of the Bloom filter bitmap and witness

### Decompression Process

1. Initialize a Bloom filter with the stored bitmap
2. For each position in the original data:
   - If the position passes the Bloom filter test, use the next bit from the witness
   - If the position fails the test, the original bit was definitely '0'

## Implementation Details

### Key Components

- `BloomFilterCompressor`: Main compression engine
- `RationalBloomFilter`: Bloom filter implementation allowing non-integer number of hash functions
- `test_bloom_compress.py`: Comprehensive test suite

### Optimal Parameter Calculation

The algorithm uses the following formulas to calculate optimal parameters:
- Optimal hash function count (k): `k = log₂(q * L² / p)` where q = 1-p and L = ln(2)
- Optimal Bloom filter length (l): `l = p * n * k * γ` where γ = 1/L

## Requirements

- Python 3.6+
- NumPy
- Pillow (PIL)
- matplotlib
- xxhash

Install dependencies:
```
pip install numpy pillow matplotlib xxhash
```

## Usage

### Basic Usage

```python
from bloom_compress import BloomFilterCompressor

# Initialize the compressor
compressor = BloomFilterCompressor()

# Compress binary data
bloom_bitmap, witness, density, input_length, ratio = compressor.compress(binary_data)

# Calculate optimal k for decompression
k, _ = compressor._calculate_optimal_params(input_length, density)

# Decompress
decompressed = compressor.decompress(bloom_bitmap, witness, input_length, k)
```

### Image Compression

```python
# Compress an image
compressed_data, ratio = compressor.compress_image("input.png", threshold=127, 
                                                 output_path="compressed.bloom")

# Decompress an image
decompressed_image = compressor.decompress_image(compressed_data, 
                                               output_path="decompressed.png")
```

### Running Tests

To run the test suite:

```
python test_bloom_compress.py
```

This will:
1. Test compression on synthetic data with various densities
2. Compare with gzip compression
3. Test image compression and verify lossless reconstruction
4. Generate visualizations and metrics

## Results

The compression performance depends on the density of the input data:
- Best performance is achieved when the density is significantly lower than the threshold (p*)
- The algorithm produces lossless reconstruction 
- For very sparse data, compression ratios can be substantially better than traditional methods

## License

MIT

## References

- "Lossless Compression with Bloom Filters"
- "Extending the Applicability of Bloom Filters by Relaxing their Parameter Constraints" by Paul Walther et al. 