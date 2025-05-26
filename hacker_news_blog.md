# Lossless Video Compression Using Rational Bloom Filters

## Source code 

https://github.com/ross39/new_bloom_filter_repo

### Setup 
- Clone repo
- Activate venv and install requirements.txt 
- Run the code using 
```bash
python youtube_bloom_compress.py https://www.youtube.com/shorts/ygelNHGhkNQ \
    --resolution 720p --preserve-color
```

### Files that matter 
The repo is still a bit of a mess. I will list the only file that is important. 
- youtube_bloom_compress.py

If all you want to see is the lossless video compression then just run the above shell command to run the code along with the correct flags passed in. This is a random youtube short just for demo. I will optimise this to work for longer videos(if you have any ideas please let me know).

## Introduction

Traditional video codecs like H.264 and H.265 achieve impressive compression by discarding "imperceptible" visual information. But what if we could guarantee perfect reconstruction while still achieving meaningful compression? This project explores an unconventional approach: repurposing Bloom filters—typically used for membership testing—as a lossless video compression mechanism.

## Bloom Filters: A Quick Primer

Bloom filters are probabilistic data structures that test set membership with remarkable space efficiency. They map elements to positions in a bit array using multiple hash functions. When querying, if all relevant positions contain 1s, the element is *probably* in the set.

The classic trade-off: Bloom filters permit false positives but never false negatives. An element might be incorrectly flagged as "present," but a truly present element will never be missed.

## The Innovation: Rational Bloom Filters

The optimal number of hash functions (k) for a Bloom filter is rarely an integer. Traditional implementations round to the nearest integer, sacrificing theoretical optimality. This project introduces Rational Bloom Filters that implement non-integer hash function counts (k*) through a probabilistic approach:

1. Always apply ⌊k*⌋ hash functions deterministically
2. Apply an additional hash function with probability equal to the fractional part of k*

For instance, with k* = 2.7, we always apply 2 hash functions and probabilistically apply a third with 70% probability. This probability is determined deterministically per element:

```python
def _determine_activation(self, item):
    hash_value = xxhash.xxh64(str(item), seed=999).intdigest()
    normalized_value = hash_value / (2**64 - 1)  # Convert to [0,1)
    return normalized_value < self.p_activation
```

This ensures consistent treatment during both insertion and lookup—critical for the compression application.

## From Membership Testing to Compression

While Bloom filters weren't designed for compression, they become surprisingly effective under specific conditions. The key insight: when a binary string has a low density of 1s (specifically below p* ≈ 0.32453), we can encode just the positions of those 1s more efficiently than storing the raw string.

The compression algorithm consists of:

1. **Bloom Filter Bitmap**: Marks positions of 1s in the original data
2. **Witness Data**: Records actual bit values for positions that trigger the Bloom filter

### Theoretical Foundation

The compression effectiveness directly depends on the density (p) of 1s in the input:

- Compression is possible when p < p* (≈ 0.32453)
- Optimal hash function count: k = log₂((1-p) * (ln(2)²) / p)
- Optimal Bloom filter size: l = p * n * k * (1/ln(2))

These parameters maximize compression ratio for a given input density.

## Video Delta Compression Architecture

Rather than compressing whole video frames, this system applies Bloom filter compression to frame differences. This capitalizes on temporal coherence—most pixels change little (or not at all) between consecutive frames, creating a sparse difference matrix ideal for this approach.

## Rigorous Result Validation

I've taken several steps to ensure the compression claims are legitimate:

### 1. Complete Accounting of Compressed Data

All decompression requirements are included in the size calculations:

- **Bloom Filter Bitmaps**: The full bit array
- **Witness Data**: All bits needed for perfect reconstruction
- **Metadata**: Frame dimensions, keyframe indices, and parameter values
- **Changed Pixel Values**: All pixel values that differ between frames

### 2. Multi-Level Verification

The validation process is exhaustive:

- **Bit-Perfect Reconstruction**: Decompressed frames must exactly match originals
- **Frame-by-Frame Validation**: Each frame is individually verified
- **Difference Visualization**: Any non-zero differences are visualized and quantified
- **End-to-End Testing**: The entire pipeline is validated from compression to decompression

### 3. Transparent Measurement

The size calculations are straightforward and reproducible:

```python
# For grayscale compression
original_size = sum(frame.nbytes for frame in frames)
compressed_size = os.path.getsize(compressed_path)
compression_ratio = compressed_size / original_size

# For color videos
compressed_gray_size = os.path.getsize(compressed_gray_path)
compressed_color_size = os.path.getsize(compressed_color_path)
compressed_total_size = compressed_gray_size + compressed_color_size
total_ratio = compressed_total_size / original_color_size
```

### 4. Self-Contained System

This approach requires no external data for decompression:

- No dictionaries or lookup tables
- All Bloom filter parameters stored within the compressed data
- Color information fully included in compressed size
- Decompression requires only the compressed files

## Final Notes

Please leave any feedback.
