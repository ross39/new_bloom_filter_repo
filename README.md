# Rational Bloom Filter Video Compression

A novel lossless video compression method based on rational Bloom filters that achieves significant space savings while guaranteeing perfect bit-exact reconstruction.

## Overview

This project implements a lossless video compression scheme using rational Bloom filters - a probabilistic data structure that allows for efficient representation of binary data. The key innovation is the use of non-integer (rational) hash functions in the Bloom filter, which theoretically enables better compression than traditional methods.

The compression system targets raw video content (Y4M, YUV, HDR, etc.) and provides:

- **True lossless compression** with bit-exact reconstruction
- **Space savings of 40-50%** on typical video content
- **Efficient encoding and decoding** with multi-threaded support
- **Support for various color spaces** (RGB, BGR, YUV)
- **Handling of high dynamic range (HDR)** content(This needs some work to make it fast and usable)

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - opencv-python
  - matplotlib
  - pandas
  - tqdm
  - requests
  - xxhash
  - Pillow
  - scikit-image
  - pyexr (for HDR support)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Compression and Decompression

```python
from improved_video_compressor import ImprovedVideoCompressor

# Initialize compressor
compressor = ImprovedVideoCompressor(
    noise_tolerance=10.0,
    keyframe_interval=30,
    use_direct_yuv=True,
    verbose=True
)

# Compress a video
compressor.compress_video(
    input_file="input_video.y4m",
    output_file="compressed.bfvc"
)

# Decompress a video
compressor.decompress_video(
    input_file="compressed.bfvc",
    output_file="decompressed.mp4"
)

# Verify lossless decompression
original_frames = compressor.extract_frames_from_video("input_video.y4m")
decompressed_frames = compressor.decompress_video("compressed.bfvc")
verification = compressor.verify_lossless(original_frames, decompressed_frames)
print(f"Lossless: {verification['lossless']}")
```

### Command Line Interface

```bash
# Compress a video
python -m improved_video_compressor compress input_video.y4m output.bfvc --max-frames 30

# Decompress a video
python -m improved_video_compressor decompress output.bfvc decompressed.mp4

# Process raw YUV file
python -m improved_video_compressor process-yuv input.yuv output.bfvc --width 1920 --height 1080 --format YUV444
```

## Benchmarking

The project includes a comprehensive benchmarking system that compares the Rational Bloom Filter compression with other lossless compression methods like FFV1, HuffYUV, and H.264 (lossless mode).

```bash
# Run the benchmark
python benchmark_compression.py

# Run benchmark with specific datasets and methods
python benchmark_compression.py --datasets y4m --methods bloom ffv1 --max-frames 10
```

See [results.md](results.md) for detailed benchmark results and instructions on how to reproduce them.

## How It Works

The compression scheme works through the following steps:

1. **Frame Extraction**: Extract frames from the input video
2. **Keyframe Selection**: Store keyframes as direct zlib-compressed frames
3. **Bloom Filter Compression**: For inter-frames, compress difference maps using rational Bloom filters
4. **Lossless Verification**: Verify bit-exact reconstruction during decompression

The rational Bloom filter uses a non-integer number of hash functions (k*) to optimize the space-accuracy tradeoff. This is implemented by using ⌊k*⌋ hash functions deterministically, plus an additional hash function applied with probability (k* - ⌊k*⌋).

## Project Structure

- `improved_video_compressor.py` - Main implementation of the compression algorithm
- `verify_true_lossless.py` - Script to verify lossless reconstruction
- `benchmark_compression.py` - Benchmark system comparing different methods
- `download_*.py` - Scripts to download test datasets
- `results.md` - Detailed benchmark results and analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{rationalbloom2023,
  author = {Author},
  title = {Rational Bloom Filter Video Compression},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/username/rational-bloom-filter-compression}
}
```
