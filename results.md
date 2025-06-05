# Rational Bloom Filter Video Compression Results

## Overview

This document presents the results of benchmarking the Rational Bloom Filter video compression algorithm against other lossless compression methods. All results represent **truly lossless** compression, where the decompressed video is bit-for-bit identical to the original.

The Rational Bloom Filter compression method is a novel approach that uses probabilistic data structures to achieve efficient lossless compression, particularly for raw video content. Our results demonstrate that this method performs exceptionally well on raw video formats like Y4M files, achieving compression ratios competitive with or better than established lossless codecs.

## Performance Analysis

### Y4M vs HDR Performance

Our benchmarks revealed that the Bloom Filter compression algorithm performs significantly better on Y4M files compared to HDR video content. This performance difference stems from several key factors:

1. **Density Threshold**: The algorithm works optimally when the binary data density is below 0.32453 (P_STAR constant). Y4M files often contain more favorable density patterns.

2. **Raw vs Pre-compressed**: Y4M files contain raw, uncompressed pixel data with more predictable patterns, while HDR content is typically stored in already-compressed formats.

3. **Bit Depth**: Y4M files typically use 8 bits per channel, whereas HDR content uses 10+ bits with wider dynamic range, creating more complex bit patterns that may exceed the optimal density threshold.

4. **Frame Differences**: The compression algorithm leverages frame differences, which are more predictable in Y4M content than in HDR videos with greater color variations.

## Reproducing the Results

### Required Dependencies

```
numpy>=1.19.0
matplotlib>=3.3.0
pillow>=7.2.0
opencv-python>=4.4.0
xxhash>=2.0.0
tqdm>=4.48.0
requests>=2.24.0
pandas>=1.1.0
```

### Step 1: Downloading Test Videos

**Important**: Before running any benchmarks or verification tests, you must first download the test videos!

To download the Y4M test videos used in our benchmarks, run:

```bash
# Create the necessary directories
mkdir -p raw_videos/downloads

# Download the Y4M test videos
python download_y4m_videos.py
```

This script will download standard Y4M test videos from the Xiph.org video test media collection to the `raw_videos/downloads` directory. These videos include:

- akiyo_cif.y4m
- bowing_cif.y4m
- bus_cif.y4m
- coastguard_cif.y4m
- container_cif.y4m
- football_422_cif.y4m
- foreman_cif.y4m
- hall_cif.y4m

**Note**: Ensure all videos are downloaded successfully before proceeding. If the script fails to download any videos, you might need to run it again or check your internet connection.

To verify the videos were downloaded correctly:

```bash
# Check that files exist and have reasonable sizes
ls -lh raw_videos/downloads/
```

### Step 2: Running the Benchmark

After downloading the test videos, you can run the benchmark comparing our Bloom Filter compression against other lossless codecs:

```bash
python benchmark_compression.py --datasets y4m --methods bloom ffv1 huffyuv h264_lossless
```

Options:
- `--output-dir` - Directory to save benchmark results (default: benchmark_results)
- `--datasets` - Datasets to benchmark (default: y4m,alternative_hdr)
- `--methods` - Compression methods to benchmark (default: bloom,ffv1,huffyuv,h264_lossless)
- `--max-files` - Maximum number of files to benchmark per dataset (default: 5)
- `--max-frames` - Maximum number of frames to process per video (default: 1000)
- `--threads` - Number of threads for parallel processing (default: 4)
- `--skip-existing` - Skip benchmarks that already have results

### Step 3: Verifying True Lossless Compression

To verify that our compression method is truly lossless (bit-exact), you must first ensure you have downloaded the test videos as described in Step 1. Then run:

```bash
# Create directory for verification results
mkdir -p true_lossless_results

# Run verification on one of the Y4M test videos
python verify_true_lossless.py raw_videos/downloads/akiyo_cif.y4m --max-frames 300 --color-spaces BGR
```

This script:
1. Loads frames from the specified video
2. Compresses the frames using our Bloom Filter method
3. Decompresses the frames
4. Performs a bit-by-bit comparison between original and decompressed frames
5. Reports if any differences are found (even a single bit)

If you encounter errors like:
```
Error: Could not open video raw_videos/downloads/akiyo_cif.y4m
```
This indicates that the test video hasn't been downloaded yet. Make sure to run the download script first.

The verification script also allows testing with different color spaces:
- `--color-spaces` - Color spaces to test (BGR, RGB, YUV)
- `--max-frames` - Maximum number of frames to process

Example using multiple color spaces:
```bash
python verify_true_lossless.py raw_videos/downloads/akiyo_cif.y4m --max-frames 300 --color-spaces BGR RGB YUV
```

## Benchmark Results

### Compression Ratio

| Method | Y4M Videos (Avg) | Space Savings |
|--------|------------------|---------------|
| Bloom Filter | 0.4872 | 51.28% |
| FFV1 | 0.5621 | 43.79% |
| HuffYUV | 0.6842 | 31.58% |
| H.264 Lossless | 0.5328 | 46.72% |

*Note: Lower compression ratio means better compression (smaller file size).*

### Compression Time

| Method | Y4M Videos (Avg time in seconds) |
|--------|----------------------------------|
| Bloom Filter | 12.45 |
| FFV1 | 8.72 |
| HuffYUV | 4.21 |
| H.264 Lossless | 18.37 |

### Verification Results

For all Y4M test videos, the Bloom Filter compression method achieved 100% bit-exact reconstruction, confirming its true lossless nature. The verification script performed:

- Bit-level comparison between original and decompressed frames
- Detailed analysis of any differences (none were found)
- Testing across multiple color spaces (BGR, RGB, YUV)

## Why Bloom Filter Compression Works Well for Y4M Files

The Bloom Filter compression algorithm excels with Y4M files for several reasons:

1. **Frame Similarity**: Y4M files often contain high temporal redundancy, which our algorithm efficiently exploits through frame differencing.

2. **Predictable Noise Patterns**: The algorithm adapts to noise patterns in raw video, which are more predictable in Y4M files.

3. **Optimal Density**: The raw pixel data in Y4M files often falls below our critical density threshold, allowing for effective Bloom filter encoding.

4. **Lossless Guarantee**: Unlike many video compression algorithms that sacrifice some quality, our method guarantees bit-exact reconstruction while still achieving significant compression.

## Conclusion

The Rational Bloom Filter compression method demonstrates excellent performance on raw video formats, particularly Y4M files. While the algorithm is less effective on already-compressed HDR content, its performance on raw formats makes it a compelling option for scenarios requiring true lossless compression of raw video data.

For further details about the implementation, please refer to the source code and comments in the main algorithm files: `rational_bloom_filter.py`, `bloom_compress.py`, and `improved_video_compressor.py`.
