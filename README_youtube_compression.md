# YouTube Bloom Filter Video Compression

This tool allows you to download YouTube videos and compress them using the Bloom filter delta compression algorithm.

## Features

- Download videos from YouTube URLs using yt-dlp
- Process entire videos or a specified number of frames
- Memory-efficient batch processing for long videos
- Preprocess videos with options for resolution, frame rate, and binarization
- Compress videos using Bloom filter compression
- Verify lossless decompression
- Generate detailed reports and visualizations

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_youtube.txt
```

Or manually:

```bash
pip install yt-dlp opencv-python numpy matplotlib pillow xxhash imageio
```

## Usage

Basic usage:

```bash
python youtube_bloom_compress.py https://www.youtube.com/watch?v=VIDEO_ID
```

### Advanced Options

```bash
python youtube_bloom_compress.py https://www.youtube.com/watch?v=VIDEO_ID \
    --resolution 360p \
    --fps 30 \
    --max-frames 0 \
    --scale 0.5 \
    --binarize \
    --threshold 127 \
    --batch-size 100 \
    --output-dir youtube_results \
    --experiment-name my_experiment
```

### Command Line Arguments

- `url`: YouTube video URL (required)
- `--resolution`: Video resolution to download (default: 360p)
- `--fps`: Target frames per second (default: 30)
- `--max-frames`: Maximum number of frames to process (0 for all frames, default: 0)
- `--scale`: Scale factor for resizing frames (default: 0.5)
- `--binarize`: Binarize frames for better compression (flag)
- `--threshold`: Threshold for binarization (0-255, default: 127)
- `--batch-size`: Number of frames to process at once (0 for all at once, default: 0)
- `--output-dir`: Output directory for results (default: youtube_compression_results)
- `--experiment-name`: Name for the experiment (default: based on video title)

## Processing Long Videos

For long videos, you can use the batch processing feature to reduce memory usage:

1. Set `--max-frames 0` to process the entire video
2. Set `--batch-size 100` (or another suitable number) to process frames in batches
3. Lower the `--scale` parameter (e.g., 0.25) to reduce frame size
4. Use `--fps 15` or less to reduce the number of frames to process

Example for processing a long video:

```bash
python youtube_bloom_compress.py https://www.youtube.com/watch?v=VIDEO_ID \
    --resolution 360p \
    --fps 15 \
    --scale 0.25 \
    --binarize \
    --batch-size 50
```

## Understanding the Results

The tool generates several files in the output directory:

- `*_compressed.bin`: The compressed video data
- `*_first_frame.png`: The first frame of the video for reference
- `*_decompressed.mp4`: The decompressed video (if ffmpeg is available)
- `*_decompressed_frames/`: Sample frames from the decompressed video
- `*_report.txt`: Detailed compression report
- `*_stats.png`: Visualization of compression statistics
- `*_differences.png`: Visualization of differences between original and decompressed frames (if any)

### Compression Report

The compression report includes:

- Video statistics (frame count, dimensions, file sizes)
- Compression ratio and space savings
- Performance metrics (compression time, frames per second)
- Bloom filter statistics (keyframes, densities, per-frame compression ratios)

### Visualization

The statistics plot shows:

1. Frame difference densities over time
2. Per-frame compression ratios
3. Histogram of difference densities
4. Summary of compression results

## Tips for Better Compression

1. Use the `--binarize` option for videos with simple visual content
2. Adjust the `--threshold` parameter to control the density of 1's in the binarized frames
3. Reduce the resolution with `--scale` for faster processing and better compression
4. Short videos with limited motion tend to compress better

## How It Works

The compression process involves:

1. **Downloading**: Fetches the YouTube video using pytube
2. **Preprocessing**: Converts video to frames, optionally resizes and binarizes them
3. **Compression**: Uses the Bloom filter delta compression algorithm:
   - Detects changes between consecutive frames
   - Compresses these changes using Bloom filters
   - Stores full keyframes periodically
4. **Decompression**: Reverses the process to reconstruct the original video
5. **Verification**: Compares original and decompressed frames to ensure lossless compression

## Limitations

- The compression is optimized for binary (black and white) frames
- Video with complex motion patterns may not compress well
- Compression ratio depends on the content of the video
- Processing large videos can be memory-intensive

## Troubleshooting

If you encounter errors:

1. Ensure you have all required dependencies installed
2. Try a shorter video or fewer frames with `--max-frames`
3. Check if the YouTube URL is valid and accessible
4. Reduce the resolution or scale factor for large videos 