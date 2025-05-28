#!/usr/bin/env python3
"""
Raw Video Acquisition Script

This script helps download and prepare raw video data for scientific testing
of the Bloom filter compression system. It focuses on obtaining truly raw,
uncompressed video from scientific datasets.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import zipfile
import shutil

# URLs for various raw video datasets
RAW_VIDEO_SOURCES = {
    "xiph-derf": {
        "url": "https://media.xiph.org/video/derf/y4m/",
        "description": "Xiph.org Derf collection (YUV4MPEG2 format)",
        "files": [
            "akiyo_cif.y4m",        # 300 frames
            "bridge_close_cif.y4m", # 2000 frames
            "highway_cif.y4m",      # 2000 frames
            "hall_monitor_cif.y4m", # 300 frames
        ]
    },
    "ultravideo-4k": {
        "url": "http://ultravideo.fi/video/Jockey_3840x2160_120fps_420_8bit_YUV.7z",
        "description": "UltraVideo 4K raw YUV sequence (8-bit 4:2:0)",
        "type": "archive",
        "format": "7z"
    },
    "harmonic-test": {
        "url": "https://www.harmonicinc.com/4k-demo-footage-download/",
        "description": "Harmonic 4K demo footage (requires manual download)"
    }
}

class RawVideoAcquisition:
    """Tool for downloading and preparing raw video for testing."""
    
    def __init__(self, output_dir="raw_videos"):
        """Initialize the acquisition tool."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.download_dir = self.output_dir / "downloads"
        self.download_dir.mkdir(exist_ok=True)
        self.converted_dir = self.output_dir / "converted"
        self.converted_dir.mkdir(exist_ok=True)
    
    def download_file(self, url, output_path):
        """
        Download a file from a URL with progress bar.
        
        Args:
            url: URL to download
            output_path: Path to save the file
        """
        print(f"Downloading {url} to {output_path}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def extract_archive(self, archive_path):
        """
        Extract an archive file.
        
        Args:
            archive_path: Path to the archive file
        
        Returns:
            Path to the extracted directory
        """
        print(f"Extracting {archive_path}")
        extract_dir = self.download_dir / Path(archive_path).stem
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        elif archive_path.endswith('.7z'):
            # Use 7z command-line tool if available
            try:
                os.makedirs(extract_dir, exist_ok=True)
                subprocess.run(['7z', 'x', str(archive_path), f'-o{extract_dir}'], check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Error: 7z extraction failed. Please install 7zip or extract manually.")
                return None
        
        else:
            print(f"Unsupported archive format: {archive_path}")
            return None
        
        return extract_dir
    
    def convert_y4m_to_raw(self, y4m_path):
        """
        Convert a Y4M file to raw YUV format using FFmpeg.
        
        Args:
            y4m_path: Path to the Y4M file
        
        Returns:
            Path to the converted file
        """
        print(f"Converting {y4m_path} to raw YUV")
        output_path = self.converted_dir / f"{Path(y4m_path).stem}.yuv"
        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', str(y4m_path),
                '-c:v', 'rawvideo',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ], check=True)
            return output_path
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: FFmpeg conversion failed. Please install FFmpeg or convert manually.")
            return None
    
    def convert_to_raw_rgb(self, input_path, width=None, height=None):
        """
        Convert a video to raw RGB format.
        
        Args:
            input_path: Path to the input video
            width: Frame width (optional)
            height: Frame height (optional)
        
        Returns:
            Path to the converted file
        """
        print(f"Converting {input_path} to raw RGB")
        output_path = self.converted_dir / f"{Path(input_path).stem}_rgb.raw"
        
        cmd = ['ffmpeg', '-i', str(input_path), '-c:v', 'rawvideo', '-pix_fmt', 'rgb24']
        
        if width is not None and height is not None:
            cmd.extend(['-s', f'{width}x{height}'])
        
        cmd.append(str(output_path))
        
        try:
            subprocess.run(cmd, check=True)
            return output_path
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: FFmpeg conversion failed. Please install FFmpeg or convert manually.")
            return None
    
    def prepare_xiph_derf_dataset(self):
        """
        Download and prepare the Xiph Derf dataset.
        
        Returns:
            Dictionary of prepared video files
        """
        print("\nPreparing Xiph Derf dataset:")
        prepared_files = {}
        
        for file in RAW_VIDEO_SOURCES["xiph-derf"]["files"]:
            url = RAW_VIDEO_SOURCES["xiph-derf"]["url"] + file
            download_path = self.download_dir / file
            
            if not download_path.exists():
                self.download_file(url, download_path)
            else:
                print(f"File already exists: {download_path}")
            
            # Convert to raw format
            converted_path = self.convert_y4m_to_raw(download_path)
            if converted_path:
                prepared_files[file] = str(converted_path)
        
        return prepared_files
    
    def prepare_ultravideo_dataset(self):
        """
        Download and prepare the UltraVideo dataset.
        
        Returns:
            Dictionary of prepared video files
        """
        print("\nPreparing UltraVideo dataset:")
        prepared_files = {}
        
        url = RAW_VIDEO_SOURCES["ultravideo-4k"]["url"]
        file_name = url.split('/')[-1]
        download_path = self.download_dir / file_name
        
        if not download_path.exists():
            self.download_file(url, download_path)
        else:
            print(f"File already exists: {download_path}")
        
        # Extract archive
        extract_dir = self.extract_archive(download_path)
        if not extract_dir:
            return prepared_files
        
        # Find YUV files in extracted directory
        yuv_files = list(extract_dir.glob("*.yuv"))
        
        for yuv_file in yuv_files:
            # Get dimensions from filename (typically in format like "video_3840x2160.yuv")
            parts = yuv_file.stem.split('_')
            dimensions = None
            
            for part in parts:
                if 'x' in part and part.replace('x', '').isdigit():
                    dimensions = part
                    break
            
            if dimensions:
                width, height = map(int, dimensions.split('x'))
                # Convert to raw RGB
                converted_path = self.convert_to_raw_rgb(yuv_file, width, height)
                if converted_path:
                    prepared_files[yuv_file.name] = str(converted_path)
            else:
                print(f"Could not determine dimensions for {yuv_file}")
        
        return prepared_files
    
    def prepare_custom_video(self, video_path, width=None, height=None):
        """
        Prepare a custom video for testing.
        
        Args:
            video_path: Path to the video
            width: Frame width (optional)
            height: Frame height (optional)
        
        Returns:
            Path to the prepared video
        """
        print(f"\nPreparing custom video: {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Error: Video file does not exist: {video_path}")
            return None
        
        # Copy to download directory
        shutil.copy(video_path, self.download_dir)
        local_path = self.download_dir / video_path.name
        
        # Convert to raw RGB
        converted_path = self.convert_to_raw_rgb(local_path, width, height)
        
        return converted_path
    
    def list_datasets(self):
        """List available raw video datasets."""
        print("\nAvailable Raw Video Datasets:")
        print("----------------------------")
        
        for key, info in RAW_VIDEO_SOURCES.items():
            print(f"{key}:")
            print(f"  Description: {info['description']}")
            if 'files' in info:
                print(f"  Files: {', '.join(info['files'])}")
            print()

def main():
    """Run the raw video acquisition tool."""
    parser = argparse.ArgumentParser(description="Raw Video Acquisition Tool")
    
    parser.add_argument("--output-dir", type=str, default="raw_videos",
                      help="Directory to save downloaded videos")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List datasets command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    
    # Download dataset command
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument("dataset", type=str, choices=RAW_VIDEO_SOURCES.keys(),
                               help="Dataset to download")
    
    # Prepare custom video command
    custom_parser = subparsers.add_parser("custom", help="Prepare a custom video")
    custom_parser.add_argument("video_path", type=str, help="Path to the video file")
    custom_parser.add_argument("--width", type=int, help="Frame width")
    custom_parser.add_argument("--height", type=int, help="Frame height")
    
    args = parser.parse_args()
    
    # Create acquisition tool
    acquisition = RawVideoAcquisition(output_dir=args.output_dir)
    
    if args.command == "list":
        acquisition.list_datasets()
    
    elif args.command == "download":
        if args.dataset == "xiph-derf":
            prepared_files = acquisition.prepare_xiph_derf_dataset()
            print("\nPrepared files:")
            for name, path in prepared_files.items():
                print(f"  {name} -> {path}")
        
        elif args.dataset == "ultravideo-4k":
            prepared_files = acquisition.prepare_ultravideo_dataset()
            print("\nPrepared files:")
            for name, path in prepared_files.items():
                print(f"  {name} -> {path}")
        
        else:
            print(f"Manual download required for dataset: {args.dataset}")
            print(f"Visit: {RAW_VIDEO_SOURCES[args.dataset]['url']}")
    
    elif args.command == "custom":
        converted_path = acquisition.prepare_custom_video(
            args.video_path, args.width, args.height)
        
        if converted_path:
            print(f"\nPrepared custom video: {converted_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 