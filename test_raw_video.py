#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2

# Import the compression algorithm
from video_delta_compress import VideoDeltaCompressor

class RawVideoTest:
    """Class to test Bloom filter compression on raw video with controlled noise."""
    
    def __init__(self, output_dir: str = "raw_video_test"):
        """Initialize the test environment."""
        self.output_dir = output_dir
        self.video_compressor = VideoDeltaCompressor()
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_raw_video(self, 
                          frame_count: int = 90, 
                          fps: int = 30,
                          width: int = 640, 
                          height: int = 480,
                          noise_level: float = 1.0,
                          movement_speed: float = 1.0) -> List[np.ndarray]:
        """
        Generate synthetic raw video frames with realistic sensor noise.
        
        Args:
            frame_count: Number of frames to generate
            fps: Frames per second (used for reporting)
            width: Frame width
            height: Frame height
            noise_level: Standard deviation of Gaussian noise (0-255 scale)
            movement_speed: Speed of object movement (higher = faster)
            
        Returns:
            List of generated frames
        """
        print(f"Generating {frame_count} frames of raw video ({width}x{height}) with noise level {noise_level}")
        
        frames = []
        
        # Create a simple scene with moving objects
        for i in range(frame_count):
            # Create base frame (gray background)
            frame = np.ones((height, width), dtype=np.uint8) * 128
            
            # Add some moving objects
            # 1. Moving circle
            circle_x = int(width/2 + width/4 * np.sin(i * 0.05 * movement_speed))
            circle_y = int(height/2 + height/4 * np.cos(i * 0.03 * movement_speed))
            cv2.circle(frame, (circle_x, circle_y), 40, 200, -1)
            
            # 2. Moving rectangle
            rect_x = int(width/2 + width/3 * np.sin(i * 0.02 * movement_speed + 1))
            rect_y = int(height/2 + height/3 * np.cos(i * 0.04 * movement_speed + 2))
            cv2.rectangle(frame, (rect_x-30, rect_y-30), (rect_x+30, rect_y+30), 50, -1)
            
            # Add realistic sensor noise (Gaussian)
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            frames.append(frame)
            
            if i % 30 == 0:
                print(f"Generated {i} frames...")
        
        print(f"Generated {len(frames)} frames of raw video")
        return frames
    
    def analyze_frame_differences(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze differences between consecutive frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing frame differences...")
        
        if len(frames) < 2:
            print("Error: Need at least 2 frames to analyze differences")
            return {}
        
        densities = []
        avg_diffs = []
        max_diffs = []
        hist_data = []
        
        for i in range(1, len(frames)):
            # Calculate frame difference
            diff = np.abs(frames[i].astype(np.int16) - frames[i-1].astype(np.int16))
            
            # Calculate metrics
            avg_diff = np.mean(diff)
            max_diff = np.max(diff)
            nonzero_density = np.count_nonzero(diff) / diff.size
            
            densities.append(nonzero_density)
            avg_diffs.append(avg_diff)
            max_diffs.append(max_diff)
            
            # Collect histogram data periodically
            if i % 10 == 0 or i == len(frames) - 1:
                hist, _ = np.histogram(diff, bins=np.arange(0, 17))  # Count diffs 0-15
                hist_data.append((i, hist))
        
        # Calculate statistics
        avg_density = np.mean(densities)
        avg_difference = np.mean(avg_diffs)
        max_difference = np.max(max_diffs)
        
        # Count frames above the critical threshold for Bloom filter compression
        critical_threshold = 0.32453  # Theoretical threshold for Bloom filter advantage
        frames_above_threshold = sum(1 for d in densities if d > critical_threshold)
        
        result = {
            "frame_count": len(frames),
            "average_density": avg_density,
            "average_difference": avg_difference,
            "max_difference": max_difference,
            "frames_above_threshold": frames_above_threshold,
            "frames_above_threshold_percent": frames_above_threshold / (len(frames) - 1) * 100,
            "densities": densities,
            "avg_diffs": avg_diffs,
            "hist_data": hist_data
        }
        
        # Display basic statistics
        print(f"Average frame difference density: {avg_density:.6f}")
        print(f"Frames above threshold (0.32453): {frames_above_threshold}/{len(frames)-1} "
              f"({result['frames_above_threshold_percent']:.2f}%)")
        
        return result
    
    def plot_difference_analysis(self, analysis: Dict, output_path: str) -> None:
        """
        Create plots visualizing the frame difference analysis.
        
        Args:
            analysis: Analysis results from analyze_frame_differences
            output_path: Path to save the plot
        """
        if not analysis:
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Frame difference densities
        axs[0, 0].plot(range(1, len(analysis["densities"]) + 1), 
                      analysis["densities"], 'o-', markersize=2)
        axs[0, 0].axhline(y=0.32453, color='r', linestyle='--', 
                        label='Critical Threshold (0.32453)')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('Difference Density')
        axs[0, 0].set_title('Frame-to-Frame Difference Densities')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Plot 2: Average difference value
        axs[0, 1].plot(range(1, len(analysis["avg_diffs"]) + 1), 
                      analysis["avg_diffs"], 'o-', color='green', markersize=2)
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('Average Difference (0-255)')
        axs[0, 1].set_title('Average Pixel Difference Between Frames')
        axs[0, 1].grid(True)
        
        # Plot 3: Difference value histograms for selected frames
        if analysis["hist_data"]:
            hist_plot = axs[1, 0]
            for frame_idx, hist in analysis["hist_data"]:
                hist_plot.plot(np.arange(0, 16), hist[:16], 
                             label=f'Frame {frame_idx}', alpha=0.7)
            
            hist_plot.set_xlabel('Difference Value')
            hist_plot.set_ylabel('Frequency')
            hist_plot.set_title('Histogram of Pixel Differences (Values 0-15)')
            hist_plot.set_yscale('log')
            hist_plot.grid(True)
            hist_plot.legend()
        
        # Plot 4: Summary information
        axs[1, 1].axis('off')
        summary_text = (
            f"Frame Difference Analysis:\n\n"
            f"Total Frames: {analysis['frame_count']}\n"
            f"Average Difference Density: {analysis['average_density']:.6f}\n"
            f"Critical Threshold: 0.32453\n\n"
            f"Frames Above Threshold: {analysis['frames_above_threshold']} / {analysis['frame_count']-1}\n"
            f"Percent Above Threshold: {analysis['frames_above_threshold_percent']:.2f}%\n\n"
            f"Average Difference Value: {analysis['average_difference']:.2f}\n"
            f"Maximum Difference Value: {analysis['max_difference']:.2f}\n\n"
            f"Conclusion: {'Not suitable' if analysis['average_density'] > 0.32453 else 'Potentially suitable'} for Bloom filter compression"
        )
        axs[1, 1].text(0.05, 0.95, summary_text, verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Difference analysis plot saved to: {output_path}")
    
    def test_compression(self, frames: List[np.ndarray], experiment_name: str) -> Dict:
        """
        Test Bloom filter compression on the given frames.
        
        Args:
            frames: List of video frames
            experiment_name: Name for the experiment
            
        Returns:
            Dictionary with compression results
        """
        print(f"Testing compression on {len(frames)} frames...")
        
        # Create output paths
        output_path = os.path.join(self.output_dir, f"{experiment_name}_compressed.bin")
        
        # Compress the video
        start_time = time.time()
        compressed_frames, metadata = self.video_compressor.compress_video(
            frames, output_path=output_path)
        compress_time = time.time() - start_time
        
        # Decompress the video
        decompress_start_time = time.time()
        decompressed_frames = self.video_compressor.decompress_video(
            compressed_frames, metadata)
        decompress_time = time.time() - decompress_start_time
        
        # Calculate size metrics
        original_size = sum(frame.nbytes for frame in frames)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = compressed_size / original_size
        
        # Verify lossless
        differences = []
        for i, (orig, decomp) in enumerate(zip(frames, decompressed_frames)):
            if orig.shape != decomp.shape:
                print(f"Error: Frame {i} shape mismatch")
                continue
                
            diff = np.abs(orig.astype(np.int16) - decomp.astype(np.int16))
            mean_diff = np.mean(diff)
            differences.append(mean_diff)
        
        avg_difference = np.mean(differences) if differences else 0
        is_lossless = avg_difference == 0
        
        # Performance metrics
        result = {
            "frame_count": len(frames),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time": compress_time,
            "decompression_time": decompress_time,
            "is_lossless": is_lossless,
            "avg_difference": avg_difference,
            "frames_per_second_compress": len(frames) / compress_time if compress_time > 0 else 0,
            "frames_per_second_decompress": len(frames) / decompress_time if decompress_time > 0 else 0,
            "keyframes": metadata.get('keyframes', []),
            "densities": metadata.get('densities', [])
        }
        
        # Print summary
        print("\nCompression Results:")
        print(f"Original Size: {original_size/1024:.2f} KB")
        print(f"Compressed Size: {compressed_size/1024:.2f} KB")
        print(f"Compression Ratio: {compression_ratio:.6f}")
        print(f"Space Savings: {(1-compression_ratio)*100:.2f}%")
        print(f"Compression Time: {compress_time:.2f} seconds")
        print(f"Decompression Time: {decompress_time:.2f} seconds")
        print(f"Lossless: {is_lossless}")
        if not is_lossless:
            print(f"Average Frame Difference: {avg_difference:.6f}")
        
        return result
    
    def save_sample_frames(self, frames: List[np.ndarray], output_dir: str, prefix: str = "frame") -> None:
        """Save sample frames for visual inspection."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Choose frames to save
        if len(frames) <= 10:
            indices = list(range(len(frames)))
        else:
            step = max(1, len(frames) // 10)
            indices = list(range(0, len(frames), step))
            if indices[-1] != len(frames) - 1:
                indices.append(len(frames) - 1)
        
        # Save frames
        for i in indices:
            frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            Image.fromarray(frames[i]).save(frame_path)
        
        print(f"Sample frames saved to: {output_dir}")
    
    def plot_compression_results(self, results: List[Dict], output_path: str) -> None:
        """
        Create a plot comparing compression results across different noise levels.
        
        Args:
            results: List of result dictionaries from test_compression
            output_path: Path to save the plot
        """
        if not results:
            return
        
        # Extract noise levels and compression ratios
        noise_levels = [r['noise_level'] for r in results]
        ratios = [r['compression_ratio'] for r in results]
        thresholds = [r['frames_above_threshold_percent'] for r in results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot compression ratio
        color = 'tab:blue'
        ax1.set_xlabel('Noise Level (σ)')
        ax1.set_ylabel('Compression Ratio', color=color)
        ax1.plot(noise_levels, ratios, 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        
        # Add secondary y-axis for threshold percentage
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Frames Above Threshold (%)', color=color)
        ax2.plot(noise_levels, thresholds, 'o-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0, top=100)
        
        # Add break-even line
        ax1.axhline(y=1.0, color='gray', linestyle='--', label='Break-even (no compression)')
        
        # Add labels
        for i, txt in enumerate(ratios):
            ax1.annotate(f"{txt:.3f}", (noise_levels[i], ratios[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.title('Effect of Noise Level on Bloom Filter Compression')
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Compression results plot saved to: {output_path}")
    
    def run_noise_level_tests(self, 
                           frame_count: int = 90, 
                           width: int = 640, 
                           height: int = 480,
                           noise_levels: List[float] = None) -> None:
        """
        Run tests with different noise levels to evaluate compression performance.
        
        Args:
            frame_count: Number of frames per test
            width: Frame width
            height: Frame height
            noise_levels: List of noise standard deviations to test
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        
        results = []
        
        for noise_level in noise_levels:
            print(f"\n{'='*40}")
            print(f"Testing noise level: {noise_level}")
            print(f"{'='*40}\n")
            
            # Generate frames with the current noise level
            experiment_name = f"noise_{noise_level:.1f}"
            frames = self.generate_raw_video(
                frame_count=frame_count,
                width=width,
                height=height,
                noise_level=noise_level
            )
            
            # Save sample frames
            samples_dir = os.path.join(self.output_dir, f"{experiment_name}_samples")
            self.save_sample_frames(frames, samples_dir)
            
            # Analyze frame differences
            diff_analysis = self.analyze_frame_differences(frames)
            self.plot_difference_analysis(
                diff_analysis, 
                os.path.join(self.output_dir, f"{experiment_name}_diff_analysis.png")
            )
            
            # Test compression
            compression_result = self.test_compression(frames, experiment_name)
            
            # Combine results
            result = {
                "noise_level": noise_level,
                **compression_result,
                "frames_above_threshold": diff_analysis.get("frames_above_threshold", 0),
                "frames_above_threshold_percent": diff_analysis.get("frames_above_threshold_percent", 0),
                "average_density": diff_analysis.get("average_density", 0)
            }
            
            results.append(result)
        
        # Plot comparative results
        self.plot_compression_results(
            results,
            os.path.join(self.output_dir, "noise_level_comparison.png")
        )
        
        # Save results to CSV for further analysis
        self.save_results_to_csv(results, os.path.join(self.output_dir, "noise_test_results.csv"))
    
    def save_results_to_csv(self, results: List[Dict], output_path: str) -> None:
        """Save test results to a CSV file."""
        import csv
        
        if not results:
            return
        
        # Determine headers from first result
        headers = list(results[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Bloom filter compression on raw video with noise')
    parser.add_argument('--output-dir', type=str, default='raw_video_test_results',
                       help='Directory to save test results')
    parser.add_argument('--frame-count', type=int, default=90,
                       help='Number of frames to generate for each test')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                       default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                       help='Noise levels to test (standard deviation)')
    
    args = parser.parse_args()
    
    # Create test runner
    test = RawVideoTest(output_dir=args.output_dir)
    
    # Run tests with different noise levels
    test.run_noise_level_tests(
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        noise_levels=args.noise_levels
    )
    
    print("\nTests completed. Check the output directory for results.")


if __name__ == "__main__":
    main() 