#!/usr/bin/env python3
"""
Simple test script to verify that the ImprovedVideoCompressor class can be imported
and used successfully.
"""

import os
import sys
import numpy as np
import cv2

# List all classes in improved_video_compressor module
print("Checking available classes in improved_video_compressor module:")
import improved_video_compressor
print(dir(improved_video_compressor))

# Try to import VideoFrameCompressor class directly
try:
    from improved_video_compressor import VideoFrameCompressor
    print("Successfully imported VideoFrameCompressor")
except ImportError as e:
    print(f"Error importing VideoFrameCompressor: {e}")
    
# Create a simple test image
print("Creating test image")
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_image, (25, 25), (75, 75), (0, 255, 0), -1)

# Save test image
cv2.imwrite("test_image.png", test_image)
print("Saved test image as test_image.png")

# Try to use VideoFrameCompressor directly
print("Testing VideoFrameCompressor")
try:
    compressor = improved_video_compressor.VideoFrameCompressor(verbose=True)
    print("Created VideoFrameCompressor instance")
except Exception as e:
    print(f"Error creating VideoFrameCompressor: {e}")

print("Test complete") 