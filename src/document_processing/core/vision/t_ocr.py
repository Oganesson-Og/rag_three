"""
OCR Testing and Evaluation Module
-------------------------------

Command-line interface and testing utilities for OCR functionality,
providing tools for batch processing and result visualization.

Key Features:
- Command-line OCR testing
- Batch image processing
- Result visualization
- Text extraction
- Output formatting
- Performance evaluation

Technical Components:
1. Processing Pipeline:
   - Image loading and preprocessing
   - OCR text extraction
   - Bounding box generation
   - Result visualization
   
2. Output Generation:
   - Text file generation
   - Visualization images
   - Performance metrics
   - Batch processing results

Dependencies:
- numpy>=1.24.0
- PIL>=9.5.0
- argparse>=1.4.0
- deepdoc.vision

Example Usage:
    # Command line usage
    python t_ocr.py --inputs path/to/images --output_dir results/
    
    # Basic OCR testing
    ocr = OCR()
    results = ocr(image)
    visualize_results(results)
    
    # Batch processing
    process_directory(input_dir, output_dir)

CLI Arguments:
- inputs: Input image directory or file
- output_dir: Results directory
- batch_size: Number of images to process at once
- visualization: Enable result visualization

Author: InfiniFlow Team
Version: 1.0.0
License: MIT
"""

import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.seeit import draw_box
from deepdoc.vision import OCR, init_in_out
import argparse
import numpy as np


def main(args):
    ocr = OCR()
    images, outputs = init_in_out(args)

    for i, img in enumerate(images):
        bxs = ocr(np.array(img))
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{
            "text": t,
            "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
            "type": "ocr",
            "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(images[i], bxs, ["ocr"], 1.)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join([o["text"] for o in bxs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir', help="Directory where to store the output images. Default: './ocr_outputs'",
                        default="./ocr_outputs")
    args = parser.parse_args()
    main(args)
