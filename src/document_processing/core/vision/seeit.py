"""
Vision Visualization Module
-------------------------

Visualization utilities for document analysis results, providing tools for
drawing bounding boxes, labels, and confidence scores on images.

Key Features:
- Bounding box visualization
- Label rendering
- Confidence score display
- Color-coded class visualization
- Batch result saving
- Custom drawing styles

Technical Components:
1. Drawing Functions:
   - Box drawing with customizable styles
   - Text label rendering
   - Color mapping for classes
   - Image saving utilities
   
2. Visualization Options:
   - Thickness calculation
   - Color generation
   - Text size handling
   - Output formatting

Dependencies:
- PIL>=9.5.0
- numpy>=1.24.0
- logging>=0.5.1.2

Example Usage:
    # Basic result visualization
    save_results(images, results, labels, output_dir='output/')
    
    # Custom visualization
    im = draw_box(
        image,
        detection_result,
        labels,
        threshold=0.5
    )
    
    # Batch processing
    def visualize_batch(images, results, labels):
        for idx, (image, result) in enumerate(zip(images, results)):
            im = draw_box(image, result, labels)
            im.save(f'output_{idx}.jpg')

Visualization Options:
- threshold: Confidence threshold for visualization
- draw_thickness: Line thickness
- color_map: Class-to-color mapping
- output_quality: JPEG quality for saved images

Author: InfiniFlow Team
Version: 1.0.0
License: MIT
"""

import logging
import os
import PIL
from PIL import ImageDraw


def save_results(image_list, results, labels, output_dir='output/', threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, im in enumerate(image_list):
        im = draw_box(im, results[idx], labels, threshold=threshold)

        out_path = os.path.join(output_dir, f"{idx}.jpg")
        im.save(out_path, quality=95)
        logging.debug("save result to: " + out_path)


def draw_box(im, result, lables, threshold=0.5):
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    color_list = get_color_map_list(len(lables))
    clsid2color = {n.lower():color_list[i] for i,n in enumerate(lables)}
    result = [r for r in result if r["score"] >= threshold]

    for dt in result:
        color = tuple(clsid2color[dt["type"]])
        xmin, ymin, xmax, ymax = dt["bbox"]
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(dt["type"], dt["score"])
        tw, th = imagedraw_textsize_c(draw, text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def imagedraw_textsize_c(draw, text):
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tw, th = right - left, bottom - top

    return tw, th
