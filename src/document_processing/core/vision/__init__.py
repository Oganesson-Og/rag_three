"""
Document Vision Processing Module
-------------------------------

Comprehensive computer vision system for document analysis, layout understanding,
and text extraction from various document formats.

Key Features:
- Document layout analysis and segmentation
- Optical Character Recognition (OCR)
- Table structure recognition
- Text region detection
- Layout element classification
- Image preprocessing and normalization
- Multi-model inference support

Technical Details:
1. Layout Recognition:
   - YOLOv10-based document layout detection
   - Support for text, titles, figures, tables, headers, footers
   - Layout element relationship analysis
   
2. OCR Processing:
   - Text detection and recognition pipeline
   - CTC-based text recognition
   - Multi-language support
   - Post-processing and text cleanup

3. Table Analysis:
   - Table structure recognition
   - Cell detection and merging
   - Header/row/column identification
   - HTML table generation

4. Core Components:
   - Recognizer: Base class for vision models
   - LayoutRecognizer: Document layout analysis
   - OCR: Text extraction system
   - TableStructureRecognizer: Table parsing
   - Operators: Image processing operations

Dependencies:
- OpenCV
- NumPy
- ONNX Runtime
- Hugging Face Hub
- PIL
- Shapely
- pyclipper

Models:
- Layout detection: YOLOv10
- Text detection: DBNet
- Text recognition: CRNN
- Table structure: Custom DETR

Example Usage:
    # Basic document processing
    from document_processing.core.vision import OCR, LayoutRecognizer, TableStructureRecognizer
    
    # Initialize with default settings
    ocr = OCR()
    layout = LayoutRecognizer()
    table = TableStructureRecognizer()
    
    # Process single document
    results = ocr.extract_text("document.pdf")
    
    # Advanced processing with options
    results = ocr.extract_text(
        document_path="document.pdf",
        language="en",
        enhance_resolution=True,
        denoise=True
    )
    
    # Layout analysis
    layout_results = layout.analyze(
        image_path="document.png",
        confidence_threshold=0.5,
        merge_boxes=True
    )
    
    # Table extraction
    tables = table.extract(
        image_path="document.png",
        output_format="html",
        preserve_styling=True
    )
    
    # Batch processing
    document_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    batch_results = ocr.process_batch(
        document_paths,
        num_workers=4,
        batch_size=8
    )
    
    # Combined pipeline
    def process_document(document_path):
        # Extract text
        text = ocr.extract_text(document_path)
        
        # Analyze layout
        image = load_image(document_path)
        layout_info = layout.analyze(image)
        
        # Extract tables
        tables = table.extract(image)
        
        return {
            "text": text,
            "layout": layout_info,
            "tables": tables
        }
    
    # Process with custom configuration
    config = {
        "ocr": {
            "language": ["en", "fr"],
            "enhance_resolution": True,
            "post_process": True
        },
        "layout": {
            "model_type": "yolov10",
            "confidence_threshold": 0.6,
            "merge_overlapping": True
        },
        "table": {
            "structure_model": "detr",
            "cell_threshold": 0.5,
            "output_format": "pandas"
        }
    }
    
    processor = DocumentVisionPipeline(config)
    results = processor.process("document.pdf")

Processing Options:
    OCR:
        - languages: List of language codes
        - enhance_resolution: Boolean for image enhancement
        - denoise: Boolean for noise reduction
        - post_process: Boolean for text cleanup
    
    Layout:
        - model_type: YOLOv10 or custom model
        - confidence_threshold: Float between 0-1
        - merge_overlapping: Boolean for box merging
        
    Table:
        - structure_model: Model selection
        - output_format: html/pandas/dict
        - preserve_styling: Boolean for style retention
"""

import io
import os
import pdfplumber

from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer




def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def init_in_out(args):
    from PIL import Image
    import os
    import traceback
    images = []
    outputs = []

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    def pdf_pages(fnm, zoomin=3):
        nonlocal outputs, images
        pdf = pdfplumber.open(fnm)
        images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                            enumerate(pdf.pages)]

        for i, page in enumerate(images):
            outputs.append(os.path.split(fnm)[-1] + f"_{i}.jpg")

    def images_and_outputs(fnm):
        nonlocal outputs, images
        if fnm.split(".")[-1].lower() == "pdf":
            pdf_pages(fnm)
            return
        try:
            fp = open(fnm, 'rb')
            binary = fp.read()
            fp.close()
            images.append(Image.open(io.BytesIO(binary)).convert('RGB'))
            outputs.append(os.path.split(fnm)[-1])
        except Exception:
            traceback.print_exc()

    if os.path.isdir(args.inputs):
        for fnm in traversal_files(args.inputs):
            images_and_outputs(fnm)
    else:
        images_and_outputs(args.inputs)

    for i in range(len(outputs)):
        outputs[i] = os.path.join(args.output_dir, outputs[i])

    return images, outputs


__all__ = [
    "OCR",
    "Recognizer",
    "LayoutRecognizer",
    "TableStructureRecognizer",
    "init_in_out",
]
