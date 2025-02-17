class OCRConfig:
    # OCR Settings
    TESSERACT_PATH = "path/to/tesseract"
    TESSERACT_LANG = "eng"  # Default language
    
    # Image Preprocessing
    MIN_CONFIDENCE = 60
    PAGE_SEGMENTATION_MODE = 1  # Automatic page segmentation
    
    # Diagram Processing
    DIAGRAM_MIN_SIZE = 100  # Minimum size in pixels
    DIAGRAM_CONFIDENCE = 0.7
    
    # PDF Processing
    PDF_DPI = 300
    ZOOM_FACTOR = 3
    
    # Model Paths
    SCIENTIFIC_MODEL_PATH = "models/scientific_notation"
    CHEMICAL_MODEL_PATH = "models/chemical_structure"
    EQUATION_MODEL_PATH = "models/equation_detection"