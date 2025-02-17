# RAG Two Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Basic Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/rag_two.git
cd rag_two


2. Create and activate a virtual environment:

For Unix/MacOS:
bash
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

For Windows:
bash
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies:
bash
pip install -r requirements.txt


## Additional Dependencies

### Detectron2

For MacOS:

bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

For Linux/Windows:
bash
python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html


### Tesseract OCR

For MacOS:
bash
brew install tesseract
For Linux:
bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev


For Windows:
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to your system PATH
4. Verify installation: `tesseract --version`

### spaCy Model

Install the English language model:
bash
python -m spacy download en_core_web_sm

## Verification

Verify the installation:
bash

python -m pytest tests/


## Troubleshooting

### Common Issues

1. Detectron2 Installation Fails
   - Ensure you have the correct version of PyTorch installed
   - Try installing from source if wheel installation fails

2. Tesseract Not Found
   - Check if Tesseract is in your system PATH
   - Set the Tesseract path in your code:
     ```python
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
     ```

3. CUDA Issues
   - Ensure CUDA toolkit matches PyTorch version
   - Try CPU-only installation if CUDA setup fails

### Getting Help

If you encounter any issues:
1. Check the [GitHub Issues](https://github.com/Oganesson-Og/rag_two/issues)
2. Create a new issue with:
   - Your system information
   - Installation steps tried
   - Complete error message
   - Any relevant logs

## Optional Dependencies

For development:
bash
pip install black isort flake8 mypy

## Updating

To update to the latest version:

bash
git pull origin main
pip install -r requirements.txt


## License

This project is licensed under the MIT License - see the LICENSE file for details.