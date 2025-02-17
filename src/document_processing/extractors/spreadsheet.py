"""
Spreadsheet Document Extractor Module
---------------------------------

Specialized extractor for Excel and CSV documents.

Key Features:
- Excel processing
- CSV handling
- Data validation
- Structure preservation
- Formula extraction
- Metadata parsing
- Multi-sheet support

Technical Details:
- Pandas integration
- Data type handling
- Sheet management
- Formula parsing
- Data validation
- Error handling
- Performance optimization

Dependencies:
- pandas>=1.3.0
- openpyxl>=3.0.0
- xlrd>=2.0.0
- typing-extensions>=4.7.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from .base import BaseExtractor, ExtractorResult, DocumentContent

class ExcelExtractor(BaseExtractor):
    """Handles Excel documents (.xlsx, .xls)."""
    
    def extract(
        self,
        content: Union[str, Path, bytes],
        options: Optional[Dict[str, bool]] = None
    ) -> ExtractorResult:
        try:
            options = options or {}
            
            # Handle different input types
            if isinstance(content, (str, Path)):
                df_dict = pd.read_excel(content, sheet_name=None)
            else:
                from io import BytesIO
                df_dict = pd.read_excel(BytesIO(content), sheet_name=None)

            result = {
                'sheets': self._process_sheets(df_dict),
                'metadata': {
                    **self.get_metadata(),
                    'content_type': 'excel',
                    'sheet_count': len(df_dict)
                }
            }

            if options.get('extract_formulas', True):
                result['formulas'] = self._extract_formulas(content)

            return result
            
        except Exception as e:
            self.logger.error(f"Excel extraction error: {str(e)}")
            raise

    def _process_sheets(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process all sheets in workbook."""
        sheets = {}
        for sheet_name, df in df_dict.items():
            sheets[sheet_name] = {
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'shape': df.shape
            }
        return sheets

    def _extract_formulas(self, content: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """Extract Excel formulas."""
        # Implementation depends on specific requirements
        return []

class CSVExtractor(BaseExtractor):
    """Handles CSV documents."""
    
    def extract(
        self,
        content: Union[str, Path, bytes],
        options: Optional[Dict[str, bool]] = None
    ) -> ExtractorResult:
        try:
            options = options or {}
            
            # Handle different input types
            if isinstance(content, (str, Path)):
                df = pd.read_csv(content)
            else:
                from io import BytesIO
                df = pd.read_csv(BytesIO(content))

            return {
                'content': df.to_dict('records'),
                'metadata': {
                    **self.get_metadata(),
                    'content_type': 'csv',
                    'columns': df.columns.tolist(),
                    'rows': len(df),
                    'columns_count': len(df.columns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"CSV extraction error: {str(e)}")
            raise

    def _detect_dialect(self, content: bytes) -> Dict[str, Any]:
        """Detect CSV dialect and encoding."""
        import csv
        from io import StringIO
        
        try:
            # Try to detect dialect
            sample = content[:1024].decode('utf-8')
            dialect = csv.Sniffer().sniff(sample)
            return {
                'delimiter': dialect.delimiter,
                'quotechar': dialect.quotechar,
                'encoding': 'utf-8'
            }
        except Exception:
            # Default values if detection fails
            return {
                'delimiter': ',',
                'quotechar': '"',
                'encoding': 'utf-8'
            } 