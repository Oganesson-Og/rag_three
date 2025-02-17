"""
Educational Domain Configuration Module
-----------------------------------

Configuration management for educational domains and subjects.
Features:
- Subject-specific configurations
- Processing rules
- Model mappings
- Educational taxonomy

Author: Keith Satuku
Created: 2024
"""

from typing import Dict

EDUCATION_DOMAINS = {
    'science': {
        'subjects': ['chemistry', 'physics', 'biology'],
        'levels': ['O-Level', 'A-Level'],
        'models': {
            'default': 'scibert',
            'chemistry': 'allenai/scibert_scivocab_uncased',
            'physics': 'allenai/scibert_scivocab_uncased',
            'biology': 'allenai/scibert_scivocab_uncased'
        },
        'preprocessing': [
            'clean_latex',
            'expand_science_abbreviations',
            'normalize_chemical_formulas'
        ],
        'keywords': {
            'chemistry': ['reaction', 'compound', 'acid', 'base', 'molecule'],
            'physics': ['force', 'energy', 'motion', 'wave', 'field'],
            'biology': ['cell', 'organism', 'system', 'evolution', 'genetics']
        }
    },
    'mathematics': {
        'subjects': ['algebra', 'geometry', 'calculus'],
        'levels': ['O-Level', 'A-Level'],
        'models': {
            'default': 'instructor',
            'algebra': 'hkunlp/instructor-large',
            'geometry': 'hkunlp/instructor-large',
            'calculus': 'hkunlp/instructor-large'
        },
        'preprocessing': [
            'clean_latex',
            'normalize_equations',
            'expand_math_symbols'
        ],
        'keywords': {
            'algebra': ['equation', 'function', 'variable', 'polynomial'],
            'geometry': ['angle', 'triangle', 'circle', 'polygon'],
            'calculus': ['derivative', 'integral', 'limit', 'function']
        }
    }
}

PREPROCESSING_CONFIGS = {
    'clean_latex': {
        'patterns': [
            r'\$[^$]+\$',              # Inline math
            r'\\\[[^\]]+\\\]',         # Display math
            r'\\begin\{.*?\}.*?\\end\{.*?\}',  # Environments
        ],
        'replacements': {}
    },
    'normalize_chemical_formulas': {
        'patterns': [
            r'([A-Z][a-z]?)(\d*)',     # Chemical elements with numbers
            r'->',                      # Reaction arrows
        ],
        'replacements': {
            '->': 'yields',
            '+': 'plus'
        }
    },
    'expand_science_abbreviations': {
        'patterns': [],
        'replacements': {
            'aq': 'aqueous',
            'conc': 'concentrated',
            'dil': 'dilute',
            'soln': 'solution'
        }
    }
}

def get_domain_config(subject: str, level: str) -> Dict:
    """Get domain-specific configuration."""
    for domain, config in EDUCATION_DOMAINS.items():
        if subject.lower() in config['subjects']:
            if level in config['levels']:
                return {
                    'domain': domain,
                    'subject': subject,
                    'level': level,
                    'model': config['models'].get(subject.lower(), config['models']['default']),
                    'preprocessing': config['preprocessing'],
                    'keywords': config['keywords'].get(subject.lower(), [])
                }
    raise ValueError(f"Unsupported subject '{subject}' or level '{level}'") 