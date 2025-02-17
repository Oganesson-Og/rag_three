"""
Synonym Processing Module
----------------------

Comprehensive synonym handling system with support for multiple sources
including WordNet and custom dictionaries.

Features:
- WordNet integration
- Custom synonym dictionary
- Redis-based caching
- Regular expression patterns
- Real-time updates

Key Components:
1. WordNet Lookup: Standard synonym database
2. Custom Dictionary: Domain-specific synonyms
3. Cache Management: Redis-based caching
4. Pattern Matching: Regex-based lookup

Technical Details:
- Efficient dictionary lookups
- Redis integration
- Regular expression optimization
- Memory-efficient storage
- Automatic updates

Dependencies:
- nltk>=3.6.0
- redis>=4.0.0
- regex>=2022.1.18

Author: Keith Satuku
Created: 2025
License: MIT
"""

import logging
import json
import os
import time
import re
from nltk.corpus import wordnet
from api.utils.file_utils import get_project_base_directory


class Dealer:
    def __init__(self, redis=None):
        """Initialize the synonym dealer.
        
        Args:
            redis: Optional Redis connection for caching
        """
        self.lookup_num = 100000000
        self.load_tm = time.time() - 1000000
        self.dictionary = None
        
        # Load custom synonym dictionary
        path = os.path.join(get_project_base_directory(), "rag/res", "synonym.json")
        try:
            self.dictionary = json.load(open(path, 'r'))
        except Exception:
            logging.warning("Missing synonym.json")
            self.dictionary = {}

        if not redis:
            logging.warning(
                "Realtime synonym is disabled, since no redis connection.")
        if not len(self.dictionary.keys()):
            logging.warning("Fail to load synonym")

        self.redis = redis
        self.load()

    def load(self):
        """Load synonyms from Redis cache if available."""
        if not self.redis:
            return

        if self.lookup_num < 100:
            return
            
        tm = time.time()
        if tm - self.load_tm < 3600:  # Update every hour
            return

        self.load_tm = time.time()
        self.lookup_num = 0
        
        # Try to load from Redis
        d = self.redis.get("kevin_synonyms")
        if not d:
            return
            
        try:
            d = json.loads(d)
            self.dictionary = d
        except Exception as e:
            logging.error("Fail to load synonym!" + str(e))

    def lookup(self, tk):
        """Look up synonyms for a given token.
        
        Args:
            tk: Token to find synonyms for
            
        Returns:
            List of synonyms
        """
        # Handle simple word lookups through WordNet
        if re.match(r"[a-z]+$", tk):
            res = list(set([
                re.sub("_", " ", syn.name().split(".")[0]) 
                for syn in wordnet.synsets(tk)
            ]) - set([tk]))
            return [t for t in res if t]

        # Update lookup counter and reload if needed
        self.lookup_num += 1
        self.load()
        
        # Look up in custom dictionary
        res = self.dictionary.get(re.sub(r"[ \t]+", " ", tk.lower()), [])
        if isinstance(res, str):
            res = [res]
        return res


if __name__ == '__main__':
    dl = Dealer()
    print(dl.dictionary)


