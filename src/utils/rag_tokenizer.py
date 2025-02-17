"""
RAG Tokenizer Module
----------------
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
from datetime import datetime
import copy
import datrie
import math
import os
import re
import string
import sys
from hanziconv import HanziConv
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pathlib import Path



def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


class RagTokenizer:
    """RAG-specific tokenization utilities."""
    
    def __init__(
        self,
        dict_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> None:
        self.config = config or {}
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Set dictionary path
        if dict_path:
            self.dict_path = dict_path
        else:
            # Try test fixtures first, then default path
            test_dict = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'tests', 'fixtures', 'huqie.txt'
            )
            if os.path.exists(test_dict):
                self.dict_path = test_dict
            else:
                self.dict_path = os.path.join(
                    os.path.dirname(__file__),
                    '..', '..', 'rag', 'res', 'huqie.txt'
                )
        
        # Initialize components
        self.denominator = 1000000
        self.dir = os.path.join(self._get_base_dir(), "rag/res", "huqie")
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.split_pattern = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；''：""【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"
        
        # Initialize trie
        self._init_trie()

    def _get_base_dir(self) -> str:
        """Get project base directory."""
        return os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir
            )
        )

    def _init_trie(self) -> None:
        """Initialize trie data structure."""
        try:
            trie_file = f"{self.dict_path}.trie"
            
            if os.path.exists(trie_file):
                self.trie = datrie.Trie.load(trie_file)
                self.logger.info(f"Loaded trie from cache: {trie_file}")
            else:
                if os.path.exists(self.dict_path):
                    self._load_dict(self.dict_path)
                    self.logger.info(f"Built trie from dictionary: {self.dict_path}")
                else:
                    self.logger.warning(f"No dictionary found at {self.dict_path}, using empty trie")
                    
        except Exception as e:
            self.logger.warning(f"Trie initialization warning: {str(e)}")
            # Continue with empty trie rather than failing

    def _load_dict(self, file_path: str) -> None:
        """Load dictionary file into trie."""
        try:
            self.logger.info(f"Building trie from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = re.split(r"[ \t]", line.strip())
                    if len(parts) >= 3:
                        key = self._get_key(parts[0])
                        freq = int(math.log(float(parts[1]) / self.denominator) + 0.5)
                        
                        if key not in self.trie or self.trie[key][0] < freq:
                            self.trie[key] = (freq, parts[2])
                            
                        self.trie[self._get_reverse_key(parts[0])] = 1
                            
            # Save trie cache
            cache_file = f"{file_path}.trie"
            self.logger.info(f"Saving trie cache to {cache_file}")
            self.trie.save(cache_file)
            
        except Exception as e:
            self.logger.error(f"Dictionary loading error: {str(e)}")
            raise

    def _get_key(self, text: str) -> str:
        """Get trie key for text."""
        return str(text.lower().encode("utf-8"))[2:-1]

    def _get_reverse_key(self, text: str) -> str:
        """Get reverse trie key for text."""
        return str(("DD" + (text[::-1].lower())).encode("utf-8"))[2:-1]

    def tokenize(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> str:
        """Tokenize text."""
        try:
            options = options or {}
            
            # Clean and normalize text
            text = re.sub(r"\W+", " ", text)
            text = self._convert_fullwidth(text).lower()
            text = self._convert_traditional(text)
            
            # Count Chinese characters
            chinese_count = sum(1 for c in text if self._is_chinese(c))
            
            # Handle non-Chinese text
            if chinese_count == 0:
                tokens = word_tokenize(text)
                return " ".join(self._normalize_english(tokens))
            
            # Process text segments
            segments = re.split(self.split_pattern, text)
            result = []
            
            for segment in segments:
                if not segment:
                    continue
                    
                if len(segment) < 2 or re.match(r"[a-z\.-]+$", segment) or re.match(r"[0-9\.-]+$", segment):
                    result.append(segment)
                    continue
                
                # Process Chinese segment
                tokens = self._process_chinese_segment(segment)
                result.extend(tokens)
            
            return " ".join(result)
            
        except Exception as e:
            self.logger.error(f"Tokenization error: {str(e)}")
            raise

    def _convert_fullwidth(self, text: str) -> str:
        """Convert fullwidth characters to halfwidth."""
        result = ""
        for char in text:
            code = ord(char)
            if code == 0x3000:
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result += chr(code)
        return result

    def _convert_traditional(self, text: str) -> str:
        """Convert traditional Chinese to simplified."""
        return HanziConv.toSimplified(text)

    def _is_chinese(self, char: str) -> bool:
        """Check if character is Chinese."""
        return '\u4e00' <= char <= '\u9fff'

    def _normalize_english(self, tokens: List[str]) -> List[str]:
        """Normalize English tokens."""
        return [
            self.stemmer.stem(self.lemmatizer.lemmatize(token))
            for token in tokens
        ]

    def _process_chinese_segment(self, text: str) -> List[str]:
        """Process Chinese text segment."""
        # Implement Chinese segmentation logic
        return [text]  # Placeholder

    def key_(self, line):
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm):
        logging.info(f"[HUQIE]:Build trie from {fnm}")
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1

            dict_file_cache = fnm + ".trie"
            logging.info(f"[HUQIE]:Build trie cache to {dict_file_cache}")
            self.trie_.save(dict_file_cache)
            of.close()
        except Exception:
            logging.exception(f"[HUQIE]:Build trie {fnm} failed")

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def _strQ2B(self, ustring):
        """Convert full-width characters to half-width characters"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # After the conversion, if it's not a half-width character, return the original character.
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        #F /= len(tks)
        L /= len(tks)
        logging.debug("[SC] {} {} {} {} {}".format(tks, len(tks), L, F, B / len(tks) + L + F))
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split()
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def fine_grained_tokenize(self, tks):
        tks = tks.split()
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split():
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
loadUserDict = tokenizer.loadUserDict
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    tks = tknzr.tokenize(
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("虽然我不怎么玩")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    logging.info(tknzr.fine_grained_tokenize(tks))
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.DEBUG = False
    tknzr.loadUserDict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        logging.info(tknzr.tokenize(line))
    of.close()
