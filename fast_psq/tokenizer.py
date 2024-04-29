import logging
import re
import string
import unicodedata

from nltk.corpus import stopwords
from mosestokenizer import MosesTokenizer, MosesPunctuationNormalizer

try:
    import jieba
    no_jieba = False
    jieba.setLogLevel(20) # mute the annoying jieba initialization log
except ImportError:
    no_jieba = True

lang_dict = {
    "en":"english", 
    "es": "spanish", 
    "fr": "french", 
    "it": "italian", 
    "nl": "dutch", 
    "de": "german", 
    "fi": "finnish", 
    "ru": "russian", 
    "hi": "hindi", 
    "zh": "chinese", 
    "fa": "persian"
} 

logging.getLogger('mosestokenizer').setLevel(logging.WARNING)
logger = logging.getLogger("fast-psq-tokenize") 

class PSQTokenizer:

    def __init__(self, lang):
        self.lang = lang
        self.tokenizer = MosesTokenizer(lang) 
        self.punct_norm = MosesPunctuationNormalizer(lang)
        try:
            self.stop = set(stopwords.words(lang_dict[lang]))
            self.stop = set().union(*[self.normalize(word, remove_stopwords=False).split() for word in self.stop])
            if len(self.stop) == 0:
                logger.warning(f"Found no stop word for lang `{self.lang}`. This may not be expected; check nltk.")
        except:
            self.stop = None

    def strip_accents(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def tokenize(self, text, lower=True, remove_punct=True, remove_diacritics=True, remove_stopwords=True):
        return self.normalize(text, lower, remove_punct, remove_diacritics, remove_stopwords).strip().split()

    def normalize(self, text, lower=True, remove_punct=True, remove_diacritics=True, remove_stopwords=True):
        if self.lang == 'zh':
            return self.chinese_normalize(text)
    
        text = text.strip()
        if not text:
            return ""
        text = " ".join(self.tokenizer(self.punct_norm(text)))        
        if lower:
            text = text.lower()
        if remove_diacritics:
            text = self.strip_accents(text)
        if remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        if remove_stopwords and self.stop:
            text = " ".join([token for token in text.split() if token not in self.stop])
        text = re.sub(r"\s\s+", " ", text)
        return text
    
    def chinese_normalize(self, text):
        assert not no_jieba, "Need jieba for Chinese tokenization"

        punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        
        text = jieba.cut_for_search(text)
        text = " ".join(text)
        
        text = text.lower()
        text = re.sub(r"[%s]+" %punc, "", text)
        text = re.sub("\s\s+", " ", text)

        return text
