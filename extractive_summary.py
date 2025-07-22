from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer as BaseTokenizer
import re
from nltk.tokenize import WordPunctTokenizer

class CustomTokenizer(BaseTokenizer):
    def __init__(self, language="english"):
        self._word_tokenizer = WordPunctTokenizer()

    def to_sentences(self, text):
        # 简单句子切分器（不会用 punkt）
        return re.split(r'(?<=[.!?]) +', text)

    def to_words(self, text):
        # 使用规则型 tokenizer，避免 punkt 依赖
        return self._word_tokenizer.tokenize(text)

def extractive_summary_sumy(texts, num_sentences=5):
    if not texts:
        return "No abstracts provided."

    full_text = " ".join(texts)
    parser = PlaintextParser.from_string(full_text, CustomTokenizer())
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary)
