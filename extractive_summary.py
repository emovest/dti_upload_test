from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer as BaseTokenizer
import re

class CustomTokenizer(BaseTokenizer):
    def __init__(self, language="english"):
        pass
    def to_sentences(self, text):
        # 简单句子切分器（不会用 punkt）
        return re.split(r'(?<=[.!?]) +', text)

def extractive_summary_sumy(texts, num_sentences=5):
    if not texts:
        return "No abstracts provided."

    full_text = " ".join(texts)
    parser = PlaintextParser.from_string(full_text, CustomTokenizer())
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary)
