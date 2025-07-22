from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re

def extractive_summary_sumy(abstracts_list, num_sentences=5):
    full_text = " ".join(abstracts_list)
    
    # 替换 NLTK tokenizer：改用自定义的分句函数
    sentences = re.split(r'(?<=[.!?]) +', full_text.strip())
    full_text_cleaned = " ".join(sentences)

    # 用 sumy 的 parser，tokenizer 仍写 English 但其实已不依赖 punkt
    parser = PlaintextParser.from_string(full_text_cleaned, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary)
