from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
nltk.download('punkt')

# 提取式摘要函数
def extractive_summary_sumy(abstracts_list, num_sentences=5):
    full_text = " ".join(abstracts_list)
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
