## Summarization with T5-small ##

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 初始化模型和分词器
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
t5_model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")

# Warm-up 预热一次模型，避免 cold start 慢
dummy = t5_tokenizer("summarize: hello world", return_tensors="pt")
with torch.no_grad():
    _ = t5_model.generate(**dummy)
    
print("✅ T5 model warm-up complete.")

def summarize_papers_with_t5(papers_df, text_column="original_abstract", max_tokens=400, min_tokens=200):
    abstracts = papers_df[text_column].tolist()
    full_text = " ".join(abstracts)
    
    # 给 T5 加上任务前缀
    full_text = full_text[:384]
    input_text = "summarize: " + full_text.strip()
    
    # 编码输入
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=384, truncation=True)

    # 生成摘要
    summary_ids = t5_model.generate(
        inputs,
        max_new_tokens=100,
        length_penalty=2.0,
        num_beams=1,
        early_stopping=True
    )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
  
