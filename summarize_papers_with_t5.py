## Summarization with T5-small ##

from transformers import T5Tokenizer, T5ForConditionalGeneration

# 初始化模型和分词器
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
t5_model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")


def summarize_papers_with_t5(papers_df, text_column="original_abstract", max_tokens=400, min_tokens=200):
    print("✅ Received DataFrame:")
    print(papers_df.head())

    if text_column not in papers_df.columns:
        return f"❌ Column '{text_column}' not found in DataFrame."

    abstracts = papers_df[text_column].tolist()

    if not abstracts:
        return "❌ No abstracts found to summarize."
    
    full_text = " ".join(abstracts)
    print("📌 Full input text length:", len(full_text))
    
    # 给 T5 加上任务前缀

    input_text = "summarize: " + full_text.strip()
    
    # 编码输入
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    print (inputs)

    # 生成摘要
    summary_ids = t5_model.generate(
        inputs,
        max_new_tokens=60,
        length_penalty=2.0,
        num_beams=1,
        early_stopping=True
    )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
  
