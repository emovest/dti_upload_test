from flask import Flask, request, jsonify
from upstash_redis import Redis
import os
from recommend_more import recommend_more_from_liked_paper, mmr, alternative_recommend_more_from_liked_paper
from recommender import predict, recommend_paper
import json
import pandas as pd
from summarize_papers_with_t5 import summarize_papers_with_t5 
from clustering import get_cluster_count, get_top_bigrams
from extractive_summary import extractive_summary_sumy
from topicmodelling import get_topic_count, get_top_words




app = Flask(__name__)

# 初始化 Upstash Redis（确保 Render 设置了这两个环境变量）
redis = Redis(
    url=os.environ.get("UPSTASH_REDIS_REST_URL"),
    token=os.environ.get("UPSTASH_REDIS_REST_TOKEN")
)

@app.route('/')
def home():
    return "✅ Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    # Dialogflow 的结构中，intent name 是在 queryResult 中
    intent = data["queryResult"]["intent"]["displayName"]
    user_input = data["queryResult"]["queryText"]
    user_id = data["session"]  # 可以简化处理

    print(f"🎯 Received intent: {intent}")

    # 如果是主推荐意图
    if intent == "getUserCrytoInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })

    # 如果是主推荐意图2
    if intent == "getUserRealEstateInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })

    # 如果是主推荐意图3
    if intent == "getUserArtsInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })


    # 如果是主推荐意图4
    if intent == "getUserGoldInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })
        
    # 如果是主推荐意图5
    if intent == "getUserQuantInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })
        

    # 如果是请求更多推荐的意图
    elif intent == "getUserIntentforMorePaper":
        liked_text = redis.get(f"{user_id}:liked_text")
        liked_label = redis.get(f"{user_id}:liked_label")

        if liked_text is None or liked_label is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I couldn't find your previous preferences. Please tell me your research interest again."]}}
                ]
            })

        more_papers = recommend_more_from_liked_paper(liked_text, liked_label, top_k=5)
        
        # 提取摘要列表
        abstract_list = more_papers["original_abstract"].tolist()
        
        # 存入 Redis（建议用 json）
        redis.set(f"{user_id}:more_abstracts", json.dumps(abstract_list))

        response_text = "📚 Here are some more papers you might like:\n\n"
        for idx, row in enumerate(more_papers.itertuples(), 1):
            response_text += (
                f"🔹 Paper {idx}:\n"
                f"📄 Title: {row.original_title}\n"
                f"📝 Abstract: {row.original_abstract}\n"
                f"— — — — —\n\n"
            )
            
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ]
        })


    # if not satisfied
    elif intent == "getUserIntentforAlternativePaper":
        liked_text = redis.get(f"{user_id}:liked_text")
        liked_label = redis.get(f"{user_id}:liked_label")

        if liked_text is None or liked_label is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I couldn't find your previous preferences. Please tell me your research interest again."]}}
                ]
            })

        mmr_cosine_recommended = alternative_recommend_more_from_liked_paper(liked_text, liked_label, top_k=5)
        
        # 提取摘要列表
        abstract_list = mmr_cosine_recommended["original_abstract"].tolist()
        
        # 存入 Redis（建议用 json）
        redis.set(f"{user_id}:more_abstracts", json.dumps(abstract_list))
        
        response_text = "📚 Here are some alternative papers you might like:\n\n"
        for idx, row in enumerate(mmr_cosine_recommended.itertuples(), 1):
            response_text += (
                f"🔹 Paper {idx}:\n"
                f"📄 Title: {row.original_title}\n"
                f"📝 Abstract: {row.original_abstract}\n"
                f"— — — — —\n\n"
            )
            
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ]
        })

    
    elif intent == "getSummary":

        more_abstracts = redis.get(f"{user_id}:more_abstracts")
        
        # 合并 DataFrame
        abstracts_list = json.loads(more_abstracts)
        df_to_summarize = pd.DataFrame(abstracts_list, columns=["original_abstract"])

        # 调用 T5 摘要函数
        summary_text = summarize_papers_with_t5(df_to_summarize)
        print("===SUMMARY TEXT===")
        print(summary_text)
        print(repr(summary_text))
        print(type(summary_text))
        
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": ["📄 Summary of Selected Papers:"]}},
                {"text": {"text": [summary_text]}}
            ]
        })

    
    elif intent == "getExtraSummary":

        more_abstracts = redis.get(f"{user_id}:more_abstracts")
        
        # 合并 DataFrame
        abstracts_list = json.loads(more_abstracts)
        
        # ⬇️ 改用 extractive summarizer，而不是 T5
        summary_text = extractive_summary_sumy(abstracts_list)
        print("===SUMMARY TEXT===")
        print(summary_text)
        print(repr(summary_text))
        print(type(summary_text))
        
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": ["📄 Summary of Selected Papers:"]}},
                {"text": {"text": [summary_text]}}
            ]
        })


    elif intent == "getCryptoClustering":
        label = "a"  # 写死 label
        count = get_cluster_count(label)

        all_keywords = []  # 初始化列表
        
        for cluster_id in range(count):
            top_bigrams = get_top_bigrams(label, cluster_id)
            bigram_str = f"Cluster {cluster_id+1}: " + ", ".join(top_bigrams)
            all_keywords.append(bigram_str)
        
        keywords_text = "\n".join(all_keywords)
        
        return jsonify({
            "fulfillmentText": f"There are {count} main clusters in the Crypto domain.\n\nTop keywords for each cluster:\n{keywords_text}"
        })

    elif intent == "getRealEstateClustering":
        label = "b"  # 写死 label
        count = get_cluster_count(label)

        all_keywords = []  # 初始化列表
        
        for cluster_id in range(count):
            top_bigrams = get_top_bigrams(label, cluster_id)
            bigram_str = f"Cluster {cluster_id+1}: " + ", ".join(top_bigrams)
            all_keywords.append(bigram_str)
        
        keywords_text = "\n".join(all_keywords)
        
        return jsonify({
            "fulfillmentText": f"There are {count} main clusters in the Crypto domain.\n\nTop keywords for each cluster:\n{keywords_text}"
        })
        
    elif intent == "getGoldClustering":
        label = "b"  # 写死 label
        count = get_cluster_count(label)

        all_keywords = []  # 初始化列表
        
        for cluster_id in range(count):
            top_bigrams = get_top_bigrams(label, cluster_id)
            bigram_str = f"Cluster {cluster_id+1}: " + ", ".join(top_bigrams)
            all_keywords.append(bigram_str)
        
        keywords_text = "\n".join(all_keywords)
        
        return jsonify({
            "fulfillmentText": f"There are {count} main clusters in the Crypto domain.\n\nTop keywords for each cluster:\n{keywords_text}"
        })

    elif intent == "getArtsClustering":
        label = "b"  # 写死 label
        count = get_cluster_count(label)

        all_keywords = []  # 初始化列表
        
        for cluster_id in range(count):
            top_bigrams = get_top_bigrams(label, cluster_id)
            bigram_str = f"Cluster {cluster_id+1}: " + ", ".join(top_bigrams)
            all_keywords.append(bigram_str)
        
        keywords_text = "\n".join(all_keywords)
        
        return jsonify({
            "fulfillmentText": f"There are {count} main clusters in the Crypto domain.\n\nTop keywords for each cluster:\n{keywords_text}"
        })

    elif intent == "getStockClustering":
        label = "b"  # 写死 label
        count = get_cluster_count(label)

        all_keywords = []  # 初始化列表
        
        for cluster_id in range(count):
            top_bigrams = get_top_bigrams(label, cluster_id)
            bigram_str = f"Cluster {cluster_id+1}: " + ", ".join(top_bigrams)
            all_keywords.append(bigram_str)
        
        keywords_text = "\n".join(all_keywords)
        
        return jsonify({
            "fulfillmentText": f"There are {count} main clusters in the Crypto domain.\n\nTop keywords for each cluster:\n{keywords_text}"
        })

    
    elif intent == "getCryptoTopic":
        label = "a"  # 
        count = get_topic_count(label)
        
        topic_lines = [f"📚 There are {count} main topics in this domain:\n"]
        
        for topic_id in range(count):
            top_words = get_top_words(label, topic_id)
            if not top_words:
                continue
            
            topic_str = f"🔹 *Topic {topic_id+1}*:\n" + ", ".join(top_words)
            topic_lines.append(topic_str)
            
        topics_text = "\n\n".join(topic_lines)
        
        return jsonify({
            "fulfillmentText": topics_text
        })

    elif intent == "getRealEstateTopic":
        label = "b"  # 
        count = get_topic_count(label)
        
        topic_lines = [f"📚 There are {count} main topics in this domain:\n"]
        
        for topic_id in range(count):
            top_words = get_top_words(label, topic_id)
            if not top_words:
                continue
            
            topic_str = f"🔹 *Topic {topic_id+1}*:\n" + ", ".join(top_words)
            topic_lines.append(topic_str)
            
        topics_text = "\n\n".join(topic_lines)
        
        return jsonify({
            "fulfillmentText": topics_text
        })

    
    elif intent == "getGlodTopic":
        label = "c"  # 
        count = get_topic_count(label)
        
        topic_lines = [f"📚 There are {count} main topics in this domain:\n"]
        
        for topic_id in range(count):
            top_words = get_top_words(label, topic_id)
            if not top_words:
                continue
            
            topic_str = f"🔹 *Topic {topic_id+1}*:\n" + ", ".join(top_words)
            topic_lines.append(topic_str)
            
        topics_text = "\n\n".join(topic_lines)
        
        return jsonify({
            "fulfillmentText": topics_text
        })

    
    elif intent == "getArtsTopic":
        label = "e"  # 
        count = get_topic_count(label)
        
        topic_lines = [f"📚 There are {count} main topics in this domain:\n"]
        
        for topic_id in range(count):
            top_words = get_top_words(label, topic_id)
            if not top_words:
                continue
            
            topic_str = f"🔹 *Topic {topic_id+1}*:\n" + ", ".join(top_words)
            topic_lines.append(topic_str)
            
        topics_text = "\n\n".join(topic_lines)
        
        return jsonify({
            "fulfillmentText": topics_text
        })

    
    elif intent == "getStockTopic":
        label = "e"  # 
        count = get_topic_count(label)
        
        topic_lines = [f"📚 There are {count} main topics in this domain:\n"]
        
        for topic_id in range(count):
            top_words = get_top_words(label, topic_id)
            if not top_words:
                continue
            
            topic_str = f"🔹 *Topic {topic_id+1}*:\n" + ", ".join(top_words)
            topic_lines.append(topic_str)
            
        topics_text = "\n\n".join(topic_lines)
        
        return jsonify({
            "fulfillmentText": topics_text
        })

    
    # 兜底情况
    else:
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": ["❓ I didn't quite understand that."]}}
            ]
        })

if __name__ == '__main__':
    app.run(debug=True)
