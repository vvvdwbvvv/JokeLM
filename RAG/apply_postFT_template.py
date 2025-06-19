import json
import os
import concurrent.futures
import time
import re
from faissdb import FaissVectorDB
from request import LiteLMClient

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')

client = LiteLMClient(
    api_key=os.getenv("OLLAMA_API_KEY"),
    endpoint=os.getenv("OLLAMA_ENDPOINT")
)

def get_embeddings_parallel(query):
    models = [
        'text-embedding-ada-002',
        'mxbai-embed-large',
        'bge-large-zh-v1.5',
        'gte-qwen2-7b-instruct:f16'
    ]
    def embed(model):
        try:
            return model, client.get_ollama_embedding(query, model=model)
        except Exception as e:
            print(f"Error getting embedding for model {model}: {e}")
            return model, None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(embed, models))
    return dict(results)

def apply_query(query):
    embeddings = get_embeddings_parallel(query)
    return embeddings

def retrieve_base(query, db):
    embedding_dict = apply_query(query)
    # Use the first valid embedding for the search
    query_vector = next((v for v in embedding_dict.values() if v is not None), None)
    if query_vector is None:
        print("Could not retrieve any valid embeddings for the query.")
        return [], ""
        
    results = db.search(query_vector, k=2)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
    results, context = retrieve_base(query, db)
    if not context:
        return "無法根據主題找到相關的笑話資料。"
        
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    請根據下列笑話資料，找出一則跟主題最相關的，並以自然、有趣的方式講出來（可做輕度改寫），讓讀者覺得好笑。
    <jokes>
    {context}
    </jokes>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
    Answer the question now, and avoid providing preamble such as 'Here is the answer', 笑點可以是諧音、雙關、政治諷刺等，但請避免失禮或攻擊性言論，改寫為更有創意的context。
    """
    response = client.get_ollama_message(
        messages=prompt,
        model='o4-mini'
    )
    return response

def load_topics(filename):
    """Loads topics from a file, filtering out empty lines."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')

    with open(os.path.join(data_dir, '1132NLP_final.json'), 'r') as f:
        data = json.load(f)
    db = FaissVectorDB("1132NLP_final", embedding_key="emb_ada")
    db.load_data(data)
    print("Initialization complete.")

    topic_file = os.path.join(data_dir, 'eval', 'gen_topics.txt')
    topics = load_topics(topic_file)

    instructions = []
    print(f"\nStarting to generate jokes for {len(topics)} topics from '{topic_file}'...")

    for i, topic in enumerate(topics):
        query = f"說一個跟「{topic}」有關的笑話"
        print(f"\n({i+1}/{len(topics)}) Processing topic: {topic}")

        # Step 1: Retrieve relevant context
        retrieved_docs, context = retrieve_base(query, db)
        print(f"  - Retrieved context for '{topic}'")

        example_joke = answer_query_base(query, db)
        instructions.append({
            "topic": topic,
            "query": query,
            "context": context,
            "instruction": "Rewrite the joke based on the provided material in a natural and entertaining way so that readers find it funny. Please remain faithful to the underlying context and only deviate from it if you are 100% certain of the answer. Answer the question immediately and avoid preamble such as ‘Here is the answer.’ The punchline may use puns, wordplay, political satire, etc., but please avoid rude or offensive remarks, and recast it in a more creative context.",
            "example_joke": example_joke,
        })
        
        time.sleep(1)
    


    output_filename = 'generated_post_finetune_jokes.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)

    print(f"\n\nProcessing complete. All results saved to '{output_filename}'.")

