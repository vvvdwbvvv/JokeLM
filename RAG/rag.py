import json
import os
import concurrent.futures
from faissdb import FaissVectorDB
from request import LiteLMClient

client = LiteLMClient(
    api_key=os.getenv("OLLAMA_API_KEY"),
    endpoint=os.getenv("OLLAMA_ENDPOINT")
)

# Load the Anthropic documentation
with open('data/1132NLP_final.json', 'r') as f:
    data = json.load(f)

# Initialize the VectorDB
db = FaissVectorDB("1132NLP_final",embedding_key="emb_ada") #TODO: add multiple embedding keys iteration for comparing performance
db.load_data(data)

def get_embeddings_parallel(query):
    models = [
        'text-embedding-ada-002',
        'mxbai-embed-large',
        'bge-large-zh-v1.5',
        'gte-qwen2-7b-instruct:f16'
    ]

    def embed(model):
        return model, client.get_ollama_embedding(query, model=model)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(embed, models))
    return dict(results)

def apply_query(query):
    embeddings = get_embeddings_parallel(query)
    return embeddings

def retrieve_base(query, db):
    embedding_dict = apply_query(query)
    query_vector = list(embedding_dict.values())[0]
    results = db.search(query_vector, k=2)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
    results, context = retrieve_base(query, db)
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
    Answer the question now, and avoid providing preamble such as 'Here is the answer', 笑點可以是諧音、雙關、政治諷刺等，但請避免失禮或攻擊性言論，可以改寫為更有創意的context。
    """
    response = client.get_ollama_message(
        messages=prompt,
        model='qwen3:4b' #claude-3-7-sonnet-latest
    )
    print(response)
    return response

if __name__ == "__main__":
    query = "說跟綠豆有關的笑話"
    apply_query(query)
    
    # Retrieve and answer the query using the vector database
    results, context = retrieve_base(query, db)
    print("Retrieved Documents:", results)
    print("Context for the Query:", context)
    
    answer = answer_query_base(query, db)
    print("Answer to the Query:", answer)