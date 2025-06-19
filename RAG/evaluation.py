import json
import os
import concurrent.futures
import time
from faissdb import FaissVectorDB
from request import LiteLMClient
import re

client = LiteLMClient(
    api_key=os.getenv("OLLAMA_API_KEY"),
    endpoint=os.getenv("OLLAMA_ENDPOINT")
)

EVALUATION_PROMPT_TEMPLATE = """
You are a meticulous comedy critic. Your task is to evaluate a generated joke based on the source material (context) it was given and the original topic.

Please evaluate the following joke based on three criteria: Originality, Humor, and Relevance.

**Topic:**
{topic}

**Source Material (Context the joke was based on):**
{context}

**Generated Joke to Evaluate:**
{generated_joke}

---
**Evaluation Criteria (Score from 1 to 10):**
1.  **Originality (獨創性):** How creative is the joke? Does it merely copy the source material, or does it offer a unique, clever twist? (1=Copied, 10=Highly Original)
2.  **Humor (好不好笑):** Is the joke genuinely funny? Is the timing and punchline effective? (1=Not Funny, 10=Hilarious)
3.  **Relevance (相關度):** Is the joke strongly related to the given topic? (1=Irrelevant, 10=Perfectly Relevant)

Please provide your evaluation **only in a valid JSON format** like the example below. Do not add any text before or after the JSON object.

{{
  "originality": <score_1_to_10>,
  "humor": <score_1_to_10>,
  "relevance": <score_1_to_10>,
  "justification": "<A brief, one-sentence explanation for your scores in Traditional Chinese.>"
}}
"""

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

def answer_query_base(query, ctx):
    if not ctx:
        return "無法根據主題找到相關的笑話資料。"
        
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    <jokes>
    {ctx}
    </jokes>
    Rewrite the joke based on the provided material in a natural and entertaining way so that readers find it funny. Please remain faithful to the underlying context and only deviate from it if you are 100% certain of the answer. Answer the question immediately and avoid preamble such as ‘Here is the answer.’ The punchline may use puns, wordplay, political satire, etc., but please avoid rude or offensive remarks, and recast it in a more creative context.
    """
    response = client.get_ollama_message(
        messages=prompt,
        model='qwen3:4b'
    )
    return response


def evaluate_joke(topic, ctx, generated_joke):
    """Calls the LLM to evaluate a generated joke and returns the scores."""
    eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        topic=topic,
        context=ctx,
        generated_joke=generated_joke
    )
    
    response_text = client.get_ollama_message(
        messages=eval_prompt,
        model='qwen3:4b'  # You can use the same or a different model for evaluation
    )
    
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            # Extract only the JSON part of the string
            json_string = response_text[json_start:json_end]
            # Parse the extracted JSON string
            evaluation_result = json.loads(json_string)
            return evaluation_result
        else:
            # If no JSON object is found, raise an error to be caught
            raise json.JSONDecodeError("No JSON object found in the response.", response_text, 0)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from evaluation response: {response_text}")
        # Return a default error structure if parsing fails
        return {
            "originality": 0, "humor": 0, "relevance": 0,
            "justification": "Error: Failed to parse LLM evaluation response."
        }
    
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

    topic_file = os.path.join(data_dir, 'eval', 'eval_topics.txt')
    topics = load_topics(topic_file)
    
    all_generated_jokes = []
    print(f"\nStarting to generate jokes for {len(topics)} topics from '{topic_file}'...")

    for i, topic in enumerate(topics):
        query = f"說一個跟「{topic}」有關的笑話"
        print(f"\n({i+1}/{len(topics)}) Processing topic: {topic}")

        # Step 1: Retrieve relevant context
        retrieved_docs, context = retrieve_base(query, db)
        print(f"  - Retrieved context for '{topic}'")

        # Step 2: Generate a joke based on the context        
        generated_joke = answer_query_base(query, context)
        # Remove any <think>...</think> sections from the generated joke
        generated_joke = re.sub(r'<think>.*?</think>', '', generated_joke, flags=re.DOTALL).strip()
        print(f"Generated Joke: {generated_joke}")

        # Step 3: Evaluate the generated joke
        evaluation = evaluate_joke(topic, context, generated_joke)
        print(f"  - Evaluation: {evaluation}")

        all_generated_jokes.append({
            "topic": topic,
            "query": query,
            "context": context,
            "generated_joke": generated_joke,
            "evaluation": evaluation
        })
        
        time.sleep(1)
    
    output_dir = os.path.join(data_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, 'generated_and_evaluated_jokes.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_generated_jokes, f, ensure_ascii=False, indent=2)

    print(f"\n\nProcessing complete. All results saved to '{output_filename}'.")

