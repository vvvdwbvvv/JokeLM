import json
import os
import anthropic
from faissdb import FaissVectorDB


client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Load the evaluation dataset
# with open('evaluation/docs_evaluation_dataset.json', 'r') as f:
#     eval_data = json.load(f)

# Load the Anthropic documentation
with open('data/1132NLP_final.json', 'r') as f:
    anthropic_docs = json.load(f)

# Initialize the VectorDB
db = FaissVectorDB("1132NLP_final",embedding_key="emb_ada")
db.load_data(anthropic_docs)

def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2500,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.content[0].text