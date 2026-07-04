from langsmith import traceable, get_current_run_tree
import openai
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(url ="http://qdrant:6333")

##define embedding function
@traceable(
        name = "embed_query",
        run_type= "embedding",
        metadata = {
            "ls_provider": "openai",
            "ls_model_name": "text-embedding-3-small"
        }
) 
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding



##retreival function
@traceable(
        name = "retrieve_data",
        run_type = "retriever"
)
def retrieve_data(query,qdrant_client, k = 5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }


##format the retrieved documents
@traceable(
        name = "format_retrieved_context",
        run_type = "prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context['retrieved_context_ids'],context['retrieved_context'],context['retrieved_context_ratings']):
        formatted_context += f"ID: {id}, Rating: {rating}, description:{chunk}\n"
    return formatted_context


##create promp template function
@traceable(
        name = "built_prompt",
        run_type = "prompt"
)
def built_prompt(preprocessed_context, question):
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock
    You will be given a question and list of retrieved context.

    Instructions:
    - Answer the question based on the provided context only
    - Do not use markdown formatting
    - Never use word context and refer to it as the available products.

    Context: {preprocessed_context}
    Question: {question}
    
    """
    return prompt


##generate answer function
@traceable(
        name = "generate_answer",
        run_type = "llm",
                metadata = {
            "ls_provider": "openai",
            "ls_model_name": "gpt-5.4-nano"
        }
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "user", "content": prompt}
        ],
        reasoning_effort = "none"
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    return response.choices[0].message.content


##combine RAG pipeline components
@traceable(
        name = "rag_pipeline"
)
def rag_pipeline( question,qdrant_client,top_k = 5):
    
    retrieved_context = retrieve_data(question,qdrant_client, k = top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = built_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    final_answer  = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrieved_context']
    }

    return final_answer