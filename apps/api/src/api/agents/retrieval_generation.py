from langsmith import traceable, get_current_run_tree
import openai
from qdrant_client import QdrantClient
import instructor
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document
from qdrant_client import models


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

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
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
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

    client = instructor.from_provider(
        "openai/gpt-5.4-nano",
        mode=instructor.Mode.RESPONSES_TOOLS
    )
    
    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt}
        ],
        reasoning={"effort": "none"},
        response_model=RAGGenerationResponse
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    return response



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
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrieved_context']
    }

    return final_answer



def rag_pipeline_wraper(question, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, top_k)

    used_context = []

    for item in result.get("references", []):
        payload = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01-hybrid-search",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        )[0][0].payload
        image_url = payload.get("image", "")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                }
            )

    return {
        "answer": result["answer"],
        "used_context": used_context
    }