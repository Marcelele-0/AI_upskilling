import logging
import json
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Azure Search parameters
search_service = os.getenv("AZURE_SEARCH_NAME")
api_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX", "witty-receipt-64hk0g5hxc")
endpoint = f"https://{search_service}.search.windows.net"
credential = AzureKeyCredential(api_key)

# Azure OpenAI parameters
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_KEY")
gpt_deployment_name = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-ai-upskilling")  # Add GPT deployment name

# Initialize original clients (keeping for compatibility)
openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=azure_openai_endpoint,
    api_key=openai_api_key
)

search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=credential
)

# Initialize LangChain components
llm = AzureChatOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=openai_api_key,
    api_version="2024-12-01-preview",
    azure_deployment=gpt_deployment_name,
    model=gpt_deployment_name,  # Use the same name for model
    temperature=0.1
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_openai_endpoint,
    api_key=openai_api_key,
    api_version="2024-12-01-preview",
    azure_deployment="text-embedding-ai-upskilling"
)

# Create Prompt Template
rag_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided context.
    
Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
)


def ask_rag(query: str) -> dict:
    """
    Main RAG function that takes a query and returns an answer using LangChain.
    Uses direct vector search and LLM call to avoid AzureSearch retriever issues.
    
    Args:
        query (str): The user's question
        
    Returns:
        dict: Contains the answer, source documents, and metadata
    """
    logger.info(f"Processing RAG query: {query}")
    
    try:
        # Use direct vector search with original search client
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-ai-upskilling",
            input=query
        ).data[0].embedding
        
        # Execute vector search
        search_results = search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "contentVector",
                "k": 3
            }]
        )
        
        # Format context from search results
        context_parts = []
        source_docs = []
        
        for doc in search_results:
            content = doc.get("content", "")
            context_parts.append(content)
            source_docs.append({
                "content": content,
                "metadata": {"id": doc.get("id"), "score": doc.get("@search.score")},
                "score": doc.get("@search.score")
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt with context
        prompt = rag_prompt_template.format(context=context, question=query)
        
        # Get LLM response
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Format response
        result = {
            "query": query,
            "answer": answer,
            "source_documents": source_docs,
            "total_sources": len(source_docs)
        }
        
        logger.info(f"RAG query completed successfully with {result['total_sources']} sources")
        return result
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return {
            "query": query,
            "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
            "source_documents": [],
            "total_sources": 0,
            "error": str(e)
        }


def vector_query(query_text: str) -> list:
    """
    Legacy vector search query function (kept for compatibility).
    
    Args:
        query_text (str): The query text to search for
        
    Returns:
        list: A list of results with id, score, and content
    """
    logger.info(f"Running legacy vector query for: {query_text}")
    
    # Generate embedding for the query
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-ai-upskilling",
        input=query_text
    ).data[0].embedding
    
    # Execute vector search
    vector_results = search_client.search(
        search_text=None,
        vector_queries=[{
            "kind": "vector",
            "vector": query_embedding,
            "fields": "contentVector",
            "k": 3
        }]
    )
    
    # Format results
    vector_output = [
        {
            "id": doc.get("id"),
            "score": doc.get("@search.score"),
            "content": doc.get("content")
        }
        for doc in vector_results
    ]
    
    return vector_output


def semantic_query(query_text: str) -> list:
    """
    Legacy semantic search query function (kept for compatibility).
    
    Args:
        query_text (str): The query text to search for
        
    Returns:
        list: A list of results with id, score, and content
    """
    logger.info(f"Running legacy semantic search for: {query_text}")
    
    # Execute semantic search
    semantic_results = search_client.search(
        search_text=query_text,
        top=3
    )
    
    # Format results
    semantic_output = [
        {
            "id": doc.get("id"),
            "score": doc.get("@search.score"),
            "content": doc.get("content")
        }
        for doc in semantic_results
    ]
    
    return semantic_output


def create_notebook(rag_result: dict, vector_results: list, semantic_results: list, notebook_path: str) -> None:
    """
    Create a Jupyter notebook with the search results including RAG output.
    
    Args:
        rag_result (dict): Results from RAG query
        vector_results (list): Results from vector search
        semantic_results (list): Results from semantic search
        notebook_path (str): Path where to save the notebook
    """
    nb = new_notebook()
    
    # Create cells with results
    nb.cells = [
        new_code_cell(
            source="# RAG Query Result (LangChain)\nrag_result = ...",
            outputs=[
                new_output(
                    "execute_result",
                    data={"application/json": rag_result},
                    execution_count=1
                )
            ]
        ),
        new_code_cell(
            source="# Vector Query Result (Legacy)\nvector_results = ...",
            outputs=[
                new_output(
                    "execute_result",
                    data={"application/json": vector_results},
                    execution_count=2
                )
            ]
        ),
        new_code_cell(
            source="# Semantic Query Result (Legacy)\nsemantic_results = ...",
            outputs=[
                new_output(
                    "execute_result",
                    data={"application/json": semantic_results},
                    execution_count=3
                )
            ]
        )
    ]
    
    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def main() -> None:
    """
    Main function demonstrating both LangChain RAG and legacy search methods.
    """
    # Query configuration
    query_text = "which is the best city to travel to in the world?"
    
    logger.info("Starting LangChain RAG demo...")
    
    # Execute RAG query using LangChain
    rag_result = ask_rag(query_text)
    print(f"\n=== RAG Answer ===")
    print(f"Question: {rag_result['query']}")
    print(f"Answer: {rag_result['answer']}")
    print(f"Sources used: {rag_result['total_sources']}")
    
    # Execute legacy searches for comparison
    logger.info("Running legacy searches for comparison...")
    vector_results = vector_query(query_text)
    semantic_results = semantic_query(query_text)
    
    # Setup notebook path
    notebook_path = os.path.join("notebooks", "langchain_queries.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    
    # Create and save notebook
    create_notebook(rag_result, vector_results, semantic_results, notebook_path)
    
    logger.info(f"Process completed and results saved to {notebook_path}.")


if __name__ == "__main__":
    main()