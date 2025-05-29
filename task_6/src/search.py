import logging
import json
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from dotenv import load_dotenv


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

# Initialize clients
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


def vector_query(query_text: str) -> list:
    """
    Run a vector search query using the provided text.
    
    Args:
        query_text (str): The query text to search for
        
    Returns:
        list: A list of results with id, score, and content
    """
    logger.info(f"Running vector query for: {query_text}")
    
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
    Run a semantic search query using the provided text.
    
    Args:
        query_text (str): The query text to search for
        
    Returns:
        list: A list of results with id, score, and content
    """
    logger.info(f"Running semantic search for: {query_text}")
    
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


def create_notebook(vector_results: list, semantic_results: list, notebook_path: str) -> None:
    """
    Create a Jupyter notebook with the search results.
    
    Args:
        vector_results (list): Results from vector search
        semantic_results (list): Results from semantic search
        notebook_path (str): Path where to save the notebook
    """
    nb = new_notebook()
    
    # Create cells with results
    nb.cells = [
        new_code_cell(
            source="# Vector Query Result\nvector_results = ...",
            outputs=[
                new_output(
                    "execute_result",
                    data={"application/json": vector_results},
                    execution_count=1
                )
            ]
        ),
        new_code_cell(
            source="# Semantic Query Result\nsemantic_results = ...",
            outputs=[
                new_output(
                    "execute_result",
                    data={"application/json": semantic_results},
                    execution_count=2
                )
            ]
        )
    ]
    
    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def main() -> None:
    """
    Main function to run vector and semantic search queries,
    then save both results to a Jupyter notebook.
    """
    # Query configuration
    query_text = "which is the best city to travel to in the world?"
    
    # Execute searches
    logger.info("Starting search queries...")
    vector_results = vector_query(query_text)
    semantic_results = semantic_query(query_text)
    
    # Setup notebook path
    notebook_path = os.path.join("notebooks", "queries.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    
    # Create and save notebook
    create_notebook(vector_results, semantic_results, notebook_path)
    
    logger.info(f"Process completed and results saved to {notebook_path}.")


if __name__ == "__main__":
    main()