import os
import json
import uuid
from dotenv import load_dotenv
import PyPDF2
import nbformat as nbf
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration, SearchableField
)
from openai import AzureOpenAI

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    config = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "AZURE_SEARCH_INDEX": os.getenv("AZURE_SEARCH_INDEX"),
        "PDF_PATH": os.getenv("PDF_PATH", "test.pdf"),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return config

def extract_chunks_from_pdf(pdf_path, chunk_size=500):
    """Extract text from PDF and split into chunks."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_openai_client(endpoint, deployment):
    """Create an AzureOpenAI client with Azure AD credentials."""
    credential = DefaultAzureCredential()
    return AzureOpenAI(
        api_version="2024-06-01",
        azure_endpoint=endpoint,
        azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token
    )

def embed_texts(openai_client, texts, deployment):
    """Generate embeddings for a list of texts."""
    return openai_client.embeddings.create(
        input=texts,
        model=deployment
    ).data

def create_search_index(search_index_client, index_name):
    """Create or recreate the Azure Search index."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536
        ),
    ]
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            algorithm_configurations=[HnswAlgorithmConfiguration(name="default")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")]
        )
    )
    try:
        search_index_client.delete_index(index_name)
    except Exception:
        pass
    search_index_client.create_index(index)

def upload_documents(search_client, docs):
    """Upload documents to Azure Search index."""
    search_client.upload_documents(documents=docs)

def run_vector_query(search_client, embedding):
    """Run a vector search query."""
    return search_client.search(
        search_text=None,
        vectors=[{
            "value": embedding,
            "fields": "embedding",
            "k": 5,
            "kind": "vector"
        }],
        select=["content"]
    )

def run_semantic_query(search_client, query):
    """Run a semantic search query."""
    return search_client.search(
        search_text=query,
        query_type="semantic",
        query_language="en-us",
        semantic_configuration_name="default",
        select=["content"],
        top=5
    )

def save_results_to_notebook(vector_result, semantic_result, output_path):
    """Save search results to a Jupyter notebook."""
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Vector Search Results"))
    nb.cells.append(nbf.v4.new_code_cell(
        "vector_results = " + json.dumps([r["content"] for r in vector_result])
    ))
    nb.cells.append(nbf.v4.new_markdown_cell("# Semantic Search Results"))
    nb.cells.append(nbf.v4.new_code_cell(
        "semantic_results = " + json.dumps([r["content"] for r in semantic_result])
    ))
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

def main():
    config = load_config()
    chunks = extract_chunks_from_pdf(config["PDF_PATH"])
    openai_client = get_openai_client(
        config["AZURE_OPENAI_ENDPOINT"],
        config["AZURE_OPENAI_DEPLOYMENT"]
    )
    credential = DefaultAzureCredential()
    search_index_client = SearchIndexClient(
        endpoint=config["AZURE_SEARCH_ENDPOINT"],
        credential=credential
    )
    create_search_index(search_index_client, config["AZURE_SEARCH_INDEX"])
    search_client = SearchClient(
        endpoint=config["AZURE_SEARCH_ENDPOINT"],
        index_name=config["AZURE_SEARCH_INDEX"],
        credential=credential
    )
    docs = []
    for chunk in chunks:
        embedding = embed_texts(openai_client, [chunk], config["AZURE_OPENAI_DEPLOYMENT"])[0].embedding
        docs.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": embedding
        })
    upload_documents(search_client, docs)
    query = "What are the main topics of the document?"
    vector_query = embed_texts(openai_client, [query], config["AZURE_OPENAI_DEPLOYMENT"])[0].embedding
    vector_result = run_vector_query(search_client, vector_query)
    semantic_result = run_semantic_query(search_client, query)
    save_results_to_notebook(vector_result, semantic_result, "queries.ipynb")
    print("✅ Done! Results saved in: queries.ipynb")

if __name__ == "__main__":
    main()
