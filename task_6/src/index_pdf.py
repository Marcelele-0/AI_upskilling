import os
import uuid
import PyPDF2
import logging
from typing import List, Dict
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load environment variables
load_dotenv()
AZURE_SEARCH_NAME = os.getenv("AZURE_SEARCH_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "brochures-index")

endpoint = f"https://{AZURE_SEARCH_NAME}.search.windows.net"
index_name = AZURE_SEARCH_INDEX
credential = AzureKeyCredential(AZURE_SEARCH_KEY)

if not endpoint or not AZURE_SEARCH_KEY:
    raise ValueError("AZURE_SEARCH_NAME and AZURE_SEARCH_KEY must be set in the .env file.")


def extract_chunks_from_pdf(pdf_path: str, chunk_size: int = 500) -> List[str]:
    """
    Extract text from a PDF file and split it into chunks of a given size.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum length of each text chunk.

    Returns:
        List[str]: List of text chunks extracted from the PDF.
    """
    logger.info(f"Extracting chunks from {pdf_path}")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.debug(f"Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks


def create_search_index() -> None:
    """
    Create the Azure Search index if it does not exist.
    """
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    try:
        existing_indexes = [idx.name for idx in index_client.list_indexes()]
        if index_name not in existing_indexes:
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", analyzer_name="en.lucene"),
                SimpleField(name="sourcefile", type="Edm.String"),
            ]
            index = SearchIndex(name=index_name, fields=fields)
            index_client.create_index(index)
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists.")
    except Exception as e:
        logger.error(f"Error creating or checking index: {e}")


def upload_documents_to_search(docs: List[Dict]) -> None:
    """
    Upload documents to Azure AI Search in batches.

    Args:
        docs (List[Dict]): List of documents to upload.
    """
    logger.info("Uploading documents to Azure AI Search.")
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        search_client.upload_documents(documents=batch)
        logger.info(f"Uploaded {len(batch)} documents.")


def main() -> None:
    """
    Main function to index PDFs into Azure AI Search.
    """
    logger.info("Starting process of indexing PDFs into Azure AI Search.")
    create_search_index()

    pdf_dir = "brochures"  # Path to the folder with PDFs
    docs: List[Dict] = []

    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            logger.info(f"Processing file: {file}")
            chunks = extract_chunks_from_pdf(pdf_path)
            for chunk in chunks:
                docs.append({
                    "id": str(uuid.uuid4()),
                    "content": chunk,
                    "sourcefile": file
                })

    if docs:
        upload_documents_to_search(docs)
    else:
        logger.warning("No documents to upload.")

    logger.info("Process completed.")


if __name__ == "__main__":
    main()