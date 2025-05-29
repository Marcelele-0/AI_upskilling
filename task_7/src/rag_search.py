import logging
import os
from typing import Dict, List
from dotenv import load_dotenv

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate


# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Clean RAG (Retrieval-Augmented Generation) system using Azure Cognitive Search and OpenAI.
    """
    
    def __init__(self):
        """Initialize the RAG system with Azure services."""
        load_dotenv()
        self._setup_azure_search()
        self._setup_openai_clients()
        self._setup_prompt_template()
        
    def _setup_azure_search(self):
        """Setup Azure Cognitive Search client."""
        logger.debug("Setting up Azure Cognitive Search client")
        
        search_service = os.getenv("AZURE_SEARCH_NAME")
        api_key = os.getenv("AZURE_SEARCH_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX", "witty-receipt-64hk0g5hxc")
        
        logger.debug(f"Azure Search service: {search_service}")
        logger.debug(f"Azure Search index: {self.index_name}")
        
        endpoint = f"https://{search_service}.search.windows.net"
        credential = AzureKeyCredential(api_key)
        
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=self.index_name,
            credential=credential
        )
        logger.debug(f"Azure Search client initialized for service: {search_service}")
        
    def _setup_openai_clients(self):
        """Setup Azure OpenAI clients for embeddings and chat."""
        logger.debug("Setting up Azure OpenAI clients")
        
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_api_key = os.getenv("AZURE_OPENAI_KEY")
        gpt_deployment_name = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-ai-upskilling")
        
        logger.debug(f"Azure OpenAI endpoint: {azure_openai_endpoint}")
        logger.debug(f"GPT deployment name: {gpt_deployment_name}")
        
        # Client for embeddings
        self.openai_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=azure_openai_endpoint,
            api_key=openai_api_key
        )
        
        # LangChain LLM for chat
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=openai_api_key,
            api_version="2024-12-01-preview",
            azure_deployment=gpt_deployment_name,
            model=gpt_deployment_name,
            temperature=0.1
        )
        logger.debug("Azure OpenAI clients initialized successfully")
        
    def _setup_prompt_template(self):
        """Setup the RAG prompt template."""
        logger.debug("Setting up RAG prompt template")
        
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Jesteś pomocnym asystentem, który odpowiada na pytania na podstawie dostarczonego kontekstu.

Kontekst:
{context}

Pytanie: {question}

Instrukcje:
- Udziel wyczerpującej odpowiedzi na podstawie powyższego kontekstu
- Jeśli kontekst nie zawiera wystarczających informacji, powiedz o tym jasno
- Odpowiadaj w języku pytania
- Bądź precyzyjny i pomocny

Odpowiedź:"""
        )
        logger.debug("RAG prompt template configured successfully")
        
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents from Azure Search using vector similarity.
        
        Args:
            query (str): User's question
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: Retrieved documents with content and metadata
        """
        logger.debug(f"Starting document retrieval for query: '{query}' (top_k={top_k})")
        
        try:
            # Generate query embedding
            logger.debug("Generating query embedding")
            query_embedding = self.openai_client.embeddings.create(
                model="text-embedding-ai-upskilling",
                input=query
            ).data[0].embedding
            logger.debug(f"Generated embedding vector of size: {len(query_embedding)}")
            
            # Execute vector search
            logger.debug("Executing vector search in Azure Cognitive Search")
            search_results = self.search_client.search(
                search_text=None,
                vector_queries=[{
                    "kind": "vector",
                    "vector": query_embedding,
                    "fields": "contentVector",
                    "k": top_k
                }]
            )
            
            # Format results
            documents = []
            for i, doc in enumerate(search_results):
                doc_info = {
                    "content": doc.get("content", ""),
                    "id": doc.get("id"),
                    "score": doc.get("@search.score")
                }
                documents.append(doc_info)
                logger.debug(f"Document {i+1}: ID={doc_info['id']}, Score={doc_info['score']:.3f}, Content length={len(doc_info['content'])}")
                
            logger.debug(f"Successfully retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            return []
    
    def generate_answer(self, query: str, documents: List[Dict]) -> str:
        """
        Generate answer using LLM based on retrieved documents.
        
        Args:
            query (str): User's question
            documents (List[Dict]): Retrieved documents
            
        Returns:
            str: Generated answer
        """
        logger.debug(f"Starting answer generation for query: '{query}'")
        logger.debug(f"Using {len(documents)} documents as context")
        
        try:
            # Create context from documents
            context_parts = [doc["content"] for doc in documents if doc["content"]]
            context = "\n\n".join(context_parts)
            
            logger.debug(f"Context created from {len(context_parts)} non-empty documents")
            logger.debug(f"Total context length: {len(context)} characters")
            
            if not context:
                logger.debug("No valid context found in retrieved documents")
                return "Przepraszam, nie znalazłem odpowiednich dokumentów, aby odpowiedzieć na Twoje pytanie."
            
            # Generate prompt and get LLM response
            logger.debug("Generating LLM prompt")
            prompt = self.rag_prompt.format(context=context, question=query)
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            logger.debug("Calling Azure OpenAI LLM for answer generation")
            response = self.llm.invoke(prompt)
            
            answer = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Answer generated successfully (length: {len(answer)} characters)")
            logger.debug(f"Generated answer preview: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return f"Przepraszam, wystąpił błąd podczas generowania odpowiedzi: {str(e)}"
    
    def ask(self, query: str) -> Dict:
        """
        Main RAG function - retrieve documents and generate answer.
        
        Args:
            query (str): User's question
            
        Returns:
            Dict: Complete RAG response with answer and sources
        """
        logger.debug(f"Processing RAG query: {query}")
        
        # Retrieve relevant documents
        documents = self.retrieve_documents(query)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Format response
        result = {
            "question": query,
            "answer": answer,
            "sources": documents,
            "source_count": len(documents)
        }
        
        logger.debug(f"RAG processing completed for query: {query}")
        return result
    
    def print_response(self, result: Dict):
        """Pretty print the RAG response."""
        print(f"\n{'='*60}")
        print(f"PYTANIE: {result['question']}")
        print(f"{'='*60}")
        print(f"ODPOWIEDŹ:\n{result['answer']}")
        print(f"\n{'='*60}")
        print(f"ŹRÓDŁA ({result['source_count']}):")
        for i, doc in enumerate(result['sources'], 1):
            print(f"\n{i}. ID: {doc['id']} | Score: {doc['score']:.3f}")
            preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            print(f"   Treść: {preview}")
        print(f"{'='*60}\n")


def main():
    """
    Main function with interactive question loop.
    """
    logger.debug("Starting RAG system")
    logger.info("🔍 Witamy w systemie RAG!")
    logger.info("Zadawaj pytania, a system znajdzie odpowiedzi w bazie dokumentów.")
    logger.info("Wpisz 'quit', 'exit' lub 'koniec', aby zakończyć.\n")
    
    try:
        # Initialize RAG system
        logger.debug("Initializing RAG system components")
        rag_system = RAGSystem()
        logger.debug("RAG system initialized successfully")
        logger.info("✅ System RAG został zainicjalizowany pomyślnie!\n")
        
        # Question loop
        question_count = 0
        while True:
            try:
                # Get user question
                question = input("❓ Twoje pytanie: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'koniec', 'q']:
                    logger.debug(f"User requested exit after {question_count} questions")
                    logger.info("👋 Dziękujemy za skorzystanie z systemu RAG!")
                    break
                    
                # Skip empty questions
                if not question:
                    logger.debug("Empty question received, prompting user again")
                    logger.info("⚠️  Proszę wpisać pytanie.\n")
                    continue
                
                # Process question
                question_count += 1
                logger.debug(f"Processing question #{question_count}: {question}")
                logger.debug("Starting RAG processing pipeline")
                logger.info("🔄 Przetwarzam pytanie...")
                
                result = rag_system.ask(question)
                
                # Log results
                logger.debug(f"Question #{question_count} processed successfully")
                logger.debug(f"Generated answer length: {len(result['answer'])} characters")
                logger.debug(f"Retrieved {result['source_count']} source documents")
                
                # Display response
                rag_system.print_response(result)
                
            except KeyboardInterrupt:
                logger.debug("Program interrupted by user (Ctrl+C)")
                logger.info("\n\n👋 Program przerwany przez użytkownika.")
                break
            except Exception as e:
                logger.error(f"Error in question loop: {str(e)}", exc_info=True)
                logger.info(f"❌ Wystąpił błąd: {str(e)}\n")
                
    except Exception as e:
        logger.error(f"Critical error during system initialization: {str(e)}", exc_info=True)
        logger.info(f"❌ Błąd inicjalizacji systemu: {str(e)}")


if __name__ == "__main__":
    main()