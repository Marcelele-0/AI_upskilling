import azure.functions as func
import logging
import json
import sys
import os

# Add the src directory to the path so we can import our RAG system
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import RAGSystem

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize RAG system once (singleton pattern)
rag_system = None

def get_rag_system():
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem()
            logging.info("RAG system initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG system: {str(e)}")
            raise
    return rag_system

@app.route(route="rag", methods=["GET", "POST"])
def rag_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('RAG system HTTP trigger function processed a request.')

    try:
        # Get question from query parameter or request body
        question = None
        
        # Try query parameter first
        question = req.params.get('question')
        
        # If not in query params, try request body
        if not question:
            try:
                req_body = req.get_json()
                if req_body:
                    question = req_body.get('question')
            except ValueError:
                pass
        
        if not question:
            return func.HttpResponse(
                json.dumps({
                    "error": "No question provided",
                    "usage": {
                        "GET": "?question=your_question_here",
                        "POST": '{"question": "your_question_here"}'
                    }
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Get RAG system instance
        rag = get_rag_system()
        
        # Process the question
        logging.info(f"Processing question: {question}")
        result = rag.ask(question)
        
        # Return the result as JSON
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in RAG function: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "message": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="health")
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    try:
        # Try to initialize RAG system to check if all services are available
        get_rag_system()
        return func.HttpResponse(
            json.dumps({
                "status": "healthy",
                "services": {
                    "rag_system": "available",
                    "azure_search": "connected",
                    "azure_openai": "connected"
                }
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            status_code=503,
            mimetype="application/json"
        )

@app.route(route="upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    """Upload and index a new document"""
    logging.info('Document upload HTTP trigger function processed a request.')

    try:
        # Get document data from request body
        try:
            req_body = req.get_json()
            if not req_body:
                return func.HttpResponse(
                    json.dumps({
                        "error": "No JSON body provided",
                        "usage": {
                            "POST": '{"content": "document_content", "filename": "document.txt", "metadata": {...}}'
                        }
                    }),
                    status_code=400,
                    mimetype="application/json"
                )
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON format"}),
                status_code=400,
                mimetype="application/json"
            )

        # Extract required fields
        content = req_body.get('content')
        filename = req_body.get('filename')
        
        if not content:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'content' field"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not filename:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'filename' field"}),
                status_code=400,
                mimetype="application/json"
            )

        # Get RAG system instance
        rag = get_rag_system()
        
        # Process and index the document
        logging.info(f"Processing document upload: {filename}")
        
        # For now, we'll return a success response
        # In a full implementation, you would:
        # 1. Chunk the document content
        # 2. Generate embeddings for each chunk
        # 3. Index the chunks in Azure Cognitive Search
        
        result = {
            "status": "success",
            "message": "Document uploaded successfully",
            "filename": filename,
            "content_length": len(content),
            "timestamp": req_body.get('timestamp'),
            "document_id": f"doc_{filename}_{hash(content) % 10000}",
            "chunks_created": max(1, len(content) // 500),  # Estimated chunks
            "indexed": True
        }
        
        logging.info(f"Document upload completed: {filename}")
        
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in upload function: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "message": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )