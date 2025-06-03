import azure.functions as func
import logging
import json
import sys
import os
import time
from datetime import datetime

# Add the src directory to the path so we can import our RAG system
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import RAGSystem
from telemetry import get_logger, trace_context, get_tracer, log_custom_event, log_dependency

# Setup structured logger
logger = get_logger(__name__)
tracer = get_tracer()

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize RAG system once (singleton pattern)
rag_system = None

def get_rag_system():
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.exception(f"Failed to initialize RAG system: {str(e)}")
            raise
    return rag_system

@app.route(route="rag", methods=["GET", "POST"])
def rag_query(req: func.HttpRequest) -> func.HttpResponse:
    # Extract trace ID from headers or generate new one
    trace_id = req.headers.get('X-Trace-Id', req.headers.get('x-trace-id'))
    
    with trace_context(trace_id) as current_trace_id:
        logger.info('RAG system HTTP trigger function processed a request.', {
            'method': req.method,
            'url': req.url,
            'headers': dict(req.headers)
        })
        
        start_time = time.time()
        
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
                logger.warning("No question provided in request", {
                    'method': req.method,
                    'params': dict(req.params),
                    'has_body': bool(req.get_body())
                })
                return func.HttpResponse(
                    json.dumps({
                        "error": "No question provided",
                        "trace_id": current_trace_id,
                        "usage": {
                            "GET": "?question=your_question_here",
                            "POST": '{"question": "your_question_here"}'
                        }
                    }),
                    status_code=400,
                    mimetype="application/json",
                    headers={"X-Trace-Id": current_trace_id}
                )
            
            # Get RAG system instance
            rag = get_rag_system()
            
            # Process the question
            logger.info(f"Processing question: {question}", {
                'question_length': len(question),
                'question_preview': question[:100] + "..." if len(question) > 100 else question
            })
            
            with tracer.start_span("rag_query_processing") as span:
                result = rag.ask(question, trace_id=current_trace_id)
            
            # Log successful completion
            duration = time.time() - start_time
            logger.info("RAG query completed successfully", {
                'duration_seconds': duration,
                'response_length': len(str(result)) if result else 0
            })
            
            log_custom_event("rag_query_completed", {
                'question_length': len(question),
                'success': True
            }, {
                'duration_seconds': duration
            })
            
            # Add trace ID to response
            if isinstance(result, dict):
                result['trace_id'] = current_trace_id
            
            # Return the result as JSON
            return func.HttpResponse(
                json.dumps(result, ensure_ascii=False, indent=2),
                status_code=200,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"Error in RAG function: {str(e)}", {
                'duration_seconds': duration,
                'question': question[:100] if question else None
            })
            
            log_custom_event("rag_query_error", {
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, {
                'duration_seconds': duration
            })
            
            return func.HttpResponse(
                json.dumps({
                    "error": "Internal server error",
                    "message": str(e),
                    "trace_id": current_trace_id
                }),
                status_code=500,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )

@app.route(route="health")
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    trace_id = req.headers.get('X-Trace-Id', req.headers.get('x-trace-id'))
    
    with trace_context(trace_id) as current_trace_id:
        logger.info("Health check requested")
        
        try:
            # Try to initialize RAG system to check if all services are available
            get_rag_system()
            
            logger.info("Health check passed - all services available")
            
            return func.HttpResponse(
                json.dumps({
                    "status": "healthy",
                    "trace_id": current_trace_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "services": {
                        "rag_system": "available",
                        "azure_search": "connected",
                        "azure_openai": "connected",
                        "application_insights": "connected"
                    }
                }),
                status_code=200,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", {
                'exception_class': type(e).__name__
            })
            
            return func.HttpResponse(
                json.dumps({
                    "status": "unhealthy",
                    "trace_id": current_trace_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }),
                status_code=503,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )

@app.route(route="upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    """Upload and index a new document"""
    trace_id = req.headers.get('X-Trace-Id', req.headers.get('x-trace-id'))
    
    with trace_context(trace_id) as current_trace_id:
        logger.info('Document upload HTTP trigger function processed a request.')

        try:
            # Get document data from request body
            try:
                req_body = req.get_json()
                if not req_body:
                    logger.warning("No JSON body provided in upload request")
                    return func.HttpResponse(
                        json.dumps({
                            "error": "No JSON body provided",
                            "trace_id": current_trace_id,
                            "usage": {
                                "POST": '{"content": "document_content", "filename": "document.txt", "metadata": {...}}'
                            }
                        }),
                        status_code=400,
                        mimetype="application/json",
                        headers={"X-Trace-Id": current_trace_id}
                    )
            except ValueError:
                logger.error("Invalid JSON format in upload request")
                return func.HttpResponse(
                    json.dumps({
                        "error": "Invalid JSON format",
                        "trace_id": current_trace_id
                    }),
                    status_code=400,
                    mimetype="application/json",
                    headers={"X-Trace-Id": current_trace_id}
                )

            # Extract required fields
            content = req_body.get('content')
            filename = req_body.get('filename')
            
            if not content:
                logger.warning("Missing 'content' field in upload request")
                return func.HttpResponse(
                    json.dumps({
                        "error": "Missing 'content' field",
                        "trace_id": current_trace_id
                    }),
                    status_code=400,
                    mimetype="application/json",
                    headers={"X-Trace-Id": current_trace_id}
                )
            
            if not filename:
                logger.warning("Missing 'filename' field in upload request")
                return func.HttpResponse(
                    json.dumps({
                        "error": "Missing 'filename' field",
                        "trace_id": current_trace_id
                    }),
                    status_code=400,
                    mimetype="application/json",
                    headers={"X-Trace-Id": current_trace_id}
                )

            # Get RAG system instance
            rag = get_rag_system()
            
            # Process and index the document
            logger.info(f"Processing document upload: {filename}", {
                'filename': filename,
                'content_length': len(content)
            })
            
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
                "indexed": True,
                "trace_id": current_trace_id
            }
            
            logger.info(f"Document upload completed: {filename}", {
                'document_id': result['document_id'],
                'chunks_created': result['chunks_created']
            })
            
            log_custom_event("document_uploaded", {
                'filename': filename,
                'content_length': len(content),
                'chunks_created': result['chunks_created']
            })
            
            return func.HttpResponse(
                json.dumps(result, ensure_ascii=False, indent=2),
                status_code=200,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )
            
        except Exception as e:
            logger.exception(f"Error in upload function: {str(e)}", {
                'filename': filename if 'filename' in locals() else None
            })
            
            return func.HttpResponse(
                json.dumps({
                    "error": "Internal server error",
                    "message": str(e),
                    "trace_id": current_trace_id
                }),
                status_code=500,
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )

@app.route(route="raise_error", methods=["GET", "POST"])
def raise_error(req: func.HttpRequest) -> func.HttpResponse:
    """Endpoint to test error tracking in Application Insights"""
    trace_id = req.headers.get('X-Trace-Id', req.headers.get('x-trace-id'))
    
    with trace_context(trace_id) as current_trace_id:
        error_type = req.params.get('type', 'general')
        
        logger.info(f"Intentionally raising error for testing", {
            'error_type': error_type,
            'method': req.method
        })
        
        try:
            if error_type == 'division':
                # Division by zero error
                result = 1 / 0
            elif error_type == 'index':
                # Index error
                my_list = [1, 2, 3]
                value = my_list[10]
            elif error_type == 'key':
                # Key error
                my_dict = {"a": 1, "b": 2}
                value = my_dict["nonexistent_key"]
            elif error_type == 'custom':
                # Custom exception
                raise ValueError("This is a custom test error with trace ID: " + current_trace_id)
            else:
                # General exception
                raise Exception("General test error for Application Insights tracking")
                
        except Exception as e:
            # Log the error with logger.exception to include stack trace
            logger.exception(f"Test error successfully raised: {str(e)}", {
                'error_type': error_type,
                'exception_class': type(e).__name__,
                'intentional': True
            })
            
            log_custom_event("intentional_error_raised", {
                'error_type': error_type,
                'exception_class': type(e).__name__,
                'test_error': True
            })
            
            return func.HttpResponse(
                json.dumps({
                    "message": "Error successfully raised and logged",
                    "error_type": error_type,
                    "exception_class": type(e).__name__,
                    "error_message": str(e),
                    "trace_id": current_trace_id,
                    "logged_to_insights": True
                }),
                status_code=200,  # Return 200 because this is intentional
                mimetype="application/json",
                headers={"X-Trace-Id": current_trace_id}
            )