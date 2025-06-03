"""
Application Insights Telemetry Configuration
Centralized logging and telemetry setup for structured logging with trace IDs.
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager
from threading import local

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import config_integration
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
from opencensus.trace.span import SpanKind
from opencensus.trace import execution_context

# Thread-local storage for trace context
_local = local()

class StructuredLogger:
    """
    Structured logger with Application Insights integration and trace ID support.
    """
    
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Add console handler for local development
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        self._setup_azure_logging()
        
    def _setup_azure_logging(self):
        """Configure Azure Application Insights logging."""
        # Get connection string from environment
        connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        if not connection_string:
            # Fallback to reading from .env if not in local.settings.json
            from dotenv import load_dotenv
            load_dotenv()
            connection_string = os.getenv('AZURE_INSIGHTS_INSTRUMENTATION_KEY')
            
        if connection_string:
            # Configure Azure Log Handler
            azure_handler = AzureLogHandler(connection_string=connection_string)
            azure_handler.setLevel(logging.INFO)
            
            # Custom formatter for structured logging
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - TraceId:%(trace_id)s - %(message)s'
            )
            azure_handler.setFormatter(formatter)
              # Add handler to logger
            if not any(isinstance(h, AzureLogHandler) for h in self.logger.handlers):
                self.logger.addHandler(azure_handler)
                self.logger.setLevel(logging.DEBUG)  # Set to DEBUG level
        else:
            self.logger.warning("No Application Insights connection string found")
    
    def get_trace_id(self) -> str:
        """Get current trace ID or generate new one."""
        trace_id = getattr(_local, 'trace_id', None)
        if not trace_id:
            trace_id = str(uuid.uuid4())
            _local.trace_id = trace_id
        return trace_id
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for current thread."""
        _local.trace_id = trace_id
    
    def clear_trace_id(self):
        """Clear trace ID for current thread."""
        if hasattr(_local, 'trace_id'):
            delattr(_local, 'trace_id')
    
    def _log_with_trace(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log message with trace ID and extra data."""
        trace_id = self.get_trace_id()
        
        # Prepare extra data for structured logging
        extra = {
            'trace_id': trace_id,
            'custom_dimensions': extra_data or {}
        }
        
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message with trace ID."""
        self._log_with_trace(logging.INFO, message, extra_data)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message with trace ID."""
        self._log_with_trace(logging.DEBUG, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning message with trace ID."""
        self._log_with_trace(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message with trace ID."""
        if exc_info:
            self.logger.error(message, exc_info=True, extra={
                'trace_id': self.get_trace_id(),
                'custom_dimensions': extra_data or {}
            })
        else:
            self._log_with_trace(logging.ERROR, message, extra_data)
    
    def exception(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log exception with trace ID and stack trace."""
        self.error(message, extra_data, exc_info=True)

class TelemetryTracer:
    """
    Application Insights tracer for distributed tracing.
    """
    
    def __init__(self):
        self.tracer = None
        self._setup_tracer()
    
    def _setup_tracer(self):
        """Setup OpenCensus tracer with Azure exporter."""
        connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        if not connection_string:
            from dotenv import load_dotenv
            load_dotenv()
            connection_string = os.getenv('AZURE_INSIGHTS_INSTRUMENTATION_KEY')
            
        if connection_string:
            # Configure integrations
            config_integration.trace_integrations(['requests', 'logging'])
              # Setup tracer with Azure exporter
            self.tracer = Tracer(
                exporter=AzureExporter(connection_string=connection_string),
                sampler=ProbabilitySampler(1.0)  # Sample all requests for development
            )
    
    @contextmanager
    def start_span(self, name: str, span_kind: SpanKind = SpanKind.UNSPECIFIED):
        """Start a new span for tracing."""
        if self.tracer:
            with self.tracer.span(name=name) as span:
                # Set span kind if span object supports it
                if hasattr(span, 'span_kind'):
                    span.span_kind = span_kind
                yield span
        else:
            yield None

# Global instances
logger = StructuredLogger(__name__)
tracer = TelemetryTracer()

def get_logger(name: str = __name__) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)

def get_tracer() -> TelemetryTracer:
    """Get the global tracer instance."""
    return tracer

def generate_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())

@contextmanager
def trace_context(trace_id: Optional[str] = None):
    """Context manager for trace ID."""
    if trace_id is None:
        trace_id = generate_trace_id()
    
    logger.set_trace_id(trace_id)
    try:
        yield trace_id
    finally:
        logger.clear_trace_id()

def log_custom_event(event_name: str, properties: Dict[str, Any] = None, measurements: Dict[str, float] = None):
    """Log custom event to Application Insights."""
    logger.info(f"Custom Event: {event_name}", {
        'event_name': event_name,
        'properties': properties or {},
        'measurements': measurements or {}
    })

def log_dependency(dependency_name: str, command: str, duration: float, success: bool = True, result_code: str = "200"):
    """Log dependency call to Application Insights."""
    logger.info(f"Dependency: {dependency_name}", {
        'dependency_name': dependency_name,
        'command': command,
        'duration_ms': duration * 1000,
        'success': success,
        'result_code': result_code
    })
