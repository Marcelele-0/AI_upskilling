"""
Test configuration and utilities for RAG System tests.
"""
import os
import pytest


@pytest.fixture
def mock_env_vars():
    """Fixture providing mock environment variables for testing."""
    return {
        'AZURE_SEARCH_NAME': 'test-search-service',
        'AZURE_SEARCH_KEY': 'test-key-12345',
        'AZURE_SEARCH_INDEX': 'test-index',
        'AZURE_SEARCH_ENDPOINT': 'https://test-search-service.search.windows.net',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_KEY': 'test-openai-key-67890',
        'AZURE_OPENAI_DEPLOYMENT': 'test-gpt-deployment',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': 'test-embedding-deployment'
    }


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.',
            'id': 'doc1',
            '@search.score': 0.95
        },
        {
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.',
            'id': 'doc2',
            '@search.score': 0.87
        },
        {
            'content': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers.',
            'id': 'doc3', 
            '@search.score': 0.82
        }
    ]


@pytest.fixture
def sample_embedding():
    """Fixture providing a sample embedding vector for testing."""
    # Return a mock 768-dimensional embedding vector
    return [0.1 * i for i in range(768)]
