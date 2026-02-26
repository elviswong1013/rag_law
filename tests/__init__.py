import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def mock_api_key():
    return "test_api_key_12345"


@pytest.fixture
def sample_text_content():
    return """
    This is a sample legal document.
    
    Section 1: Legal Procedures
    Legal procedures must follow established protocols and regulations.
    All documents must be properly filed and maintained.
    
    Section 2: Court System
    The court system operates under specific guidelines.
    Judges must follow legal precedents and statutes.
    
    Section 3: Contracts
    Contracts must be signed by all parties involved.
    Terms and conditions must be clearly stated.
    """
