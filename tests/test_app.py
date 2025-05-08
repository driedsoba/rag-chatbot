import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock chainlit and its dependencies before importing app
sys.modules['chainlit'] = MagicMock()
sys.modules['chainlit.message'] = MagicMock()

class TestAppFunctionality(unittest.TestCase):
    
    @patch('boto3.client')
    @patch.dict(os.environ, {"AWS_REGION": "ap-southeast-1", "AWS_DEFAULT_REGION": "us-east-1"})
    def test_s3_connection(self, mock_boto3_client):
        """Test that S3 connection works with mocked client"""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Import locally
        with patch('os.makedirs'):  # Prevent directory creation
            from app import s3, AWS_REGION
        
        # Verify s3 client was created with expected region
        mock_boto3_client.assert_called_with('s3', region_name=AWS_REGION)
    
    @patch('langchain_community.vectorstores.faiss.FAISS')
    def test_vector_db_retriever(self, mock_faiss):
        """Test that vector retriever is configured with k=5"""
        # Import critical pieces first
        with patch('os.makedirs'), patch('boto3.client'), patch('app.s3.download_file'):
            # Setup mocks
            mock_vector_db = MagicMock()
            mock_faiss.load_local.return_value = mock_vector_db
            mock_faiss.from_documents.return_value = mock_vector_db
            
            # Mock os.path.exists to return True 
            with patch('os.path.exists', return_value=True):
                from app import retriever
            
            # Verify retriever was called with k=5
            mock_vector_db.as_retriever.assert_called_with(search_kwargs={"k": 5})
    
    @patch('langchain_community.llms.bedrock.Bedrock')
    @patch('langchain_community.embeddings.bedrock.BedrockEmbeddings')
    @patch.dict(os.environ, {"AWS_REGION": "ap-southeast-1", "AWS_DEFAULT_REGION": "us-east-1"})
    def test_bedrock_region_configuration(self, mock_embeddings, mock_llm):
        """Test that Bedrock models use us-east-1 region"""
        # Setup mocks
        mock_embed_instance = MagicMock()
        mock_embeddings.return_value = mock_embed_instance
        
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        # Import with mocks to avoid real S3 calls
        with patch('os.makedirs'), patch('boto3.client'), patch('app.s3.download_file'), \
             patch('os.path.exists', return_value=True), \
             patch('langchain_community.vectorstores.faiss.FAISS'):
            
            # Import directly to get constants
            from app import LLM_MODEL, EMBED_MODEL, AWS_REGION
        
        # Verify correct model IDs and region are used
        mock_embeddings.assert_called_with(model_id=EMBED_MODEL, region_name=AWS_REGION)
        mock_llm.assert_called_with(
            model_id=LLM_MODEL,
            region_name=AWS_REGION,
            streaming=True,
            model_kwargs={"temperature": 0.2, "maxTokenCount": 512}
        )

if __name__ == '__main__':
    unittest.main()