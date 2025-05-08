import os
import unittest
from unittest.mock import patch, MagicMock

# Import functionality to test
import sys
sys.path.append('.')

class TestAppFunctionality(unittest.TestCase):
    
    @patch('boto3.client')
    @patch.dict(os.environ, {"AWS_REGION": "ap-southeast-1", "AWS_DEFAULT_REGION": "us-east-1"})
    def test_s3_connection(self, mock_boto3_client):
        """Test that S3 connection works with mocked client"""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Avoid loading all dependencies during test collection
        from app import s3
        
        # Verify s3 client was created with expected region - AWS_DEFAULT_REGION
        mock_boto3_client.assert_called_with('s3', region_name="us-east-1")
    
    @patch('langchain_community.vectorstores.faiss.FAISS')
    def test_vector_db_retriever(self, mock_faiss):
        """Test that vector retriever is configured with k=5"""
        mock_vector_db = MagicMock()
        mock_faiss.load_local.return_value = mock_vector_db
        mock_faiss.from_documents.return_value = mock_vector_db
        
        # Avoid loading all dependencies during test collection
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
        
        # Import to trigger the code that creates the models
        with patch('os.path.exists', return_value=True), \
             patch('langchain_community.vectorstores.faiss.FAISS'):
            from app import embed_model, llm
            
        # Verify correct region is used for both models
        mock_embeddings.assert_called_with(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
        mock_llm.assert_called_with(
            model_id="amazon.titan-text-express-v1",
            region_name="us-east-1",
            streaming=True,
            model_kwargs={"temperature": 0.2, "maxTokenCount": 512}
        )

if __name__ == '__main__':
    unittest.main()