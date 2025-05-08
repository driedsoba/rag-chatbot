import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# import config.py from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAppFunctionality(unittest.TestCase):

    @patch.dict(os.environ, {"AWS_DEFAULT_REGION": "region1"})
    @patch('boto3.client')
    def test_s3_connection(self, mock_boto3_client):
        """Test that S3 client is created with the correct region."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        # Patch all the config.py side-effects
        with patch('os.makedirs'), \
             patch('langchain_community.embeddings.bedrock.BedrockEmbeddings'), \
             patch('langchain_community.llms.bedrock.Bedrock'), \
             patch('langchain_community.vectorstores.faiss.FAISS'), \
             patch('os.path.exists', return_value=True):

            # Import from config, not app
            from config import s3, AWS_REGION

        # Ensure the mock was called with the correct arguments
        mock_boto3_client.assert_called_once_with("s3", region_name="region1")
        self.assertEqual(AWS_REGION, "region1")

    @patch.dict(os.environ, {"AWS_DEFAULT_REGION": "region1"})
    def test_vector_db_retriever(self):
        """Test that vector retriever is configured with k=5."""
        with patch('langchain_community.embeddings.bedrock.BedrockEmbeddings'), \
             patch('langchain_community.llms.bedrock.Bedrock'), \
             patch('langchain_community.vectorstores.faiss.FAISS') as mock_faiss, \
             patch('boto3.client'), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True):

            # Make FAISS.load_local / from_documents return our fake
            mock_vector_db = MagicMock()
            mock_vector_db.as_retriever = MagicMock()  # Ensure as_retriever is mocked
            mock_faiss.load_local.return_value = mock_vector_db
            mock_faiss.from_documents.return_value = mock_vector_db

            # Import retriever from config
            from config import retriever

        # Ensure the mock was called with the correct arguments
        mock_vector_db.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    @patch.dict(os.environ, {
        "AWS_DEFAULT_REGION": "region2",
        "EMBED_MODEL_ID": "embedX",
        "LLM_MODEL_ID": "llmY"
    })
    def test_bedrock_region_configuration(self):
        """Test that Bedrock models use the correct region and model IDs."""
        with patch('langchain_community.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings, \
             patch('langchain_community.llms.bedrock.Bedrock') as mock_llm, \
             patch('langchain_community.vectorstores.faiss.FAISS'), \
             patch('boto3.client'), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True):

            # Return dummy embeddings & llm instances
            mock_embeddings.return_value = MagicMock()
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke = AsyncMock()  # Mimic Runnable behavior
            mock_llm.return_value = mock_llm_instance

            from config import LLM_MODEL, EMBED_MODEL, AWS_REGION

        # Should pick up the AWS_DEFAULT_REGION we set
        self.assertEqual(AWS_REGION, "region2")

        # Verify that the Bedrock embeddings and LLM were called with correct parameters
        mock_embeddings.assert_called_once_with(
            model_id=EMBED_MODEL,
            region_name="region2"
        )
        mock_llm.assert_called_once_with(
            model_id=LLM_MODEL,
            region_name="region2",
            streaming=True,
            model_kwargs={"temperature": 0.2, "maxTokenCount": 512}
        )


if __name__ == '__main__':
    unittest.main()