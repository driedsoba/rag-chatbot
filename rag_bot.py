# rag_bot.py
import os
from dotenv import load_dotenv
import boto3
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings

# Load AWS creds from .env (if you used .env)
load_dotenv()

# 1. Download FAQ document from S3
s3 = boto3.client('s3', region_name="ap-southeast-1")
bucket = os.getenv("S3_BUCKET") # rag-faqstore
if bucket is None:
    raise ValueError("S3_BUCKET environment variable not set.")
s3.download_file(bucket, 'faq.txt', 'data/faq.txt')

# 2. Load & split the document
loader = TextLoader("data/faq.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed & index with FAISS
embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
vector_db = FAISS.from_documents(chunks, embed_model)
retriever = vector_db.as_retriever()

# 4. Initialize Bedrock LLM
llm = Bedrock(
    model_id="amazon.titan-text-express-v1",  
    region_name= "us-east-1"
)

# 5. Build RAG chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Test a question
if __name__ == "__main__":
    question = "What is your hobby?"
    answer = qa.run(question)
    print("Q:", question)
    print("A:", answer)
