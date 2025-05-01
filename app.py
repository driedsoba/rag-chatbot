# app.py
import os
from dotenv import load_dotenv
import boto3
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl

# Load .env
load_dotenv()

# 1) Download faq.txt from S3
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
bucket = os.getenv("S3_BUCKET")
s3.download_file(bucket, "faq.txt", "data/faq.txt")

# 2) Prepare the RAG chain
loader = TextLoader("data/faq.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)
vector_db = FAISS.from_documents(chunks, embed_model)
retriever = vector_db.as_retriever()

llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 3) Chainlit handler
@cl.on_message
async def main(message: str):
    answer = qa_chain.run(message)
    await cl.send_message(answer)
