# app.py
import os
from dotenv import load_dotenv
import boto3
import chainlit as cl
from chainlit import Message

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA

# 0) Load .env
load_dotenv()

# 1) Download faq.txt from S3
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
bucket = os.getenv("S3_BUCKET")
s3.download_file(bucket, "data/faq.txt", "data/faq.txt")

# 2) Prepare the RAG chain
loader = TextLoader("data/faq.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
vector_db = FAISS.from_documents(chunks, embed_model)
retriever = vector_db.as_retriever()

llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    streaming=True  # ← True for Chainlit async
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 3) Chainlit handler
last_etag = None

@cl.on_message
async def main(message: str):
    global last_etag, qa_chain

    # check current ETag of the S3 object
    head = s3.head_object(Bucket=bucket, Key="data/faq.txt")
    if head["ETag"] != last_etag:
        # new file: re-download and rebuild
        s3.download_file(bucket, "data/faq.txt", "data/faq.txt")
        docs = TextLoader("data/faq.txt").load()
        chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        vector_db = FAISS.from_documents(chunks, embed_model)
        retriever = vector_db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        last_etag = head["ETag"]

    answer = await qa_chain.arun(message)
    await cl.send_message(answer)
