# app.py
import os
import asyncio
from dotenv import load_dotenv
import boto3

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

import chainlit as cl

# load environment
load_dotenv()
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET  = os.getenv("S3_BUCKET", "rag-faqstore")

# 1) fetch your FAQ from S3
s3 = boto3.client("s3", region_name=AWS_REGION)
s3.download_file(S3_BUCKET, "data/faq.txt", "data/faq.txt")

# 2) build your vector index
loader   = TextLoader("data/faq.txt")
docs     = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

embed_model = BedrockEmbeddings(
    model_id    = "amazon.titan-embed-text-v1",
    region_name = AWS_REGION,
)

vector_db = FAISS.from_documents(chunks, embed_model)
retriever = vector_db.as_retriever()

llm = Bedrock(
    model_id    = "amazon.titan-text-express-v1",
    region_name = AWS_REGION,
    streaming   = True,           # ← turn on streaming for async calls
)

qa_chain = RetrievalQA.from_chain_type(
    llm        = llm,
    retriever  = retriever,
    return_source_documents = False,
)

# 3) Chainlit handler
@cl.on_message
async def main(message):
    user_input = message.content
    # you can still use .arun() now that streaming=True
    answer = await qa_chain.arun(user_input)
    # send back the answer to the user
    await cl.send("Answering...")
    await cl.send(answer)  # send the actual answer to the user

