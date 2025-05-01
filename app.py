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
    streaming=True  # ← must be True for Chainlit async
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 3) Chainlit handler
@cl.on_message
async def main(message: cl.Message):
    user_input = message.content

    # loading indicator
    await Message("Thinking…").send()

    # run the RAG chain
    answer = await qa_chain.arun(user_input)

    # send the final answer
    await Message(answer).send()
