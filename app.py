import os
import boto3
import chainlit as cl
from dotenv import load_dotenv

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 0) Load .env if present (optional)
load_dotenv()

# 1) Read config from env
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET           = os.getenv("S3_BUCKET")
EMBED_MODEL_ID      = os.getenv("EMBED_MODEL_ID")
LLM_MODEL_ID        = os.getenv("LLM_MODEL_ID")

# 2) Download your FAQ from S3
s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
s3.download_file(S3_BUCKET, "data/faq.txt", "data/faq.txt")

# 3) Load & split your FAQ
loader   = TextLoader("data/faq.txt")
docs     = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

# 4) Embed + build FAISS index (in-memory)
embed_model = BedrockEmbeddings(
    model_id=EMBED_MODEL_ID,
    region_name=AWS_DEFAULT_REGION
)
vector_db = FAISS.from_documents(chunks, embed_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 5) Bedrock LLM (streaming for Chainlit)
llm = Bedrock(
    model_id=LLM_MODEL_ID,
    region_name=AWS_DEFAULT_REGION,
    streaming=True
)

# 6) Build a simple RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 7) Chainlit handler
@cl.on_message
async def main(message: str):
    answer = await qa_chain.arun(message)
    await cl.send_message(answer)
