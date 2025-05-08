# config.py
import os
from dotenv import load_dotenv
import boto3
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# ─── 0) Load ENV ───────────────────────────────────────────────────────────────
load_dotenv()
AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET   = os.getenv("S3_BUCKET")
EMBED_MODEL = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
LLM_MODEL   = os.getenv("LLM_MODEL_ID",   "amazon.titan-text-express-v1")

# ─── 1) Fetch FAQ from S3 ──────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=AWS_REGION)
os.makedirs("data", exist_ok=True)
s3.download_file(S3_BUCKET, "data/faq.txt", "data/faq.txt")

# ─── 2) Load & chunk ──────────────────────────────────────────────────────────
loader   = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
docs     = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

# ─── 3) Embed + FAISS (persist once) ─────────────────────────────────────────
embed_model = BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)
if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local(
        "faiss_index",
        embed_model,
        allow_dangerous_deserialization=True
    )
else:
    vector_db = FAISS.from_documents(chunks, embed_model)
    vector_db.save_local("faiss_index")

# ─── 4) Retriever ─────────────────────────────────────────────────────────────
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
