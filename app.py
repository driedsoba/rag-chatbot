# app.py
import os
import chainlit as cl
from dotenv import load_dotenv
import boto3

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ────────────────────────────────────────────────────────────────────────────
# 0) Load ENV
load_dotenv()
AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET   = os.getenv("S3_BUCKET")
EMBED_MODEL = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
LLM_MODEL   = os.getenv("LLM_MODEL_ID",   "amazon.titan-text-express-v1")
# ────────────────────────────────────────────────────────────────────────────

# 1) Fetch FAQ from S3
s3 = boto3.client("s3", region_name=AWS_REGION)
os.makedirs("data", exist_ok=True)
s3.download_file(S3_BUCKET, "data/faq.txt", "data/faq.txt")

# 2) Load & chunk
loader   = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
docs     = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

# 3) Embed + FAISS (persist once)
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
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 4) Custom “paraphrase in two sentences” prompt
STUFF_PROMPT = """You are Jun Le’s friendly assistant.  Answer in **at most two sentences** and **paraphrase** rather than copy.

Context:
{context}

Question:
{question}

Answer:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=STUFF_PROMPT)

# 5) LLM + conversational chain w/ memory
llm = Bedrock(
    model_id=LLM_MODEL,
    region_name=AWS_REGION,
    streaming=True,
    model_kwargs={"temperature": 0.2, "maxTokenCount": 512},
)
memory  = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": prompt},
)

# ────────────────────────────────────────────────────────────────────────────
# 6) Greet the user as soon as the page loads
@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I’m Jun Le’s personal assistant – ask me anything about him in two sentences or less."
    ).send()

# 7) Handle incoming questions
@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()

    # “I don’t know” fallback
    docs_and_scores = vector_db.similarity_search_with_score(user_text, k=5)
    if not docs_and_scores or docs_and_scores[0][1] < 0.1:
        await cl.Message(content="Sorry, I don’t have that info—could you rephrase?").send()
        return

    # Run the chain and reply
    res = await qa_chain.ainvoke({"question": user_text})
    await cl.Message(content=res["answer"]).send()
