import os
import json
import boto3
from dotenv import load_dotenv
import chainlit as cl
from chainlit import Message

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.bedrock import BedrockEmbeddings

load_dotenv()

# 1) Download your FAQ from S3
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
s3.download_file(
    os.getenv("S3_BUCKET"),
    "data/faq.txt",
    "data/faq.txt"
)

# 2) Load, split, embed & build (or reload) FAISS index
loader = TextLoader("data/faq.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embed_model = BedrockEmbeddings(
    model_id=os.getenv("EMBED_MODEL_ID"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

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

# 3) Prepare Bedrock streaming client
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION"))
MODEL_ID = os.getenv("LLM_MODEL_ID")

def converse_stream(model_id: str, messages: list[dict], temperature=0.2, max_tokens=400):
    payload = {
        "messages": messages,
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": max_tokens
        }
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json"
    )["responseStream"]

    for event in response:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        text = delta.get("text")
        if text:
            yield text

# 4) Chainlit handler
@cl.on_message
async def main(message: Message):
    # extract the raw user input
    user_text = message.content  

    # 4a) Retrieve context
    docs_and_scores = retriever.get_relevant_documents(user_text)
    if not docs_and_scores or max(d.score for d in docs_and_scores) < 0.1:
        # simple fallback
        await Message(content="Sorry, I don’t have data on that—can you rephrase?").send()
        return

    # 4b) Build system prompt with context
    context = "\n\n".join(d.page_content for d in docs_and_scores)
    system_prompt = (
        "You are Jun Le’s personal assistant helping to answer people's questions about him. Use the following context to answer:\n\n"
        + context
    )

    msgs = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user",   "content": [{"text": user_text}]}
    ]

    # 4c) Stream the answer back
    reply = Message()
    async for chunk in converse_stream(MODEL_ID, msgs):
        await reply.stream(chunk)
    await reply.send()
