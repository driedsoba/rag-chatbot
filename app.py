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

load_dotenv()

# 1) Sync docs from S3
s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
s3.download_file(os.getenv("S3_BUCKET"), "data/faq.txt", "data/faq.txt")

# 2) Load & split
loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) Embeddings & (persistent) FAISS
embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
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

# 4) LLM & Conversational Chain
llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    streaming=True,
    model_kwargs={
        "temperature": 0.2,
        "max_tokens_to_sample": 512,
    }
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 5) Handlers
@cl.on_message
async def main(user_input: str):
    docs_and_scores = retriever.get_relevant_documents(user_input)
    if not docs_and_scores or max(d.score for d in docs_and_scores) < 0.1:
        await cl.send_message("Sorry, I don’t have data on that—can you rephrase?")
        return

    result = await qa_chain.acall({"question": user_input})
    await cl.send_message(result["answer"])
