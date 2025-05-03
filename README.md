# ChatWithJunLe

How “ChatWithJunLe” is standing up as a personalised RAG chatbot.

---

## 1. Architecture in a Nutshell

- **AWS EC2**  
  Hosts a Docker container running Chainlit + Python app.
- **Docker**  
  Packages app, dependencies, FAISS index, and Chainlit server.
- **S3**  
  Stores `faq.txt` that the bot indexes at startup.
- **FAISS**  
  Vector-search index, persisted to disk to only embed once.
- **AWS Bedrock**  
  Provides embeddings & streaming LLM inference.
- **Nginx + Let’s Encrypt**  
  Reverse-proxies `chatwithjunle.com` → `localhost:8000` over HTTPS.
- **GoDaddy (or any DNS)**  
  Points custom domain at the EC2 public IP.

---

## 2. Deployment Overview

1. **Domain & DNS**  
   • Create A record for `chatwithjunle.com` → your EC2 IP.  
2. **EC2 Instance**  
   • Launch with an IAM role that reads S3 bucket & call Bedrock.  
   • Install Docker & Nginx.  
3. **Clone & Configure**  
   • `git clone … && cd chatwithjunle`  
   • Create a `.env` with:
     ```text
     AWS_DEFAULT_REGION=us-east-1
     S3_BUCKET=<your-bucket>
     EMBED_MODEL_ID=amazon.titan-embed-text-v1
     LLM_MODEL_ID=amazon.titan-text-express-v1
     ```
4. **Build & Run**  
   ```bash
   docker build -t rag-chatbot .
   docker run -d --name rag-chatbot \
     -p 127.0.0.1:8000:8000 \
     -e AWS_DEFAULT_REGION \
     -e S3_BUCKET \
     -e EMBED_MODEL_ID \
     -e LLM_MODEL_ID \
     --restart unless-stopped \
     rag-chatbot
