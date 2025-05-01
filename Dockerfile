# 1. Use a slim Python base
FROM python:3.10-slim

# 2. Install system tools (for faiss)
RUN apt-get update \
  && apt-get install -y build-essential \
  && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 6. Expose Chainlit
EXPOSE 8000

# 7. Start Chainlit using our new app.py
CMD ["chainlit","run","app.py","--host","127.0.0.1","--port","8000"]