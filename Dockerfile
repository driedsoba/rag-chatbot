# 1. Use a slim Python base
FROM python:3.10-slim

# 2. Install system tools (for faiss)
RUN apt-get update \
  && apt-get install -y build-essential curl \
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

# 7. Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 8. Start Chainlit and health check in background
CMD ["sh", "-c", "python health_check.py & chainlit run app.py --host 0.0.0.0 --port 8000"]