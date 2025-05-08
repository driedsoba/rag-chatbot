import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# HTTP server that returns 200 OK for /health

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            # Check that key components are running
            try:
                # Check if FAISS index exists
                if not os.path.exists("faiss_index"):
                    self.send_response(503)  # Service Unavailable
                    self.end_headers()
                    self.wfile.write(b"FAISS index not found")
                    return
                
                # All checks passed
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

if __name__ == "__main__":
    port = int(os.environ.get("HEALTH_PORT", 8000))
    server = HTTPServer(('localhost', port), HealthCheckHandler)
    print(f"Starting health check server on port {port}")
    server.serve_forever()