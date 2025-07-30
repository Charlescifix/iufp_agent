#!/usr/bin/env python3
"""
Simple HTTP server to serve the IUFP chat interface
This avoids CORS issues when accessing from file:// URLs
"""

import http.server
import socketserver
import webbrowser
import os
import threading
import time

PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the HTTP server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"IUFP Chat Interface Server running at:")
        print(f"   http://localhost:{PORT}/")
        print(f"   http://localhost:{PORT}/index.html")
        print(f"")
        print(f"Direct chat access:")
        print(f"   http://localhost:{PORT}/iufp_chat.html")
        print(f"")
        print(f"Make sure the API server is also running at:")
        print(f"   http://localhost:8000")
        print(f"")
        print(f"Press Ctrl+C to stop the server")
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{PORT}/')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    start_server()