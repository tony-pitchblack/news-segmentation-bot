cat << 'EOF' > simple_server.py
import asyncio
import websockets
import socketserver
import http.server

# WebSocket handler function
async def hello(websocket, path):
    await websocket.send('Hello, World!')

# HTTP server to respond to curl
def run_http_server():
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Hello, World!\n')
            else:
                super().do_GET()

    port = 9090
    with socketserver.TCPServer(("", port), MyHandler) as httpd:
        print(f'Starting HTTP server on port {port}')
        httpd.serve_forever()

# WebSocket server to handle WebSocket requests
async def run_websocket_server():
    port = 9090
    print(f'Starting WebSocket server on ws://0.0.0.0:{port}')
    async with websockets.serve(hello, '0.0.0.0', port):
        await asyncio.Future()  # run forever

# Run both WebSocket and HTTP servers concurrently
async def main():
    await asyncio.gather(
        run_websocket_server(),
        asyncio.to_thread(run_http_server)
    )

# Start the servers
if __name__ == '__main__':
    asyncio.run(main())
EOF
