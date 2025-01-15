import logging
from fastapi import FastAPI, WebSocket, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logger = logging.getLogger("server_logger")
logger.setLevel(logging.DEBUG)

# StreamHandler for console output
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response

# WebSocket server using FastAPI
app = FastAPI()

# Add the logging middleware
app.add_middleware(LoggingMiddleware)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket connection accepted")
    await websocket.accept()  # Accept the WebSocket connection
    try:
        while True:
            message = await websocket.receive_text()  # Receive text message from client
            logger.info(f"Received: {message}")
            if message == "PING":
                logger.info("Sending PONG")
                await websocket.send_text("PONG")  # Send PONG to client
            else:
                logger.warning("Unknown message received")
                await websocket.send_text("Unknown message")
    except Exception as e:
        logger.error(f"Connection closed due to: {e}")
