"""Admin server for drone simulator monitoring."""
import asyncio
import json
import logging
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Dict, Set, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AdminServer:
    """Admin server for monitoring drone simulator connections."""
    
    def __init__(self, host: str = "localhost", port: int = 8766, main_server=None):
        """Initialize the admin server."""
        self.host = host
        self.port = port
        self.main_server = main_server  # Reference to the main DroneSimulatorServer
        self.admin_connections: Set[WebSocketServerProtocol] = set()
        self.admin_key = "admin_secret"  # Simple authentication
    
    async def register_admin(self, websocket: WebSocketServerProtocol) -> None:
        """Register an admin connection."""
        self.admin_connections.add(websocket)
        logger.info(f"Admin connected: {websocket.remote_address}")
    
    async def unregister_admin(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister an admin connection."""
        if websocket in self.admin_connections:
            self.admin_connections.remove(websocket)
        logger.info(f"Admin disconnected: {websocket.remote_address}")
    
    async def handle_admin_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle an admin connection."""
        try:
            # Wait for authentication
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "admin_auth" and data.get("key") == self.admin_key:
                await self.register_admin(websocket)
                
                # Process admin commands
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("type") == "get_all_connections":
                        await self.send_connection_update(websocket)
            else:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Authentication failed"
                }))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_admin(websocket)
    
    async def send_connection_update(self, websocket: WebSocketServerProtocol) -> None:
        """Send connection update to admin."""
        if not self.main_server:
            await websocket.send(json.dumps({
                "type": "connection_update",
                "connections": {}
            }))
            return
        
        connections_data = {}
        for conn_id in self.main_server.connections:
            connections_data[conn_id] = {
                "telemetry": self.main_server.drones[conn_id].telemetry,
                "metrics": self.main_server.metrics[conn_id]
            }
        
        await websocket.send(json.dumps({
            "type": "connection_update",
            "connections": connections_data
        }))
    
    async def broadcast_update(self) -> None:
        """Broadcast connection updates to all admin clients."""
        if not self.admin_connections:
            return
        
        connections_data = {}
        if self.main_server:
            for conn_id in self.main_server.connections:
                connections_data[conn_id] = {
                    "telemetry": self.main_server.drones[conn_id].telemetry,
                    "metrics": self.main_server.metrics[conn_id]
                }
        
        message = json.dumps({
            "type": "connection_update",
            "connections": connections_data
        })
        
        for admin in self.admin_connections:#just >> self.admin_connections
            try:
                await admin.send(message)
            except websockets.exceptions.ConnectionClosed:
                pass
    
    async def start_server(self) -> None:
        """Start the admin WebSocket server."""
        async with websockets.serve(self.handle_admin_connection, self.host, self.port):
            logger.info(f"Admin server started on ws://{self.host}:{self.port}")
            
            # Broadcast updates periodically
            while True:
                await self.broadcast_update()
                await asyncio.sleep(2)  # Update every 2 seconds