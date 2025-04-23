"""Admin dashboard for monitoring all drone connections."""
import asyncio
import json
import logging
import sys
import websockets
import datetime
from tabulate import tabulate  # You may need to install this: pip install tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DashboardClient:
    """Admin dashboard for monitoring drone simulator connections."""
    
    def __init__(self, uri: str = "ws://localhost:8766"):
        """Initialize the dashboard client."""
        self.uri = uri
        self.connections = {}
        self.update_interval = 2  # seconds
    
    async def connect(self) -> None:
        """Connect to the admin WebSocket endpoint."""
        try:
            async with websockets.connect(self.uri) as websocket:
                # Authenticate as admin (if needed)
                await websocket.send(json.dumps({"type": "admin_auth", "key": "admin_secret"}))
                
                # Start monitoring
                await self.monitor_connections(websocket)
                
        except Exception as e:
            logger.error(f"Error: {e}")
    
    async def monitor_connections(self, websocket) -> None:
        """Monitor all drone connections."""
        try:
            # Request initial data
            await websocket.send(json.dumps({"type": "get_all_connections"}))
            
            # Setup display loop
            while True:
                # Receive updates
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "connection_update":
                    self.connections = data.get("connections", {})
                    self.display_connections()
                
                # Request updates periodically
                await asyncio.sleep(self.update_interval)
                await websocket.send(json.dumps({"type": "get_all_connections"}))
                
        except KeyboardInterrupt:
            print("\nExiting dashboard...")
    
    def display_connections(self) -> None:
        """Display all connections in a table format."""
        # Clear screen (works on most terminals)
        print("\033c", end="")
        
        print(f"Drone Simulator Dashboard - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Active Connections: {len(self.connections)}\n")
        
        if not self.connections:
            print("No active connections")
            return
        
        # Prepare table data
        table_data = []
        for conn_id, data in self.connections.items():
            table_data.append([
                conn_id[:8] + "...",  # Truncated connection ID
                data["metrics"]["iterations"],
                data["metrics"]["total_distance"],
                data["telemetry"]["x_position"],
                data["telemetry"]["y_position"],
                f"{data['telemetry']['battery']:.1f}%",
                data["telemetry"]["sensor_status"]
            ])
        
        # Display table
        headers = ["Connection", "Iterations", "Distance", "X Pos", "Y Pos", "Battery", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\nPress Ctrl+C to exit")

def main() -> None:
    """Start the dashboard client."""
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8766"
    dashboard = DashboardClient(uri)
    try:
        asyncio.run(dashboard.connect())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")

if __name__ == "__main__":
    main()