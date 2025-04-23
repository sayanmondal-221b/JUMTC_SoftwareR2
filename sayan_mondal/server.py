"""WebSocket server for drone simulator."""
# filepath: /Users/trishit_debsharma/Documents/Code/Mechatronic/software_round2/drone_simulator/server.py
import asyncio
import json
import uuid
import time
from typing import Dict, Any
import websockets
from websockets.server import WebSocketServerProtocol
from drone_simulator.drone import DroneSimulator
from logging_config import get_logger

logger = get_logger("server")

class DroneSimulatorServer:
    """WebSocket server to manage multiple drone simulator sessions."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize the server."""
        logger.info(f"Initializing DroneSimulatorServer on {host}:{port}")
        self.host = host
        self.port = port
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.drones: Dict[str, DroneSimulator] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.last_activity: Dict[str, float] = {}  # Track last activity time for each connection
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}  # Track heartbeat tasks
        self.start_time = time.time()
        logger.debug("Server initialized")

    async def register(self, websocket: WebSocketServerProtocol) -> str:
        """Register a new client connection."""
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        
        # Log connection details
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New connection from {client_info} - assigned ID: {connection_id}")
        
        # Create drone instance for this connection
        self.drones[connection_id] = DroneSimulator(f"telemetry_{connection_id}.json")
        
        # Initialize metrics
        self.metrics[connection_id] = {
            "iterations": 0,
            "total_distance": 0,
            "connection_time": 0,
            "last_position": 0,
            "commands_sent": 0,
            "client_ip": websocket.remote_address[0]
        }
        
        # Record activity time
        self.last_activity[connection_id] = time.time()
        
        logger.info(f"Client registered: {connection_id} from {client_info}")
        logger.info(f"Active connections: {len(self.connections)}")
        return connection_id

    async def unregister(self, connection_id: str) -> None:
        """Unregister a client connection."""
        if connection_id in self.connections:
            # Calculate session duration
            session_duration = 0
            if connection_id in self.last_activity:
                session_duration = time.time() - self.last_activity.get(connection_id, time.time())
            
            # Log metrics before removing
            if connection_id in self.metrics:
                metrics = self.metrics[connection_id]
                logger.info(f"Session stats for {connection_id}: "
                           f"Duration: {session_duration:.1f}s, "
                           f"Commands: {metrics.get('commands_sent', 0)}, "
                           f"Iterations: {metrics.get('iterations', 0)}, "
                           f"Distance: {metrics.get('total_distance', 0):.1f}")
            
            # Clean up resources
            try:
                websocket = self.connections[connection_id]
                client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
                logger.info(f"Unregistering client {connection_id} from {client_info}")
            except:
                logger.info(f"Unregistering client {connection_id}")
                
            del self.connections[connection_id]
            
        if connection_id in self.drones:
            # Check if drone crashed
            drone = self.drones[connection_id]
            if hasattr(drone, 'crashed') and drone.crashed:
                logger.warning(f"Unregistering crashed drone {connection_id}: {drone.crash_reason}")
            del self.drones[connection_id]
            
        if connection_id in self.metrics:
            del self.metrics[connection_id]
            
        if connection_id in self.last_activity:
            del self.last_activity[connection_id]

        # Cancel and remove heartbeat task if exists
        if connection_id in self.heartbeat_tasks:
            if not self.heartbeat_tasks[connection_id].done():
                self.heartbeat_tasks[connection_id].cancel()
            del self.heartbeat_tasks[connection_id]
            
        logger.info(f"Client unregistered: {connection_id}")
        logger.info(f"Active connections: {len(self.connections)}")

    async def handle_drone_command(self, connection_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a drone command and update metrics."""
        logger.debug(f"Processing command from {connection_id}: {data}")
        
        # Check if connection still exists
        if connection_id not in self.drones or connection_id not in self.metrics:
            logger.warning(f"Cannot process command - connection {connection_id} no longer exists")
            return {
                "status": "error",
                "message": "Connection no longer exists"
            }
            
        drone = self.drones[connection_id]
        metrics = self.metrics[connection_id]
        
        # Update last activity time
        self.last_activity[connection_id] = time.time()
        
        # Increment command count
        metrics["commands_sent"] = metrics.get("commands_sent", 0) + 1
        
        try:
            # Get previous position for distance calculation
            prev_position = drone.telemetry["x_position"]
            
            # Update drone telemetry based on user input
            telemetry = drone.update_telemetry(data)
            
            # Calculate metrics
            if data.get("speed", 0) != 0 and data.get("altitude", 0) != 0:
                metrics["iterations"] += 1
                distance_traveled = abs(telemetry["x_position"] - prev_position)
                metrics["total_distance"] += distance_traveled
                logger.info(f"Client {connection_id} flight iteration {metrics['iterations']}: "
                           f"Distance: +{distance_traveled:.1f}, Total: {metrics['total_distance']:.1f}")
            
            metrics["last_position"] = telemetry["x_position"]
            
            # Encode telemetry in string format
            telemetry_str = (
                f"X-{telemetry['x_position']}-"
                f"Y-{telemetry['y_position']}-"
                f"BAT-{telemetry['battery']}-"
                f"GYR-{telemetry['gyroscope']}-"
                f"WIND-{telemetry['wind_speed']}-"
                f"DUST-{telemetry['dust_level']}-"
                f"SENS-{telemetry['sensor_status']}"
            )
            
            # Include metrics and encoded telemetry in the response
            response = {
                "status": "success",
                "telemetry": telemetry_str,
                "metrics": {
                    "iterations": metrics["iterations"],
                    "total_distance": metrics["total_distance"]
                }
            }
            logger.debug(f"Command processed successfully for {connection_id}")
            return response
            
        except ValueError as e:
            crash_message = str(e)
            logger.warning(f"Drone crashed for {connection_id}: {crash_message}")

            final_telemetry_str = (
                f"X-{drone.telemetry['x_position']}-"
                f"Y-{drone.telemetry['y_position']}-"
                f"BAT-{drone.telemetry['battery']}-"
                f"GYR-{drone.telemetry['gyroscope']}-"
                f"WIND-{drone.telemetry['wind_speed']}-"
                f"DUST-{drone.telemetry['dust_level']}-"
                f"SENS-{drone.telemetry['sensor_status']}"
            )
            
            # Create a detailed crash response
            response = {
                "status": "crashed",
                "message": crash_message,
                "metrics": {
                    "iterations": metrics["iterations"],
                    "total_distance": metrics["total_distance"]
                },
                "final_telemetry": final_telemetry_str,
                "connection_terminated": True
            }
            
            logger.info(f"Sending crash response to {connection_id}: {crash_message}")
            return response

    async def handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection."""
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New connection handler started for client: {client_info}")
        
        connection_id = await self.register(websocket)
        
        try:
            # Send initial connection message
            welcome_msg = {
                "status": "connected",
                "connection_id": connection_id,
                "message": "Welcome to the Drone Simulator! Send commands to control your drone."
            }
            await websocket.send(json.dumps(welcome_msg))
            logger.info(f"Welcome message sent to {connection_id}")
            
            # Start heartbeat task for this connection (as a separate task)
            logger.debug(f"Starting heartbeat task for {connection_id}")
            self.heartbeat_tasks[connection_id] = asyncio.create_task(
                self.connection_heartbeat(connection_id, websocket)
            )
            
            # Process messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received from {connection_id}: {data}")
                    
                    # Update last activity time
                    if connection_id in self.last_activity:
                        self.last_activity[connection_id] = time.time()
                    else:
                        logger.warning(f"Connection {connection_id} no longer registered")
                        break
                    
                    # Process the command
                    response = await self.handle_drone_command(connection_id, data)
                    
                    # Check if connection still exists before sending response
                    if connection_id not in self.connections:
                        logger.warning(f"Cannot send response - connection {connection_id} no longer exists")
                        break
                    
                    # Send response back to client
                    await websocket.send(json.dumps(response))
                    logger.debug(f"Response sent to {connection_id}")
                    
                    # If the drone has crashed, terminate the connection
                    if response.get("status") == "crashed" and response.get("connection_terminated", False):
                        logger.info(f"Terminating connection for {connection_id} due to drone crash")
                        await websocket.close(code=1000, reason=f"Drone crashed: {response.get('message')}")
                        break
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received from {connection_id}: {message}")
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid JSON format"
                    }))
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed for {connection_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e}", exc_info=True)
        finally:
            await self.unregister(connection_id)

    async def connection_heartbeat(self, connection_id: str, websocket: WebSocketServerProtocol) -> None:
        """Send periodic pings to keep the connection alive."""
        logger.debug(f"Heartbeat started for {connection_id}")
        
        try:
            while True:
                # Check if connection still exists before attempting ping
                if connection_id not in self.connections:
                    logger.debug(f"Connection {connection_id} no longer exists, stopping heartbeat")
                    break
                
                # Send a ping to check the connection
                logger.debug(f"Sending ping to {connection_id}")
                try:
                    pong_waiter = await websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5)  # Changed from 10 to 5 seconds
                    logger.debug(f"Received pong from {connection_id}")
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning(f"Ping timeout for {connection_id}, closing connection")
                    try:
                        # Close connection with status code 1011 (internal error)
                        await websocket.close(code=1011, reason="Ping timeout")
                    except:
                        pass
                    break
                
                # Check for inactivity more frequently
                if connection_id in self.last_activity:
                    current_time = time.time()
                    last_active = self.last_activity.get(connection_id, 0)
                    inactivity_duration = current_time - last_active
                    
                    logger.debug(f"Client {connection_id} inactive for {inactivity_duration:.1f}s")
                    
                    if inactivity_duration > 5:  # Changed from 120 to 5 seconds inactivity timeout
                        logger.warning(f"Client {connection_id} inactive for {inactivity_duration:.1f}s, closing connection")
                        try:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "Connection closed due to inactivity",
                            }))
                            await websocket.close(code=1000, reason="Inactivity timeout")
                        except:
                            pass
                        break
                
                # Wait before next ping - check more frequently
                await asyncio.sleep(5)  # Changed from 30 to 5 seconds
                
        except asyncio.CancelledError:
            logger.debug(f"Heartbeat task cancelled for {connection_id}")
        except Exception as e:
            logger.error(f"Error in heartbeat for {connection_id}: {e}", exc_info=True)

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        # Configure server with ping_interval and ping_timeout
        server = await websockets.serve(
            self.handle_connection, 
            self.host, 
            self.port,
            ping_interval=5,  # Changed from 30 to 5 seconds
            ping_timeout=5,   # Changed from 10 to 5 seconds
            max_size=10_485_760  # 10MB max message size (default is 1MB)
        )
        
        logger.info(f"Server started successfully on ws://{self.host}:{self.port}")
        logger.info("Waiting for connections...")
        
        # Stats logging task
        async def log_periodic_stats():
            while True:
                await asyncio.sleep(300)  # Log stats every 5 minutes
                uptime = time.time() - self.start_time
                connected_clients = len(self.connections)
                
                # Calculate total metrics across all drones
                total_iterations = sum(m.get("iterations", 0) for m in self.metrics.values())
                total_distance = sum(m.get("total_distance", 0) for m in self.metrics.values())
                total_commands = sum(m.get("commands_sent", 0) for m in self.metrics.values())
                
                logger.info(f"Server stats - Uptime: {uptime:.1f}s, Clients: {connected_clients}, "
                           f"Total iterations: {total_iterations}, "
                           f"Total distance: {total_distance:.1f}, "
                           f"Total commands: {total_commands}")
        
        # Start stats logging task separately
        stats_task = asyncio.create_task(log_periodic_stats())
        
        # Keep server running forever
        await asyncio.Future()  # This line replaces the while True loop


def main() -> None:
    """Start the drone simulator server."""
    logger.info("Starting Drone Simulator Server...")
    
    server = DroneSimulatorServer(host="0.0.0.0")
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"Server crashed: {e}", exc_info=True)


if __name__ == "__main__":
    main()