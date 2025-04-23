"""Test client for drone simulator WebSocket server."""
# filepath: /Users/trishit_debsharma/Documents/Code/Mechatronic/software_round2/drone_simulator/client.py
import asyncio
import json
import sys
import websockets
import time
from typing import Dict, Any, Optional
from logging_config import get_logger


logger = get_logger("client")

class DroneClient:
    """WebSocket client for testing the drone simulator."""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        """Initialize the client."""
        self.uri = uri
        self.connection_id = None
        self.telemetry = None
        self.metrics = None
        self.start_time = time.time()
        self.command_count = 0
        logger.info(f"Drone client initialized with server URI: {uri}")
    
    async def connect(self) -> None:
        """Connect to the WebSocket server."""
        logger.info(f"Attempting to connect to {self.uri}")
        print(f"Attempting to connect to {self.uri}...")
        print("Make sure the server is running (python run_server.py)")
        
        try:
            # Configure ping_interval and ping_timeout properly
            logger.debug("Establishing WebSocket connection")
            async with websockets.connect(
                self.uri, 
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait 10 seconds for pong response
                close_timeout=5    # Wait 5 seconds for close to complete
            ) as websocket:
                # Receive welcome message
                response = await websocket.recv()
                data = json.loads(response)
                self.connection_id = data.get("connection_id")
                logger.info(f"Connected successfully with ID: {self.connection_id}")
                logger.info(f"Server message: {data['message']}")
                
                print(f"Connected with ID: {self.connection_id}")
                print(f"Server says: {data['message']}")
                
                # Interactive control of the drone
                await self.interactive_control(websocket)
                
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Connection closed abnormally: {e}")
            print("\nThe connection was closed unexpectedly. Possible reasons:")
            print("- Server crashed or restarted")
            print("- Network issues causing ping timeout")
            print("- Server closed the connection due to inactivity")
            
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally by the server")
            
        except ConnectionRefusedError:
            logger.error(f"Connection refused. Is the server running at {self.uri}?")
            print("\nTroubleshooting steps:")
            print("1. Make sure the server is running: python run_server.py")
            print("2. Check if the server is listening on the correct address")
            print("3. Check if there are any firewalls blocking the connection")
            print("4. Try 'ws://127.0.0.1:8765' instead of 'ws://localhost:8765'")
            
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            print(f"\nConnection error: {e}")
        
        finally:
            # Log session summary
            session_duration = time.time() - self.start_time
            logger.info(f"Session summary - "
                      f"Duration: {session_duration:.1f}s, "
                      f"Commands sent: {self.command_count}, "
                      f"Connection ID: {self.connection_id}")
    
    async def send_command(self, websocket, speed: int, altitude: int, movement: str) -> Optional[Dict[str, Any]]:
        """Send a command to the drone server and return the response."""
        try:
            data = {
                "speed": speed,
                "altitude": altitude,
                "movement": movement
            }
            self.command_count += 1
            logger.info(f"Sending command #{self.command_count}: {data}")
            
            await websocket.send(json.dumps(data))
            
            response = await websocket.recv()
            response_data = json.loads(response)
            
            # Check if the drone has crashed
            if response_data.get("status") == "crashed":
                crash_message = response_data.get('message', 'Unknown crash')
                logger.warning(f"Drone crashed: {crash_message}")
                
                print(f"\n*** DRONE CRASHED: {crash_message} ***")
                print("Connection will be terminated.")
                
                # Update metrics one last time
                if "metrics" in response_data:
                    self.metrics = response_data["metrics"]
                    logger.info(f"Final metrics: {self.metrics}")
                
                # Show final telemetry
                if "final_telemetry" in response_data:
                    self.telemetry = response_data["final_telemetry"]
                    logger.info(f"Final telemetry: {self.telemetry}")
                    self.display_status()
                
                print("\nFinal Flight Statistics:")
                print(f"Total distance traveled: {self.metrics.get('total_distance', 0)}")
                print(f"Successful flight iterations: {self.metrics.get('iterations', 0)}")
                print("\nConnection terminated due to crash")
                
                # Return None to indicate a crash occurred
                return None
            
            logger.debug(f"Received response: {response_data}")
            return response_data
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"Connection closed while sending command: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error sending command: {e}", exc_info=True)
            return None
    
    async def interactive_control(self, websocket) -> None:
        """Interactively control the drone through the console."""
        logger.info("Starting interactive control")
        
        print("\n==== Drone Simulator Interactive Console ====")
        print("Commands: 'exit' to quit, 'help' for instructions, 'auto' for auto pilot")
        print("Input format: speed,altitude,movement (e.g., '2,0,fwd')")
        print("Keep-alive pings are sent automatically every 20 seconds")
        
        help_text = """
Commands:
- speed: integer 0-5
- altitude: positive or negative integer
- movement: 'fwd' or 'rev'

Examples:
- 3,0,fwd   # Move forward at speed 3
- 0,5,fwd   # Gain altitude by 5 units
- 2,-1,rev  # Move backward and descend 1 unit
- auto      # Start auto pilot mode
- exit      # Exit the client
- help      # Show this help message
- status    # Show current telemetry and metrics
- ping      # Send a keep-alive command (0,0,fwd)
        """
        
        try:
            while True:
                command = input("\nEnter command: ")
                logger.debug(f"User entered command: {command}")
                
                if command.lower() == 'exit':
                    logger.info("User requested exit")
                    break
                    
                if command.lower() == 'help':
                    print(help_text)
                    continue
                
                if command.lower() == 'status' and self.telemetry:
                    self.display_status()
                    continue
                
                if command.lower() == 'auto':
                    logger.info("Starting auto pilot mode")
                    await self.auto_pilot(websocket)
                    continue
                    
                if command.lower() == 'ping':
                    print("Sending keep-alive ping...")
                    logger.info("User requested ping")
                    data = await self.send_command(websocket, 0, 0, "fwd")
                    if data:
                        self.update_state(data)
                        print("Keep-alive successful")
                    continue
                
                try:
                    # Parse command
                    parts = command.split(',')
                    if len(parts) != 3:
                        print("Invalid command format. Use: speed,altitude,movement")
                        logger.warning(f"Invalid command format: {command}")
                        continue
                        
                    speed = int(parts[0])
                    altitude = int(parts[1])
                    movement = parts[2].strip()
                    
                    # Send command
                    data = await self.send_command(websocket, speed, altitude, movement)
                    if data:
                        self.update_state(data)
                        self.display_status()
                    elif data is None:  # Crash occurred
                        break
                        
                except ValueError as e:
                    print(f"Invalid input format: {e}")
                    print("Use format: speed,altitude,movement (e.g., '2,0,fwd')")
                    logger.warning(f"Invalid input format: {e}")
                
        except KeyboardInterrupt:
            logger.info("User interrupted the client with Ctrl+C")
            print("\nExiting...")
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection to server was closed")
            print("\nConnection to server was closed")
    
    async def auto_pilot(self, websocket) -> None:
        """Run an automated test sequence."""
        logger.info("Starting auto pilot sequence")
        print("\n==== Auto Pilot Mode ====")
        print("Press Ctrl+C to exit auto pilot")
        
        try:
            # Test sequence
            actions = [
                (2, 0, "fwd"),   # Move forward
                (3, 1, "fwd"),   # Move forward and gain altitude
                (4, 2, "fwd"),   # Move forward faster and gain more altitude
                (5, 0, "fwd"),   # Max speed
                (3, -1, "fwd"),  # Slow down and descend
                (2, 0, "rev"),   # Reverse
                (3, 0, "rev"),   # Reverse faster
                (1, 1, "fwd"),   # Slow forward and gain altitude
                (0, 0, "fwd"),   # Stop
            ]
            
            for i, (speed, altitude, movement) in enumerate(actions, 1):
                logger.info(f"Auto pilot step {i}/{len(actions)}: "
                          f"speed={speed}, altitude={altitude}, movement={movement}")
                print(f"\nAuto pilot step {i}/{len(actions)}")
                print(f"Sending command: speed={speed}, altitude={altitude}, movement={movement}")
                
                data = await self.send_command(websocket, speed, altitude, movement)
                if data:
                    self.update_state(data)
                    self.display_status()
                else:
                    logger.warning("Auto pilot aborted due to crash or error")
                    print("Auto pilot aborted")
                    return
                    
                await asyncio.sleep(1)  # Pause between commands
                
            logger.info("Auto pilot sequence completed successfully")
            print("\nAuto pilot sequence completed")
            
        except KeyboardInterrupt:
            logger.info("Auto pilot stopped by user")
            print("\nAuto pilot stopped")
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection to server was closed during auto pilot")
            print("\nConnection to server was closed")
    
    def update_state(self, data: Dict[str, Any]) -> None:
        """Update client state with server response."""
        if data["status"] == "success":
            self.telemetry = data["telemetry"]
            self.metrics = data["metrics"]
            logger.debug(f"Updated state with telemetry: {self.telemetry}")
            logger.debug(f"Updated state with metrics: {self.metrics}")
        else:
            logger.warning(f"Error response: {data['message']}")
            print(f"\nError: {data['message']}")
            if "metrics" in data:
                self.metrics = data["metrics"]
    
    def display_status(self) -> None:
        """Display current telemetry and metrics."""
        if not self.telemetry:
            print("No telemetry data available yet")
            return
            
        print("\n----- Telemetry -----")
        # print(f"Position: ({self.telemetry['x_position']}, {self.telemetry['y_position']})")
        # print(f"Battery: {self.telemetry['battery']:.1f}%")
        # print(f"Wind Speed: {self.telemetry['wind_speed']}%")
        # print(f"Dust Level: {self.telemetry['dust_level']}%")
        # print(f"Sensor Status: {self.telemetry['sensor_status']}")
        # print(f"Gyroscope: {self.telemetry['gyroscope']}")
        print(self.telemetry)
        
        print("\n----- Metrics -----")
        print(f"Successful Iterations: {self.metrics['iterations']}")
        print(f"Total Distance: {self.metrics['total_distance']}")
        
        # logger.info(f"Status displayed - Position: ({self.telemetry['x_position']}, {self.telemetry['y_position']}), "
        #            f"Battery: {self.telemetry['battery']:.1f}%, "
        #            f"Iterations: {self.metrics['iterations']}, "
        #            f"Distance: {self.metrics['total_distance']}")

def main() -> None:
    """Start the drone client."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        uri = sys.argv[1]
    else:
        uri = "ws://localhost:8765"
    
    logger.info(f"Starting Drone Client with server URI: {uri}")
    
    client = DroneClient(uri)
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
        print("\nClient stopped by user")

if __name__ == "__main__":
    main()