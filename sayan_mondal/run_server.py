"""
Standalone script to run the drone simulator server.
This makes it easier to start the server without importing modules.
"""
# filepath: /Users/trishit_debsharma/Documents/Code/Mechatronic/software_round2/drone_simulator/run_server.py
import asyncio
import sys
import os
import argparse
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_simulator.server import DroneSimulatorServer
from drone_simulator.admin_server import AdminServer
from drone_simulator.logging_config import get_logger

logger = get_logger("run_server")

def main():
    """Run the drone simulator server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Drone Simulator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind server (default: 8765)")
    parser.add_argument("--admin-port", type=int, default=8766, help="Port for admin server (default: 8766)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Log startup information
    logger.info(f"Starting Drone Simulator Server on {args.host}:{args.port}")
    logger.info(f"Admin server will run on {args.host}:{args.admin_port}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Create server instances
    main_server = DroneSimulatorServer(host=args.host, port=args.port)
    admin_server = AdminServer(host=args.host, port=args.admin_port, main_server=main_server)
    
    async def run_servers():
        """Run both servers together."""
        # Start both servers as tasks
        main_task = asyncio.create_task(main_server.start_server())
        admin_task = asyncio.create_task(admin_server.start_server())
        
        try:
            # Wait for both servers to finish (they run forever unless interrupted)
            await asyncio.gather(main_task, admin_task)
        except asyncio.CancelledError:
            logger.info("Server tasks cancelled")
    
    # Run the servers
    print(f"Starting Drone Simulator Server on ws://{args.host}:{args.port}")
    print(f"Admin server available on ws://{args.host}:{args.admin_port}")
    print("Press Ctrl+C to stop the server")
    
    start_time = time.time()
    
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        uptime = time.time() - start_time
        logger.info(f"Server stopped by user after running for {uptime:.1f} seconds")
        print(f"\nServer stopped after running for {uptime:.1f} seconds")
    except Exception as e:
        logger.critical(f"Server crashed: {e}", exc_info=True)
        print(f"\nServer crashed: {e}")

if __name__ == "__main__":
    main()