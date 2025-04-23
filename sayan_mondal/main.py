"""Example usage of drone simulator."""
import json
import time
from drone import DroneSimulator

def main():
    """Run the drone simulator example."""
    drone = DroneSimulator()
    
    # Example user input
    user_input = {
        "speed": 2,
        "altitude": 0,
        "movement": "fwd"
    }
    
    try:
        while True:
            try:
                telemetry = drone.update_telemetry(user_input)
                print(json.dumps(telemetry, indent=3))
            except ValueError as e:
                print(e)
                break
                
            # Add delay
            time.sleep()
            
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    main()