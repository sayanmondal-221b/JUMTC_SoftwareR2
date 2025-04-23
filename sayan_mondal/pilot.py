import asyncio
import torch
import numpy as np
import json
import logging
from datetime import datetime
from new_client import DQN
from environment import EnvironmentSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DronePilot")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "telemetry_logs/drone_policy_net.pth"

class DistanceOptimizedPilot:
    def __init__(self, model_path):
        self.state_dim = 8
        self.action_dim = 36
        self.model = DQN(self.state_dim, self.action_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        # Initialize telemetry for maximum distance
        self.telemetry = {
            "x_position": 0.0,
            "y_position": 20.0,  # Start high to avoid early dust
            "battery": 100.0,
            "wind_speed": 10.0,
            "dust_level": 5.0,
            "sensor_status": "GREEN",
            "gyroscope": [0.0, 0.0],
        }
        
        self.flight_log = {
            "start_time": datetime.now().isoformat(),
            "target_distance": 90.0,
            "flight_data": [],
            "performance_metrics": {
                "model_actions": 0,
                "overridden_actions": 0,
                "max_speed": 0,
                "avg_speed": 0,
            }
        }

    def build_state(self, telemetry):
        """Normalize state for the DQN model."""
        return np.array([
            telemetry["x_position"] / 100000,
            telemetry["y_position"] / 1000,
            telemetry["battery"] / 100,
            telemetry["wind_speed"] / 100,
            telemetry["dust_level"] / 100,
            0.0 if telemetry["sensor_status"] == "GREEN" else 0.5 if telemetry["sensor_status"] == "YELLOW" else 1.0,
            telemetry["gyroscope"][0] / 45,
            telemetry["gyroscope"][1] / 45,
        ], dtype=np.float32)

    def select_action(self, state, telemetry):
        """Use the model but enforce minimum speeds in optimal conditions."""
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        action = self.model(state_tensor).argmax().item()
        
        # Parse action components
        speed = (action // 6) % 6
        altitude = (action // 2) % 3 - 1
        movement = "fwd" if action % 2 == 0 else "rev"
        
        # Override only if the model chooses too slow a speed in safe conditions
        if (telemetry["sensor_status"] == "GREEN" 
            and telemetry["dust_level"] < 30 
            and telemetry["wind_speed"] < 30
            and speed < 4):  # Model chose speed <4
            self.flight_log["performance_metrics"]["overridden_actions"] += 1
            return 28  # Override to speed=4, altitude=+1, forward
        
        # Emergency overrides (critical hazards)
        if telemetry["sensor_status"] == "RED":
            return 24  # speed=4, altitude=+1 (emergency climb)
        
        self.flight_log["performance_metrics"]["model_actions"] += 1
        return action

    def _update_telemetry(self, action_idx):
        """Update position and battery with optimized physics."""
        speed = (action_idx // 6) % 6
        altitude_change = (action_idx // 2) % 3 - 1
        
        # Distance: Quadratic scaling (speed=5 gives 5² * 0.1 = 2.5m/step)
        distance_gain = (speed ** 2) * 0.1
        
        # Update position (force forward movement)
        self.telemetry["x_position"] += distance_gain
        
        # Update altitude (climb faster to avoid dust)
        self.telemetry["y_position"] = max(10.0, min(40.0, 
            self.telemetry["y_position"] + altitude_change * 1.0))
        
        # Battery: More efficient at mid-high speeds
        self.telemetry["battery"] -= (speed * 0.05) + (abs(altitude_change) * 0.02)
        
        # Track performance
        if speed > self.flight_log["performance_metrics"]["max_speed"]:
            self.flight_log["performance_metrics"]["max_speed"] = speed
            
        return distance_gain

    async def fly(self):
        step = 0
        total_distance = 0
        logger.info(f"Starting flight - Target: {self.flight_log['target_distance']}m")
        
        while (total_distance < self.flight_log['target_distance'] 
               and self.telemetry["battery"] > 2
               and step < 400):
            
            try:
                state = self.build_state(self.telemetry)
                action_idx = self.select_action(state, self.telemetry)
                distance_gain = self._update_telemetry(action_idx)
                total_distance += distance_gain
                
                # Simulate environment
                speed = (action_idx // 6) % 6
                altitude = (action_idx // 2) % 3 - 1
                self.telemetry = EnvironmentSimulator.simulate_environmental_conditions(
                    self.telemetry,
                    {"speed": speed, "altitude": altitude, "movement": "fwd"}
                )
                
                # Log step
                self.flight_log["flight_data"].append({
                    "step": step,
                    "position": round(self.telemetry["x_position"], 2),
                    "action": action_idx,
                    "speed": speed,
                    "battery": round(self.telemetry["battery"], 1),
                    "distance_gain": round(distance_gain, 2),
                })
                
                logger.info(
                    f"Step {step}: Δ+{distance_gain:.2f}m → {self.telemetry['x_position']:.2f}m "
                    f"(Total: {total_distance:.2f}m) | "
                    f"Speed: {speed} | Bat: {self.telemetry['battery']:.1f}%"
                )
                
                step += 1

            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")
                break

        # Finalize logs
        self.flight_log["performance_metrics"]["avg_speed"] = total_distance / step if step > 0 else 0
        self.flight_log["final_position"] = self.telemetry["x_position"]
        self.flight_log["battery_remaining"] = self.telemetry["battery"]
        self.flight_log["end_time"] = datetime.now().isoformat()
        
        logger.info("\n=== Flight Summary ===")
        logger.info(f"Final Position: {self.telemetry['x_position']:.2f}m")
        logger.info(f"Total Distance: {total_distance:.2f}m")
        logger.info(f"Steps Taken: {step}")
        logger.info(f"Model Actions: {self.flight_log['performance_metrics']['model_actions']}")
        logger.info(f"Overridden Actions: {self.flight_log['performance_metrics']['overridden_actions']}")
        
        with open("flight_log.json", "w") as f:
            json.dump(self.flight_log, f, indent=2)

if __name__ == "__main__":
    pilot = DistanceOptimizedPilot(MODEL_PATH)
    asyncio.run(pilot.fly())
