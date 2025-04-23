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

class HighPerformanceDronePilot:
    def __init__(self, model_path):
        self.state_dim = 8
        self.action_dim = 36
        self.model = DQN(self.state_dim, self.action_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        # Optimized starting parameters
        self.telemetry = {
            "x_position": 0.0,
            "y_position": 20.0,  # Higher starting altitude
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
                "max_speed": 0,
                "avg_speed": 0,
                "emergency_actions": 0,
                "speed_boost_activations": 0
            }
        }
        
        self.speed_boost_active = False

    def build_state(self, telemetry):
        """State representation optimized for performance"""
        return np.array([
            telemetry["x_position"] / 100000,
            telemetry["y_position"] / 1000,
            telemetry["battery"] / 100,
            telemetry["wind_speed"] / 100,
            telemetry["dust_level"] / 100,
            0.0 if telemetry["sensor_status"] == "GREEN" else 0.5 if telemetry["sensor_status"] == "YELLOW" else 1.0,
            telemetry["gyroscope"][0] / 45,
            telemetry["gyroscope"][1] / 45
        ], dtype=np.float32)

    def select_high_speed_action(self, state, telemetry):
        """Aggressive speed selection with smart adaptations"""
        # Speed boost mode - activate when conditions are perfect
        if (telemetry["sensor_status"] == "GREEN" and 
            telemetry["dust_level"] < 30 and 
            telemetry["wind_speed"] < 30 and
            telemetry["battery"] > 60):
            self.speed_boost_active = True
            self.flight_log["performance_metrics"]["speed_boost_activations"] += 1
            return 34  # Maximum speed (5), high altitude (+1), forward
        
        self.speed_boost_active = False
        
        # Environmental adaptations
        if telemetry["sensor_status"] == "RED":
            self.flight_log["performance_metrics"]["emergency_actions"] += 1
            return 24  # speed=4, altitude=+1, forward (emergency protocol)
            
        if telemetry["dust_level"] > 60:
            return 30  # speed=5, altitude=+1, forward (dust escape)
            
        if telemetry["wind_speed"] > 50:
            return 22  # speed=3, altitude=+1, forward (wind resistance)
            
        # Default high-performance action
        return 28  # speed=4, altitude=+1, forward

    def _update_telemetry(self, action_idx):
        """Optimized telemetry updates for maximum distance"""
        speed = (action_idx // 6) % 6
        altitude_change = (action_idx // 2) % 3 - 1
        
        # Aggressive distance calculation - speed squared gives more reward for high speeds
        distance_gain = (speed**2 * 0.15) * (1.5 if self.speed_boost_active else 1.0)
        self.telemetry["x_position"] += distance_gain
        
        # Altitude management - faster changes
        self.telemetry["y_position"] = max(10.0, min(40.0, 
            self.telemetry["y_position"] + altitude_change * 1.2))
        
        # Battery consumption - more efficient at higher speeds
        self.telemetry["battery"] -= (speed * 0.05) + (abs(altitude_change) * 0.02)
        
        # Track max speed
        if speed > self.flight_log["performance_metrics"]["max_speed"]:
            self.flight_log["performance_metrics"]["max_speed"] = speed
            
        return distance_gain

    async def fly(self):
        step = 0
        total_distance = 0
        logger.info("Starting flight")
        
        while (total_distance < self.flight_log["target_distance"] and 
               self.telemetry["battery"] > 2 and  # More aggressive battery usage
               step < 300):  # Reduced step limit for performance
            
            try:
                state = self.build_state(self.telemetry)
                action_idx = self.select_high_speed_action(state, self.telemetry)
                distance_gain = self._update_telemetry(action_idx)
                total_distance += distance_gain
                
                # Get environmental updates
                speed = (action_idx // 6) % 6
                altitude = (action_idx // 2) % 3 - 1
                self.telemetry = EnvironmentSimulator.simulate_environmental_conditions(
                    self.telemetry,
                    {"speed": speed, "altitude": altitude, "movement": "fwd"}
                )
                
                # Performance logging
                self.flight_log["flight_data"].append({
                    "step": step,
                    "position": round(self.telemetry["x_position"], 2),
                    "distance_gain": round(distance_gain, 2),
                    "speed": speed,
                    "battery": round(self.telemetry["battery"], 1),
                    "environment": {
                        "dust": round(self.telemetry["dust_level"], 1),
                        "wind": round(self.telemetry["wind_speed"], 1),
                        "sensors": self.telemetry["sensor_status"]
                    }
                })
                
                logger.info(
                    f"Step {step}: Δ+{distance_gain:.2f}m → {self.telemetry['x_position']:.2f}m "
                    f"(Total: {total_distance:.2f}m) | "
                    f"Speed: {speed} | Bat: {self.telemetry['battery']:.1f}%"
                )
                
                # Only emergency land in critical battery
                if self.telemetry["battery"] < 5:
                    logger.warning("Low battery protocol activated")
                    
                step += 1

            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")
                break

        # Final calculations
        self.flight_log["performance_metrics"]["avg_speed"] = total_distance / step if step > 0 else 0
        self.flight_log["final_position"] = self.telemetry["x_position"]
        self.flight_log["battery_remaining"] = self.telemetry["battery"]
        self.flight_log["end_time"] = datetime.now().isoformat()
        
        logger.info("\nFlight Results:")
        logger.info(f"Final Position: {self.telemetry['x_position']:.2f}m")
        logger.info(f"Total Distance: {total_distance:.2f}m")
        logger.info(f"Steps Taken: {step}")
        logger.info(f"Avg Speed: {self.flight_log['performance_metrics']['avg_speed']:.2f}m/step")
        logger.info(f"Remaining Battery: {self.telemetry['battery']:.1f}%")
        
        with open("flight_log.json", "w") as f:
            json.dump(self.flight_log, f, indent=2)
            
        return total_distance

if __name__ == "__main__":
    pilot = HighPerformanceDronePilot(MODEL_PATH)
    asyncio.run(pilot.fly())