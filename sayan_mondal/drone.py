"""Drone simulator main class."""
from typing import Dict, Union, Any
from validators import validate_drone_input
from telemetry import TelemetryManager
from environment import EnvironmentSimulator
from logging_config import get_logger
import math

logger = get_logger("drone")

class DroneSimulator:
    """Simulates drone flight and telemetry."""
    
    def __init__(self, telemetry_file: str = 'telemetry.json'):
        """Initialize drone simulator."""
        logger.info(f"Initializing drone simulator with telemetry file: {telemetry_file}")
        self.telemetry_manager = TelemetryManager(telemetry_file)
        self.telemetry = self.telemetry_manager.get_telemetry()
        self.movement_speed = 5
        self.max_x_position = 100000
        self.user_input = None
        self.iteration_count = 0
        self.total_distance = 0
        self.crashed = False
        self.crash_reason = None
        self.drone_id = telemetry_file.split("_")[-1].split(".")[0] if "_" in telemetry_file else "main"
        logger.debug(f"Drone {self.drone_id} initialized with telemetry: {self.telemetry}")

    def validate_input(self) -> Union[bool, str]:
        """Validate user input."""
        logger.debug(f"Validating input: {self.user_input}")
        result = validate_drone_input(self.user_input)
        if result is not True:
            logger.warning(f"Invalid input: {result}")
        return result

    def update_telemetry(self, user_input: Dict[str, Union[int, str]]) -> Dict:
        """Update drone telemetry based on user input."""
        logger.info(f"Drone {self.drone_id} - Updating telemetry with input: {user_input}")
        
        # If drone is already crashed, don't process new commands
        if self.crashed:
            error_msg = f"Drone has crashed: {self.crash_reason}. Cannot accept new commands."
            logger.error(f"Drone {self.drone_id} - {error_msg}")
            raise ValueError(error_msg)
            
        self.user_input = user_input
        validation_result = self.validate_input()
        if validation_result is not True:
            error_msg = f"Invalid input data: {validation_result}"
            logger.error(f"Drone {self.drone_id} - {error_msg}")
            raise ValueError(error_msg)
        
        # Store previous values for comparison
        prev_x_position = self.telemetry["x_position"]
        prev_y_position = self.telemetry["y_position"]
        prev_sensor_status = self.telemetry.get("sensor_status", "GREEN")
        
        try:
            self._update_position()
            logger.debug(f"Drone {self.drone_id} - Position updated: "
                         f"X: {prev_x_position} -> {self.telemetry['x_position']}, "
                         f"Y: {prev_y_position} -> {self.telemetry['y_position']}")
            
            prev_battery = self.telemetry["battery"]
            self._update_battery()
            logger.debug(f"Drone {self.drone_id} - Battery updated: "
                         f"{prev_battery:.1f}% -> {self.telemetry['battery']:.1f}%")
            
            self._update_environmental_conditions()
            logger.debug(f"Drone {self.drone_id} - Environmental conditions updated: "
                         f"Wind: {self.telemetry['wind_speed']}, Dust: {self.telemetry['dust_level']}")
            
            # Check for sensor status changes and issue warnings
            current_sensor_status = self.telemetry.get("sensor_status", "GREEN")
            current_altitude = self.telemetry.get("y_position", 0)
            
            if current_sensor_status == "RED" and prev_sensor_status != "RED":
                # Sensor status just changed to RED
                logger.warning(f"Drone {self.drone_id} - CRITICAL: Sensor status changed to RED. "
                             f"Current altitude: {current_altitude}. Must descend below altitude 3.")
                
                if current_altitude > 3:
                    logger.warning(f"Drone {self.drone_id} - EMERGENCY: Immediate descent required. "
                                 f"Current altitude of {current_altitude} exceeds safe limit of 3 for RED status.")
            
            elif current_sensor_status == "YELLOW" and prev_sensor_status != "YELLOW":
                # Sensor status just changed to YELLOW
                logger.warning(f"Drone {self.drone_id} - WARNING: Sensor status changed to YELLOW. "
                             f"Current altitude: {current_altitude}. Maximum safe altitude is 1000.")
                
                if current_altitude > 800:  # Warn if approaching limit
                    logger.warning(f"Drone {self.drone_id} - CAUTION: Approaching altitude limit. "
                                 f"Current altitude {current_altitude} approaching limit of 1000 for YELLOW status.")
            
            # Perform crash checks after all updates
            self._check_drone_crash()
            
            # Calculate distance traveled
            distance = abs(self.telemetry["x_position"] - prev_x_position)
            self.total_distance += distance
            
            # Count iterations when both speed and y_position are non-zero
            if user_input.get("speed", 0) != 0 and self.telemetry.get("y_position", 0) != 0:
                self.iteration_count += 1
                logger.info(f"Drone {self.drone_id} - Flight iteration {self.iteration_count}: "
                           f"Distance traveled: +{distance:.1f}, Total: {self.total_distance:.1f}")
            
            # Save updated telemetry
            self.telemetry_manager.update_telemetry(self.telemetry)
            
            return self.telemetry
            
        except ValueError as e:
            # Mark the drone as crashed and save the crash reason
            self.crashed = True
            self.crash_reason = str(e)
            logger.critical(f"Drone {self.drone_id} - CRASHED: {self.crash_reason}")
            
            # Save the final state
            self.telemetry_manager.update_telemetry(self.telemetry)
            
            # Re-raise the exception
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get drone performance metrics."""
        metrics = {
            "iterations": self.iteration_count,
            "total_distance": self.total_distance
        }
        
        if self.crashed:
            metrics["crashed"] = True
            metrics["crash_reason"] = self.crash_reason
        
        logger.debug(f"Drone {self.drone_id} - Metrics retrieved: {metrics}")    
        return metrics
    
    def reset(self) -> None:
        """Reset the drone to its initial state."""
        logger.info(f"Drone {self.drone_id} - Resetting to initial state")
        self.telemetry = {
            "x_position": 0,
            "y_position": 0, 
            "battery": 100,
            "gyroscope": [0.0, 0.0, 0.0],
            "wind_speed": 0,
            "dust_level": 0,
            "sensor_status": "GREEN"
        }
        self.telemetry_manager.update_telemetry(self.telemetry)
        self.iteration_count = 0
        self.total_distance = 0
        self.crashed = False
        self.crash_reason = None
        logger.info(f"Drone {self.drone_id} - Reset complete")
    
    def _update_position(self) -> None:
        """Update drone position based on user input."""
        speed = self.user_input.get("speed", 0)
        altitude_change = self.user_input.get("altitude", 0)
        movement = self.user_input.get("movement", None)
        
        # Update position based on movement
        if movement == "fwd":
            self.telemetry["x_position"] = self.telemetry["x_position"] + speed
        elif movement == "rev":
            self.telemetry["x_position"] = self.telemetry["x_position"] - speed

        # Update altitude
        if abs(altitude_change) > 0:
            self.telemetry["y_position"] = self.telemetry["y_position"] + altitude_change
    
    def _update_battery(self) -> None:
        """
        Update battery level based on drone operations.
        
        Uses a continuous function to model battery drain based on altitude:
        - Lower altitudes have higher air resistance, causing more battery drain
        - Higher altitudes have less air resistance, causing less battery drain
        """
        speed = self.user_input.get("speed", 0)
        altitude_change = self.user_input.get("altitude", 0)
        current_altitude = self.telemetry["y_position"]
        
        # Base drain calculation from speed and altitude change
        base_drain = (0.5 * speed + abs(altitude_change) * 0.005)
        
        # Calculate altitude factor using a continuous function
        # We'll use an exponential decay function: f(y) = a + (b-a) * e^(-c*y)
        # where:
        # - y is the altitude
        # - a is the minimum multiplier (at infinite altitude)
        # - b is the maximum multiplier (at ground level)
        # - c controls the rate of decay
        
        min_multiplier = 0.6    # Minimum drain factor at very high altitude
        max_multiplier = 1.8    # Maximum drain factor at ground level
        decay_rate = 0.03       # Controls how quickly the factor decreases with altitude
        
        # Calculate altitude factor
        altitude_factor = min_multiplier + (max_multiplier - min_multiplier) * math.exp(-decay_rate * current_altitude)
        
        # Apply the altitude factor to the base drain
        total_drain = base_drain * altitude_factor
        
        # Apply a minimum drain (even when hovering)
        minimum_drain = 0.1
        drain_amount = max(total_drain, minimum_drain)
        
        # Update battery level
        prev_battery = self.telemetry["battery"]
        self.telemetry["battery"] = max(0, prev_battery - drain_amount)
        
        # Log detailed battery information
        logger.debug(f"Drone {self.drone_id} - Battery drain details: "
                    f"Base drain: {base_drain:.2f}%, "
                    f"Altitude: {current_altitude:.1f}, "
                    f"Altitude factor: {altitude_factor:.2f}x, "
                    f"Total drain: {drain_amount:.2f}%, "
                    f"Battery: {prev_battery:.1f}% -> {self.telemetry['battery']:.1f}%")
        
        if self.telemetry["battery"] < 20:
            logger.warning(f"Drone {self.drone_id} - Low battery: {self.telemetry['battery']:.1f}%")

    def _update_environmental_conditions(self) -> None:
        """Update environmental conditions affecting the drone."""
        self.telemetry = EnvironmentSimulator.simulate_environmental_conditions(
            self.telemetry, self.user_input
        )

    def _check_drone_crash(self) -> None:
        """
        Check if drone has crashed based on current telemetry.
        
        Checks for various crash conditions:
        1. Battery depletion
        2. Negative altitude (ground collision)
        3. Exceeding maximum x position
        4. Sensor status safety violations
           - RED status: Must maintain altitude below 3
           - YELLOW status: Must maintain altitude below 1000
        """
        if self.telemetry["battery"] <= 0:
            self.telemetry["battery"] = 0
            raise ValueError("Drone has crashed due to battery depletion.")
            
        if self.telemetry["y_position"] < 0:
            self.telemetry["y_position"] = 0  # Reset to ground level
            raise ValueError("Drone has crashed due to negative altitude.")
            
        if abs(self.telemetry["x_position"]) > self.max_x_position:
            raise ValueError("Drone has crashed due to exceeding max x position.")
        
        # Check sensor status safety violations
        sensor_status = self.telemetry.get("sensor_status", "GREEN")
        current_altitude = self.telemetry.get("y_position", 0)
        
        if sensor_status == "RED" and current_altitude > 3:
            # RED status: Must land or stay near ground (altitude < 3)
            logger.critical(f"Drone {self.drone_id} - Safety violation: Altitude {current_altitude} exceeds "
                         f"maximum safe altitude of 3 for RED sensor status")
            raise ValueError("Drone has crashed due to unsafe altitude with RED sensor status. Maximum safe altitude is 3.")
        
        elif sensor_status == "YELLOW" and current_altitude > 1000:
            # YELLOW status: Must stay below altitude 1000
            logger.critical(f"Drone {self.drone_id} - Safety violation: Altitude {current_altitude} exceeds "
                         f"maximum safe altitude of 1000 for YELLOW sensor status")
            raise ValueError("Drone has crashed due to unsafe altitude with YELLOW sensor status. Maximum safe altitude is 1000.")
        
        # For GREEN status, no altitude restrictions