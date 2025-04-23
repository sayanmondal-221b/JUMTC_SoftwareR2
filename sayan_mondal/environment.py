"""Environmental simulation for drone simulator."""
# filepath: /Users/trishit_debsharma/Documents/Code/Mechatronic/software_round2/drone_simulator/environment.py
import random
import math
from typing import Dict, Any, List
from logging_config import get_logger

logger = get_logger("environment")

class EnvironmentSimulator:
    """Simulates environmental conditions affecting the drone."""
    
    # Constants for gyroscope calculations
    MAX_WIND_TILT_DEGREES = 40.0  # Maximum tilt due to wind (degrees)
    MAX_MOVEMENT_TILT_DEGREES = 60.0  # Maximum safe tilt due to movement (degrees)
    CRITICAL_TILT_DEGREES = 45.0  # Tilt beyond which drone becomes unstable
    
    # Conversion between degrees and gyro values (-1 to 1)
    # 90 degrees = 1.0 in gyro value, so 45 degrees = 0.5
    DEGREES_TO_GYRO = 1.0 / 90.0  
    
    @staticmethod
    def calculate_gyroscope_values(telemetry: Dict[str, Any], user_input: Dict[str, Any]) -> List[float]:
        """
        Calculate realistic gyroscope values based on wind and drone movement.
        
        Args:
            telemetry: Current drone telemetry
            user_input: User command input
        
        Returns:
            List of gyroscope values [x, y, z]
        """
        # Extract relevant data
        wind_speed = telemetry.get("wind_speed", 0)
        drone_speed = user_input.get("speed", 0)
        movement_direction = user_input.get("movement", "fwd")
        altitude = telemetry.get("y_position", 0)
        
        # Base random instability (small fluctuations)
        # More stable at higher altitudes (less ground effect turbulence)
        altitude_stability_factor = min(1.0, altitude / 50.0)  # 0.0 at ground level, 1.0 at altitude 50+
        base_instability = 0.1 * (1 - altitude_stability_factor)
        
        # Generate base random fluctuations
        gyro_x = random.uniform(-base_instability, base_instability)
        gyro_y = random.uniform(-base_instability, base_instability)
        gyro_z = random.uniform(-base_instability, base_instability)
        
        # 1. Calculate wind effect on gyroscope
        # --------------------------------------
        # Wind direction (random but consistent for a given simulation step)
        wind_direction_rad = random.uniform(0, 2 * math.pi)
        
        # Wind effect scales with wind speed (0-100) and is limited to MAX_WIND_TILT_DEGREES
        wind_effect_magnitude = (wind_speed / 100.0) * EnvironmentSimulator.MAX_WIND_TILT_DEGREES
        
        # Convert to gyro values (-1 to 1 scale)
        wind_effect_gyro = wind_effect_magnitude * EnvironmentSimulator.DEGREES_TO_GYRO
        
        # Apply wind effect based on direction
        wind_effect_x = wind_effect_gyro * math.cos(wind_direction_rad)
        wind_effect_y = wind_effect_gyro * math.sin(wind_direction_rad)
        
        # 2. Calculate movement effect on gyroscope
        # -----------------------------------------
        # Movement effect increases with speed
        movement_effect_magnitude = (drone_speed / 5.0) * 20.0  # 5 is max speed, resulting in 20 degrees tilt
        
        # Direction affects which axis experiences tilt
        if movement_direction == "fwd":
            movement_effect_x = movement_effect_magnitude * EnvironmentSimulator.DEGREES_TO_GYRO
            movement_effect_y = 0
        elif movement_direction == "rev":
            movement_effect_x = -movement_effect_magnitude * EnvironmentSimulator.DEGREES_TO_GYRO
            movement_effect_y = 0
        else:
            movement_effect_x = 0
            movement_effect_y = 0
        
        # 3. Combine all effects
        # ----------------------
        final_gyro_x = gyro_x + wind_effect_x + movement_effect_x
        final_gyro_y = gyro_y + wind_effect_y + movement_effect_y
        final_gyro_z = gyro_z  # Z-axis rotation (yaw) less affected by these factors
        
        # Ensure values stay within -1 to 1 range
        final_gyro_x = max(-1.0, min(1.0, final_gyro_x))
        final_gyro_y = max(-1.0, min(1.0, final_gyro_y))
        final_gyro_z = max(-1.0, min(1.0, final_gyro_z))
        
        # Log the gyroscope calculation details
        logger.debug(f"Gyroscope calculation details: "
                   f"Wind: {wind_speed}% at {math.degrees(wind_direction_rad):.1f}°, "
                   f"Speed: {drone_speed}, Direction: {movement_direction}, "
                   f"Altitude: {altitude}, "
                   f"Wind effect: [{wind_effect_x:.3f}, {wind_effect_y:.3f}], "
                   f"Movement effect: [{movement_effect_x:.3f}, {movement_effect_y:.3f}], "
                   f"Final values: [{final_gyro_x:.3f}, {final_gyro_y:.3f}, {final_gyro_z:.3f}]")
        
        # Check if tilt exceeds critical value (45 degrees = 0.5 in gyro scale)
        # Only movement-induced tilt can cause a crash, not wind-induced tilt
        movement_tilt_magnitude = math.sqrt(movement_effect_x**2 + movement_effect_y**2)
        movement_tilt_degrees = movement_tilt_magnitude / EnvironmentSimulator.DEGREES_TO_GYRO
        
        if movement_tilt_degrees > EnvironmentSimulator.CRITICAL_TILT_DEGREES:
            logger.warning(f"Critical tilt detected: {movement_tilt_degrees:.1f}° exceeds {EnvironmentSimulator.CRITICAL_TILT_DEGREES}°")
            # Return special gyroscope values indicating crash
            # We'll handle the crash detection in the simulate_environmental_conditions method
            # But flag the values so they're identifiable
            return [1.0 if final_gyro_x > 0 else -1.0, 
                    1.0 if final_gyro_y > 0 else -1.0, 
                    1.0 if final_gyro_z > 0 else -1.0]
            
        return [final_gyro_x, final_gyro_y, final_gyro_z]
    
    @staticmethod
    def simulate_environmental_conditions(telemetry: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update telemetry with simulated environmental conditions.
        
        Args:
            telemetry: Current drone telemetry
            user_input: User command input, if available
        
        Returns:
            Updated telemetry with new environmental conditions
        """
        # Copy telemetry to avoid modifying the original
        updated_telemetry = telemetry.copy()
        
        # Random wind changes (with some persistence from previous state)
        prev_wind = telemetry.get("wind_speed", 0)
        wind_change = random.uniform(-15, 15)  # Wind can change by up to 15% in either direction
        updated_telemetry["wind_speed"] = max(0, min(100, prev_wind + wind_change))
        
        # Random dust changes (with some persistence)
        prev_dust = telemetry.get("dust_level", 0)
        dust_change = random.uniform(-10, 10)
        updated_telemetry["dust_level"] = max(0, min(100, prev_dust + dust_change))
        
        # Random events
        if random.random() < 0.3:  # 30% chance of dust storm
            logger.info("Dust storm event triggered")
            dust_severity = random.uniform(30, 70)  # How severe is the dust storm
            updated_telemetry["dust_level"] = min(100, updated_telemetry["dust_level"] + dust_severity)
            updated_telemetry["wind_speed"] = min(100, updated_telemetry["wind_speed"] + dust_severity)
        
        # Update gyroscope values based on conditions and movement
        if user_input:
            updated_telemetry["gyroscope"] = EnvironmentSimulator.calculate_gyroscope_values(
                telemetry, user_input
            )
            
            # Check for extreme gyroscope values indicating a crash
            gyro_magnitude = math.sqrt(sum(g**2 for g in updated_telemetry["gyroscope"]))
            if gyro_magnitude > 1.7:  # Close to maximum possible magnitude (√3)
                logger.critical(f"Drone has lost stability due to excessive tilt: {gyro_magnitude:.2f}")
                raise ValueError("Drone has crashed due to excessive tilt and loss of stability.")
        else:
            # If no user input (first initialization), use mild random values
            updated_telemetry["gyroscope"] = [
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1)
            ]
        
        # Update sensor status based on conditions
        if updated_telemetry["dust_level"] > 90 or updated_telemetry["wind_speed"] > 90:
            updated_telemetry["sensor_status"] = "RED"
            logger.warning(f"Sensor status RED - Dust: {updated_telemetry['dust_level']}%, "
                         f"Wind: {updated_telemetry['wind_speed']}%")
        elif updated_telemetry["dust_level"] > 60 or updated_telemetry["wind_speed"] > 60:
            updated_telemetry["sensor_status"] = "YELLOW"
            logger.info(f"Sensor status YELLOW - Dust: {updated_telemetry['dust_level']}%, "
                      f"Wind: {updated_telemetry['wind_speed']}%")
        else:
            updated_telemetry["sensor_status"] = "GREEN"
        
        return updated_telemetry