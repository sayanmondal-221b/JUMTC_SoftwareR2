"""Telemetry management for drone simulator."""
import json
from typing import Dict, Any

class TelemetryManager:
    """Manages drone telemetry data."""
    
    def __init__(self, telemetry_file: str = 'telemetry.json'):
        """Initialize telemetry manager."""
        self.telemetry_file = telemetry_file
        self.telemetry = self._load_telemetry()
        
    def _load_telemetry(self) -> Dict[str, Any]:
        """Load telemetry data from file or create default."""
        initial_telemetry = {
            "x_position": 0,
            "y_position": 0, 
            "battery": 100,
            "gyroscope": [0.0, 0.0, 0.0],
            "wind_speed": 0,
            "dust_level": 0,
            "sensor_status": "GREEN"
        }
        
        try:
            with open(self.telemetry_file, 'r') as f:
                data = f.read()
                if data:  # Check if file is not empty
                    return json.load(f)
                else:
                    return initial_telemetry
        except (FileNotFoundError, json.JSONDecodeError):
            # Save initial telemetry if file doesn't exist
            self.save_telemetry(initial_telemetry)
            return initial_telemetry
    
    def save_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Save telemetry data to file."""
        with open(self.telemetry_file, 'w') as f:
            json.dump(telemetry, f)
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data."""
        return self.telemetry
    
    def update_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Update telemetry data and save to file."""
        self.telemetry = telemetry
        self.save_telemetry(telemetry)