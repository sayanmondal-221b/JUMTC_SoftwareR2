"""Input validation for drone simulator."""
from typing import Dict, Union, List, Tuple, Any

def validate_dict_input(input_data: Any) -> Union[bool, str]:
    """Validate if input is a dictionary."""
    if not isinstance(input_data, dict):
        return "Input must be a dictionary"
    return True

def validate_required_keys(input_data: Dict, required_keys: List[str]) -> Union[bool, str]:
    """Validate if all required keys are present."""
    for key in required_keys:
        if key not in input_data:
            return f"Missing required key: {key}"
    return True

def validate_speed(speed: Any) -> Union[bool, str]:
    """Validate speed value."""
    if not isinstance(speed, int):
        return f"'speed' must be an integer, got {type(speed).__name__}"
    if not (0 <= speed <= 5):
        return f"'speed' must be between 0 and 5, got {speed}"
    return True

def validate_altitude(altitude: Any) -> Union[bool, str]:
    """Validate altitude value."""
    if not isinstance(altitude, int):
        return f"'altitude' must be an integer, got {type(altitude).__name__}"
    return True

def validate_movement(movement: Any) -> Union[bool, str]:
    """Validate movement value."""
    if not isinstance(movement, str):
        return f"'movement' must be a string"
    if movement not in ["fwd", "rev"]:
        return f"'movement' must be one of ['fwd', 'rev'], got '{movement}'"
    return True

def validate_drone_input(input_data: Dict[str, Any]) -> Union[bool, str]:
    """Validate all drone input parameters."""
    # First check if it's a dictionary
    dict_validation = validate_dict_input(input_data)
    if dict_validation is not True:
        return dict_validation
    
    # Check required keys
    required_keys = ["speed", "altitude", "movement"]
    keys_validation = validate_required_keys(input_data, required_keys)
    if keys_validation is not True:
        return keys_validation
    
    # Validate individual fields
    speed_validation = validate_speed(input_data["speed"])
    if speed_validation is not True:
        return speed_validation
        
    altitude_validation = validate_altitude(input_data["altitude"])
    if altitude_validation is not True:
        return altitude_validation
        
    movement_validation = validate_movement(input_data["movement"])
    if movement_validation is not True:
        return movement_validation
    
    return True