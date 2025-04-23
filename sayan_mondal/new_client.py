
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import json
import asyncio
import websockets
import pygame
import sys
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Constants
STATE_DIM = 8  # x_pos, y_pos, battery, wind, dust, sensor_status, gyro_x, gyro_y
ACTION_DIM = 36  # All combinations of speed (0-5), altitude change (-1,0,1), movement (fwd,rev)
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
LEARNING_RATE = 0.002

# Pygame visualization constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
DRONE_COLOR = (0, 255, 0)
CRASH_COLOR = (255, 0, 0)
TRAIL_COLOR = (100, 100, 255)
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)

class DroneEnvWrapper:
    """Wrapper for the drone simulator to make it compatible with RL."""
    
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.connection_id = None
        self.state = None
        self.last_distance = 0
        self.episode_stats = {
            'distance': [],
            'duration': [],
            'steps': []
        }
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            response = await self.websocket.recv()
            data = json.loads(response)
            self.connection_id = data.get("connection_id")
            print(f"Connected to drone simulator with ID: {self.connection_id}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
        
    async def reset(self):
        """Reset the environment and return initial state."""
        if self.websocket is None:
            if not await self.connect():
                return np.zeros(STATE_DIM, dtype=np.float32)
            
        try:
            # Send neutral command to get initial state
            await self.websocket.send(json.dumps({"speed": 0, "altitude": 0, "movement": "fwd"}))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if "telemetry" not in data:
                print("No telemetry in response:", data)
                return np.zeros(STATE_DIM, dtype=np.float32)
                
            # Parse telemetry
            self.state = self._parse_telemetry(data["telemetry"])
            self.last_distance = data.get("metrics", {}).get("total_distance", 0)
            return self.state
            
        except Exception as e:
            print(f"Error in reset: {e}")
            # Try to reconnect if there's an error
            self.websocket = None
            return np.zeros(STATE_DIM, dtype=np.float32)

    async def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        if self.websocket is None:
            if not await self.connect():
                return np.zeros(STATE_DIM, dtype=np.float32), -10, True, {}

        try:
            # Convert action index to simulator command
            speed, altitude, movement = self._decode_action(action_idx)
            
            # Send command to simulator
            await self.websocket.send(json.dumps({
                "speed": speed,
                "altitude": altitude,
                "movement": movement
            }))
            
            # Get response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            # Handle crash cases
            if data.get("status") == "crashed":
                telemetry = data.get("final_telemetry", "")
                metrics = data.get("metrics", {})
                crash_reason = data.get("message", "Unknown reason")
                print(f"Drone crashed: {crash_reason}")
                
                # Parse final telemetry if available
                next_state = self._parse_telemetry(telemetry) if telemetry else np.zeros(STATE_DIM, dtype=np.float32)
                
                # Calculate final reward
                reward = self._calculate_reward(data, True)
                
                return next_state, reward, True, data
            
            # Normal operation
            telemetry = data.get("telemetry", "")
            if not telemetry:
                print("No telemetry in response:", data)
                return np.zeros(STATE_DIM, dtype=np.float32), -1, False, data
                
            next_state = self._parse_telemetry(telemetry)
            reward = self._calculate_reward(data, False)
            
            # Update state
            self.state = next_state
            
            return next_state, reward, False, data
            
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed, attempting to reconnect...")
            self.websocket = None
            return np.zeros(STATE_DIM, dtype=np.float32), -10, True, {}
            
        except json.JSONDecodeError:
            print("Invalid JSON response from server")
            return np.zeros(STATE_DIM, dtype=np.float32), -1, False, {}
            
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(STATE_DIM, dtype=np.float32), -1, False, {}


    def _parse_telemetry(self, telemetry_str: str) -> np.ndarray:
        """Parse telemetry string into state vector."""
        try:
            # Initialize default values
            x_pos = 0.0
            y_pos = 0.0
            battery = 100.0
            wind = 0.0
            dust = 0.0
            sensor_status = 0.0
            gyro_x = 0.0
            gyro_y = 0.0

            # Handle empty telemetry string
            if not telemetry_str or telemetry_str.isspace():
                return np.array([x_pos, y_pos, battery, wind, dust, sensor_status, gyro_x, gyro_y], dtype=np.float32)

            # Split the telemetry string into key-value pairs
            parts = [p.strip() for p in telemetry_str.split('-') if p.strip()]

            # Parse each component
            i = 0
            while i < len(parts):
                if parts[i] == 'X' and i+1 < len(parts):
                    try:
                        x_pos = float(parts[i+1])
                    except ValueError:
                        x_pos = 0.0
                    i += 2
                elif parts[i] == 'Y' and i+1 < len(parts):
                    try:
                        y_pos = float(parts[i+1])
                    except ValueError:
                        y_pos = 0.0
                    i += 2
                elif parts[i] == 'BAT' and i+1 < len(parts):
                    try:
                        battery = float(parts[i+1])
                    except ValueError:
                        battery = 100.0
                    i += 2
                elif parts[i] == 'GYR' and i+1 < len(parts):
                    gyro_str = parts[i+1].strip('[]')
                    gyro_parts = [g.strip() for g in gyro_str.split(',') if g.strip()]
                    if len(gyro_parts) >= 2:
                        try:
                            gyro_x = float(gyro_parts[0])
                            gyro_y = float(gyro_parts[1])
                        except ValueError:
                            gyro_x = 0.0
                            gyro_y = 0.0
                    i += 2
                elif parts[i] == 'WIND' and i+1 < len(parts):
                    try:
                        wind = float(parts[i+1])
                    except ValueError:
                        wind = 0.0
                    i += 2
                elif parts[i] == 'DUST' and i+1 < len(parts):
                    try:
                        dust = float(parts[i+1])
                    except ValueError:
                        dust = 0.0
                    i += 2
                elif parts[i] == 'SENS' and i+1 < len(parts):
                    sensor_map = {"GREEN": 0, "YELLOW": 1, "RED": 2}
                    sensor_status = sensor_map.get(parts[i+1], 0)
                    i += 2
                else:
                    i += 1

            # Create normalized state vector with safety checks
            state = np.array([
                max(-1.0, min(1.0, x_pos / 100000)),  # Clamped x position
                max(0.0, min(1.0, y_pos / 1000)),      # Clamped y position
                max(0.0, min(1.0, battery / 100)),     # Clamped battery
                max(0.0, min(1.0, wind / 100)),        # Clamped wind
                max(0.0, min(1.0, dust / 100)),        # Clamped dust
                max(0.0, min(1.0, sensor_status / 2)), # Clamped sensor status
                max(-1.0, min(1.0, gyro_x / 45)),     # Clamped gyro x
                max(-1.0, min(1.0, gyro_y / 45))       # Clamped gyro y
            ], dtype=np.float32)

            return state

        except Exception as e:
            print(f"Critical error parsing telemetry: {e}")
            print(f"Telemetry string: {telemetry_str}")
            # Return a safe default state
            return np.array([0, 0.5, 1.0, 0, 0, 0, 0, 0], dtype=np.float32)

        
    def _decode_action(self, action_idx: int) -> Tuple[int, int, str]:
        """Convert action index to (speed, altitude, movement) tuple."""
        # Action space breakdown:
        # 6 speeds (0-5) × 3 altitude changes (-1,0,1) × 2 movements (fwd,rev) = 36 total actions
        
        # Simplified to 18 actions:
        # speed (0-5), altitude change (-1,0,1), movement (fwd,rev)
        # We'll alternate between fwd and rev for each speed/altitude combo
        
        speed = (action_idx // 3) % 6
        altitude = (action_idx % 3) - 1  # -1, 0, or 1
        movement = "fwd" if (action_idx // 36) % 2 == 0 else "rev"
        
        return speed, altitude, movement
        
    def _calculate_reward(self, data: Dict, done: bool, current_state: np.ndarray = None) -> float:
        """Calculate reward for current step.
        
        Args:
            data: Dictionary containing telemetry and metrics
            done: Whether the episode is complete
            current_state: Current state vector (optional)
        
        Returns:
            float: Calculated reward
        """
        reward = 0
        
        # Get metrics safely
        metrics = data.get("metrics", {})
        current_distance = metrics.get("total_distance", 0)
        
        # Distance reward (keep original scaling)
        distance_delta = current_distance - self.last_distance
        self.last_distance = current_distance
        reward += distance_delta * 20
        
        # Battery penalty (keep original scaling)
        if "telemetry" in data:
            try:
                battery_part = data["telemetry"].split('-BAT-')[1].split('-')[0]
                battery = float(battery_part)
                reward -= (100 - battery) * 0.01
            except (IndexError, ValueError):
                pass
        
        # Altitude reward (new addition)
        if current_state is not None and len(current_state) > 1:
            current_altitude = current_state[1] * 1000  # Convert normalized to meters
            sensor_status = data.get("sensor_status", 0)
            
            # Small constant reward for maintaining safe altitude
            if current_altitude > 10 and sensor_status < 1.0:
                reward += 0.1  # Small bonus for being above minimum safe altitude
                
            # Penalize being too low in danger zones
            if sensor_status > 1.5 and current_altitude < 15:
                reward -= 0.5
        
        # Crash penalty (keep original)
        if done:
            reward -= 100
            self.episode_stats['distance'].append(current_distance)
            self.episode_stats['duration'].append(metrics.get("flight_time", 0))
            self.episode_stats['steps'].append(metrics.get("iterations", 0))
        
        # Small step penalty (keep original)
        reward -= 0.05
        
        return reward
    
    def save_stats(self, filename: str):
        folder = os.path.dirname(filename)
        if folder:  # Only create folder if path contains a directory
            os.makedirs(folder, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.episode_stats, f, indent=4)

class DQN(nn.Module):
    """Deep Q-Network for drone control."""
    
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )
        
    def __len__(self):
        return len(self.buffer)

class DroneVisualizer:
    """Pygame visualization for drone training."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Drone RL Training")
        self.font = pygame.font.SysFont('Arial', 16)
        self.bold_font = pygame.font.SysFont('Arial', 18, bold=True)
        self.clock = pygame.time.Clock()
        self.trail = []
        self.max_trail_length = 200
        self.max_distance = 0
        self.best_episode = 0
        self.episode_start_time = 0

    
    def _decode_action(self, action_idx):

        speed = (action_idx // 3) % 6
        altitude = (action_idx % 3) - 1
        movement = "fwd" if (action_idx // 36) % 2 == 0 else "rev"
        return speed, altitude, movement
        
    def update(self, state, action, reward, episode, epsilon, total_distance):
        """Update visualization with current state."""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Update max distance tracking
        if total_distance > self.max_distance:
            self.max_distance = total_distance
            self.best_episode = episode

        self.screen.fill(BACKGROUND_COLOR)
        if not hasattr(self, "drone_pos"):
            self.drone_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]

        # Scale normalized state deltas
        dx = int(state[0] * 1000)  # x movement
        dy = int(state[1] * 1000)  # y movement (screen inverted)

        # Update position
        self.drone_pos[0] += dx
        self.drone_pos[1] -= dy

        # Keep it on screen
        self.drone_pos[0] = max(20, min(SCREEN_WIDTH - 20, self.drone_pos[0]))
        self.drone_pos[1] = max(20, min(SCREEN_HEIGHT - 20, self.drone_pos[1]))

        x_pos, y_pos = self.drone_pos

        
        # Add to trail (only when moving)
        if len(self.trail) == 0 or abs(x_pos - self.trail[-1][0]) > 5 or abs(y_pos - self.trail[-1][1]) > 5:
            self.trail.append((x_pos, y_pos))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
            
        # Draw trail with gradient (recent = brighter)
        for i in range(1, len(self.trail)):
            alpha = min(255, 50 + 205 * i / len(self.trail))
            color = (*TRAIL_COLOR[:3], alpha)
            if len(self.trail[i]) > 2:  # If using RGBA
                pygame.draw.line(self.screen, color, self.trail[i-1], self.trail[i], 3)
            else:  # RGB
                pygame.draw.line(self.screen, TRAIL_COLOR, self.trail[i-1], self.trail[i], 2)
        
        # Draw drone with orientation indicator
        pygame.draw.circle(self.screen, DRONE_COLOR, (x_pos, y_pos), 10)
        # Add orientation indicator (triangle pointing in movement direction)
        action_desc = self._action_to_string(action)
        if "fwd" in action_desc:
            points = [(x_pos, y_pos-15), (x_pos-10, y_pos+10), (x_pos+10, y_pos+10)]
        else:  # reverse
            points = [(x_pos, y_pos+15), (x_pos-10, y_pos-10), (x_pos+10, y_pos-10)]
        pygame.draw.polygon(self.screen, (255, 255, 0), points)
        
        # Display info panel
        self._draw_info_panel(state, action, reward, episode, epsilon, total_distance)
        
        pygame.display.flip()
        self.clock.tick(30)
        
    def _draw_info_panel(self, state, action, reward, episode, epsilon, total_distance):
        panel_rect = pygame.Rect(10, 10, 300, 220)  # Larger panel
        pygame.draw.rect(self.screen, (30, 30, 50), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 150), panel_rect, 2, border_radius=8)
        
        # Add more metrics
        y_offset = 15
        self._draw_text(f"Episode: {episode}", 20, y_offset, self.bold_font)
        self._draw_text(f"Epsilon: {epsilon:.3f}", 20, y_offset + 25)
        self._draw_text(f"Reward: {reward:.2f}", 20, y_offset + 45)
        self._draw_text(f"Distance: {total_distance:.1f}m", 20, y_offset + 70, self.bold_font)
        self._draw_text(f"Max Distance: {self.max_distance:.1f}m", 20, y_offset + 90)
        self._draw_text(f"(Ep. {self.best_episode})", 200, y_offset + 85)
        self._draw_text(f"Battery: {state[2]*100:.1f}%", 20, y_offset + 115)
        self._draw_text(f"Wind: {state[3]*100:.1f}%", 20, y_offset + 130)
        self._draw_text(f"Dust: {state[4]*100:.1f}%", 20, y_offset + 150)
        
        # Sensor status with color coding
        sensor_status = ["GREEN", "YELLOW", "RED"][min(2, int(state[5] * 2))]
        color = (0, 255, 0) if sensor_status == "GREEN" else (255, 255, 0) if sensor_status == "YELLOW" else (255, 0, 0)
        self._draw_text(f"Sensor: {sensor_status}", 20, y_offset + 170, color=color)
        
        self._draw_text(f"Action: {self._action_to_string(action)}", 20, y_offset + 195)
        
    def _draw_text(self, text, x, y, font=None, color=TEXT_COLOR):
        """Helper method to draw text."""
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
        
    def _action_to_string(self, action_idx):
        """Convert action index to human-readable string."""
        speed, altitude, movement = self._decode_action(action_idx)
        altitude_desc = {
            -1: "Descend",
            0: "Level",
            1: "Climb"
        }.get(altitude, "Unknown")
        return f"Spd {speed} | {altitude_desc} | {movement.upper()}"
        
    def close(self):
        pygame.quit()

class DroneRLAgent:
    """Reinforcement learning agent for drone control."""
    
    def __init__(self):
        self.env = DroneEnvWrapper()
        self.policy_net = DQN(STATE_DIM, ACTION_DIM)
        self.target_net = DQN(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.episode_rewards = []
        self.visualizer = DroneVisualizer()

    def _decode_action(self, action_idx: int) -> Tuple[int, int, str]:
        """Convert action index to (speed, altitude, movement) tuple."""
        # Action space breakdown:
        # 6 speeds (0-5) × 3 altitude changes (-1,0,1) × 2 movements (fwd,rev) = 36 total actions
        
        speed = (action_idx // 3) % 6
        altitude = (action_idx % 3) - 1  # -1, 0, or 1
        movement = "fwd" if (action_idx // 36) % 2 == 0 else "rev"
        return speed, altitude, movement

    def select_action(self, state):
        # Current altitude from normalized state
        current_alt = state[1] * 1000  
        sensor_status = state[5] * 2  # 0=GREEN, 1=YELLOW, 2=RED
        
        # Get random or policy action
        if random.random() < self.epsilon:
            action = random.randint(0, ACTION_DIM - 1)
        else:
            with torch.no_grad():
                action = self.policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        
        # Decode and validate action
        speed, altitude, movement = self._decode_action(action)
        
        # Safety overrides
        if current_alt < 5 and altitude < 0:  # Near ground
            altitude = 0  # Prevent descending further
        elif sensor_status == 2 and current_alt > 2:  # RED zone
            altitude = -1  # Force descent
        elif sensor_status == 1 and current_alt > 900:  # YELLOW zone
            altitude = max(-1, altitude)  # Limit ascent
            
        # Re-encode safe action
        safe_action = (speed * 3) + (altitude + 1)
        return min(safe_action, ACTION_DIM-1)

    async def train(self, num_episodes=1000):
        """Train the agent."""
        if not await self.env.connect():
            print("Failed to connect to simulator")
            return
            
        for episode in range(num_episodes):
            try:
                state = await self.env.reset()
                if state is None:
                    print("Reset failed, reconnecting...")
                    if not await self.env.connect():
                        continue
                    state = await self.env.reset()
                
                total_reward = 0
                done = False
                
                while not done:
                    # Select action with safety checks
                    action = self.select_action(state)
                    
                    # Take action
                    next_state, reward, done, info = await self.env.step(action)
                    total_reward += reward
                    
                    # Store transition
                    self.memory.push(state, action, reward, next_state, done)
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update(state, action, reward, episode, 
                                             self.epsilon, info.get("metrics", {}).get("total_distance", 0))
                    
                    # Move to next state
                    state = next_state
                    
                    # Optimize model
                    if len(self.memory) > BATCH_SIZE:
                        self.optimize_model()
                        
                    # Check for pygame events
                    if self.visualizer:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.visualizer.close()
                                return
                                
                # Update target network
                if episode % TARGET_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                # Decay epsilon
                self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
                
                # Log episode stats
                self.episode_rewards.append(total_reward)
                print(f"Episode {episode}: Reward={total_reward:.1f}, Epsilon={self.epsilon:.2f}, "
                      f"Distance={info.get('metrics', {}).get('total_distance', 0)}")
                # In your training loop:
                self.visualizer.update(state, action, reward, episode, 
                self.epsilon, info.get("metrics", {}).get("total_distance", 0))

            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                self.env.websocket = None  # Force reconnect
                continue
                
        #Save training stats
        log_dir = "telemetry_logs"
        os.makedirs(log_dir, exist_ok=True)
        if hasattr(self.env, 'save_stats'):
            filename = f"drone_training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.env.save_stats(os.path.join(log_dir, filename))

        if self.visualizer:
            self.visualizer.close()
        
        torch.save(self.policy_net.state_dict(), os.path.join(log_dir, "drone_policy_net.pth"))
        print("Trained model saved to drone_policy_net.pth")

            
    def optimize_model(self):
        """Perform one step of optimization."""
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        # Compute Q values
        current_q = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target_q = reward + (1 - done) * GAMMA * next_q
            
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

async def main():
    """Main function to run RL training."""
    agent = DroneRLAgent()
    try:
        await agent.train(num_episodes=1000)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save model
        torch.save(agent.policy_net.state_dict(), "drone_policy_net.pth")
        print("Model saved to drone_policy_net.pth")

if __name__ == "__main__":
    asyncio.run(main())