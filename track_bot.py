"""
track_bot.py
Handles sensor data, fitness calculation, and state tracking.
"""

import math
import time
from javascript import require

Vec3 = require('vec3')

class BotTracker:
    def __init__(self, bot, target_pos, ray_max_dist=10):
        self.bot = bot
        # Convert target dictionary to Vec3
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        
        self.is_active = False 
        self.reached_goal = False
        self.tick_count = 0
        self.start_time = 0.0
        self.run_duration = 0.0
        
        self.fitness_score = 0.0
        self.max_distance_to_goal = None # Furthest distance (start)
        self.min_distance_to_goal = None # Closest approach
        self.ray_max_dist = ray_max_dist

    def reset_for_run(self):
        """Resets all metrics for a new generation."""
        self.is_active = False # distinct from 'enabled', waits for TP to finish
        self.reached_goal = False
        self.tick_count = 0
        self.fitness_score = 0.0
        self.run_duration = 0.0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None
        
        # We do NOT set start_time here; we set it when the run officially begins
        # to account for teleport delay.

    def get_distance_to_goal(self):
        """Calculates 3D distance to the target position."""
        if not self.bot.entity: return 999.0
        try:
            return float(self.bot.entity.position.distanceTo(self.target_pos))
        except:
            return 999.0

    def get_direction_to_goal(self):
        """Returns a normalized vector pointing to the goal."""
        if not self.bot.entity: return [0, 0, 0]
        
        pos = self.bot.entity.position
        dx = self.target_pos.x - pos.x
        dy = self.target_pos.y - pos.y
        dz = self.target_pos.z - pos.z
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 0.001: return [0.0, 0.0, 0.0]
        
        return [dx/dist, dy/dist, dz/dist]

    def _cast_ray(self, yaw_offset, pitch_offset):
        """
        Casts a ray from the bot's eye position.
        Returns 0.0 (far/nothing) to 1.0 (immediate obstacle).
        """
        if not self.bot.entity: return 0.0
        
        try:
            yaw = self.bot.entity.yaw + yaw_offset
            pitch = self.bot.entity.pitch + pitch_offset
            
            # Start at eye level
            pos = self.bot.entity.position.offset(0, 1.62, 0)
            
            # Calculate direction vector
            vx = -math.sin(yaw) * math.cos(pitch)
            vy = math.sin(pitch)
            vz = -math.cos(yaw) * math.cos(pitch)
            
            # Step through the ray
            for dist in range(1, self.ray_max_dist + 1):
                # Calculate check position
                check_pos = pos.offset(vx * dist, vy * dist, vz * dist)
                
                # Check for block
                block = self.bot.blockAt(check_pos)
                if block and block.boundingBox == 'block':
                    # Linear falloff: 1.0 at distance 0, 0.0 at max_dist
                    return 1.0 - (dist / self.ray_max_dist)
                    
        except Exception:
            return 0.0
            
        return 0.0 

    def get_sensor_input_vector(self):
        """
        Collects all sensor data for the neural network.
        Returns a dictionary containing flattened input arrays.
        """
        if not self.bot.entity or not self.is_active:
            # Return empty/zero inputs if entity doesn't exist yet or is inactive
            return {
                'rays': [0] * 21,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'orientation': [0, 0],
                'goal_direction': [0, 0, 0],
                'on_ground': 0
            }
            
        # 1. RAYCASTING (Vision)
        rays = []
        # Eye level (Horizontal awareness)
        for i in range(8): 
            rays.append(self._cast_ray(i * (math.pi/4), 0))
        # Downward angled (Platform awareness)
        for i in range(8): 
            rays.append(self._cast_ray(i * (math.pi/4), -0.5))
        # Jump trajectory (Landing awareness)
        for angle in [0, math.pi/4, -math.pi/4]: 
            rays.append(self._cast_ray(angle, -1.0))
        # Special (Ceiling + Deep Void)
        rays.append(self._cast_ray(0, -1.5))
        rays.append(self._cast_ray(0, 0.5))
        
        # 2. STATE VECTORS
        pos = self.bot.entity.position
        vel = self.bot.entity.velocity
        
        # Normalize inputs reasonably
        norm_pos = [
            (pos.x - self.target_pos.x) / 20.0,
            (pos.y - self.target_pos.y) / 10.0,
            (pos.z - self.target_pos.z) / 20.0
        ]
        
        norm_vel = [
            max(-1, min(1, vel.x)),
            max(-1, min(1, vel.y)),
            max(-1, min(1, vel.z))
        ]
        
        norm_orient = [
            self.bot.entity.yaw / math.pi,
            self.bot.entity.pitch / (math.pi/2)
        ]

        return {
            'rays': rays,
            'position': norm_pos,
            'velocity': norm_vel,
            'orientation': norm_orient,
            'goal_direction': self.get_direction_to_goal(),
            'on_ground': 1 if self.bot.entity.onGround else 0
        }

    def check_goal_reached(self):
        """Checks if bot is within the goal bounding box."""
        if not self.bot.entity: return False
        
        pos = self.bot.entity.position
        
        # Define goal bounds (1.5 block radius horizontally, 2 blocks vertically)
        dx = abs(pos.x - self.target_pos.x)
        dy = abs(pos.y - self.target_pos.y)
        dz = abs(pos.z - self.target_pos.z)
        
        return dx < 1.5 and dz < 1.5 and dy < 2.0

    def update_fitness(self):
        """
        Calculates fitness score based on progress towards goal.
        Called once at the end of the run.
        """
        if not self.bot.entity and self.max_distance_to_goal is None:
            return 0.0
            
        current_dist = self.get_distance_to_goal()
        
        # Initialize baselines if this is the first update
        if self.max_distance_to_goal is None:
            self.max_distance_to_goal = current_dist
        if self.min_distance_to_goal is None:
            self.min_distance_to_goal = current_dist
            
        # Update closest approach
        if current_dist < self.min_distance_to_goal:
            self.min_distance_to_goal = current_dist
            
        # --- SCORING LOGIC ---
        
        # 1. Progress Score: Difference between start dist and closest approach
        # Multiplied by 10 to make 1 block = 10 points
        progress = max(0, self.max_distance_to_goal - self.min_distance_to_goal)
        score = progress * 10.0
        
        # 2. Survival Score: Small reward for staying alive (0.1 per tick)
        # Only counts if the bot was active!
        score += min(50.0, self.tick_count * 0.1)
        
        # 3. Completion Bonus
        if self.reached_goal:
            score += 500.0  # Big flat bonus
            
            # Speed Bonus: Reward faster times
            time_taken = time.time() - self.start_time
            score += max(0, (15.0 - time_taken) * 20.0)
            
        self.fitness_score = max(0, score)
        return self.fitness_score