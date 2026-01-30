"""
Bot tracking and sensor data collection for neuromorphic parkour training.
Handles voxel mapping and vector inputs for neural processing.
"""

import math
from javascript import require

Vec3 = require('vec3')

class BotTracker:
    """Tracks bot state and provides sensory input vectors for neuromorphic processing."""
    
    def __init__(self, bot, target_pos, scan_radius=5):
        """
        Initialize bot tracker.
        
        Args:
            bot: Mineflayer bot instance
            target_pos: Vec3 or dict with 'x', 'y', 'z' coordinates
            scan_radius: The distance to scan blocks in all directions
        """
        self.bot = bot
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        
        # Performance tracking
        self.is_active = False  # Not active until start_new_run() is called
        self.is_alive = True
        self.reached_goal = False
        self.tick_count = 0
        self.max_distance_to_goal = None
        self.fitness_score = 0.0
        
        # Sensor configuration
        self.scan_radius = scan_radius # Configurable radius from main.py

    def reset_for_run(self):
        """Resets stats for a fresh parkour attempt."""
        self.is_active = True
        self.reached_goal = False
        self.tick_count = 0
        self.fitness_score = 0.0
        self.max_distance_to_goal = None
        
    def get_position_vector(self):
        """Returns bot's current position as [x, y, z]."""
        if not self.bot.entity: return [0, 0, 0]
        pos = self.bot.entity.position
        return [pos.x, pos.y, pos.z]
    
    def get_velocity_vector(self):
        """Returns bot's current velocity as [vx, vy, vz]."""
        if not self.bot.entity: return [0, 0, 0]
        vel = self.bot.entity.velocity
        return [vel.x, vel.y, vel.z]
    
    def get_yaw_pitch(self):
        """Returns bot's orientation as [yaw, pitch] in radians."""
        if not self.bot.entity: return [0, 0]
        return [self.bot.entity.yaw, self.bot.entity.pitch]
    
    def get_distance_to_goal(self):
        """Returns Euclidean distance to target."""
        if not self.bot.entity: return 999.0
        return self.bot.entity.position.distanceTo(self.target_pos)
    
    def get_direction_to_goal(self):
        """Returns normalized direction vector [dx, dy, dz] pointing to goal."""
        if not self.bot.entity: return [0, 0, 0]
        pos = self.bot.entity.position
        
        dx = self.target_pos.x - pos.x
        dy = self.target_pos.y - pos.y
        dz = self.target_pos.z - pos.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < 0.001:
            return [0.0, 0.0, 0.0]
        
        return [dx/distance, dy/distance, dz/distance]
    
    def get_voxel_grid(self):
        """
        Returns flattened voxel grid of surrounding blocks based on scan_radius.
        Used to feed spatial data into the neuromorphic network.
        """
        if not self.bot.entity: return []
        pos = self.bot.entity.position.floored()
        voxel_data = []
        
        # Scan in order: Y (bottom to top), Z (back to front), X (left to right)
        for dy in range(-self.scan_radius, self.scan_radius + 1):
            for dz in range(-self.scan_radius, self.scan_radius + 1):
                for dx in range(-self.scan_radius, self.scan_radius + 1):
                    target_pos = pos.offset(dx, dy, dz)
                    block = self.bot.blockAt(target_pos)
                    
                    if block is None:
                        voxel_data.append(-1)  # Unloaded chunk
                    elif block.name == 'air':
                        voxel_data.append(0)   # Air
                    elif 'water' in block.name or 'lava' in block.name:
                        voxel_data.append(2)   # Liquid
                    else:
                        voxel_data.append(1)   # Solid block
        
        return voxel_data
    
    def get_block_below(self):
        """Returns block type directly below bot for ground detection."""
        if not self.bot.entity: return -1
        pos = self.bot.entity.position.floored()
        block = self.bot.blockAt(pos.offset(0, -1, 0))
        
        if block is None: return -1
        elif block.name == 'air': return 0
        elif 'water' in block.name or 'lava' in block.name: return 2
        else: return 1
    
    def get_sensor_input_vector(self):
        """
        Compiles all sensory inputs into a single vector for the neuromorphic computer.
        Includes the 'is_sprinting' state necessary for momentum-based parkour.
        """
        is_sprinting = 0
        if self.bot.entity and hasattr(self.bot, 'controlState'):
            is_sprinting = 1 if self.bot.controlState.get('sprint', False) else 0
        
        return {
            'voxel_grid': self.get_voxel_grid(),
            'position': self.get_position_vector(),
            'velocity': self.get_velocity_vector(),
            'orientation': self.get_yaw_pitch(),
            'goal_direction': self.get_direction_to_goal(),
            'goal_distance': self.get_distance_to_goal(),
            'block_below': self.get_block_below(),
            'is_sprinting': is_sprinting
        }
    
    def update_fitness(self):
        """Calculate fitness score based on progress toward the red wool."""
        current_distance = self.get_distance_to_goal()
        
        if self.max_distance_to_goal is None:
            self.max_distance_to_goal = current_distance
        
        # Fitness: reward progress, penalize time spent
        progress = self.max_distance_to_goal - current_distance
        time_penalty = self.tick_count * 0.01
        
        self.fitness_score = max(0, progress - time_penalty)
        
        # Big bonus for reaching the red wool target
        if self.reached_goal:
            self.fitness_score += 1000.0
        
        return self.fitness_score
    
    def check_goal_reached(self, tolerance=1.5):
        """Check if bot is near the target coordinates."""
        if not self.bot.entity: return False
        return self.get_distance_to_goal() < tolerance

    def check_death(self, death_y=-20):
        """Checks if the bot has fallen below the configurable threshold."""
        if not self.bot.entity: return False
        if self.bot.entity.position.y < death_y:
            self.is_alive = False
            return True
        return False