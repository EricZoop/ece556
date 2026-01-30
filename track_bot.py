"""
Bot tracking and sensor data collection for neuromorphic parkour training.
Handles voxel mapping and vector inputs for neural processing.
"""

import math
from javascript import require # type: ignore

Vec3 = require('vec3')

class BotTracker:
    """Tracks bot state and provides sensory input vectors for neuromorphic processing."""
    
    def __init__(self, bot, target_pos, scan_radius=5):
        self.bot = bot
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        
        # Performance tracking
        self.is_active = False 
        self.is_alive = True
        self.reached_goal = False
        self.tick_count = 0
        self.max_distance_to_goal = None
        self.fitness_score = 0.0
        self.start_pos = None  # Track start position
        
        # Sensor configuration
        self.scan_radius = scan_radius

    def reset_for_run(self):
        self.is_active = True
        self.reached_goal = False
        self.tick_count = 0
        self.fitness_score = 0.0
        self.max_distance_to_goal = None
        
        # Capture start position for displacement checks
        if self.bot.entity:
            self.start_pos = self.bot.entity.position.clone()
        else:
            self.start_pos = None
        
    def get_position_vector(self):
        if not self.bot.entity: return [0, 0, 0]
        pos = self.bot.entity.position
        return [pos.x, pos.y, pos.z]
    
    def get_velocity_vector(self):
        if not self.bot.entity: return [0, 0, 0]
        vel = self.bot.entity.velocity
        return [vel.x, vel.y, vel.z]
    
    def get_yaw_pitch(self):
        if not self.bot.entity: return [0, 0]
        return [self.bot.entity.yaw, self.bot.entity.pitch]
    
    def get_distance_to_goal(self):
        if not self.bot.entity: return 999.0
        return self.bot.entity.position.distanceTo(self.target_pos)
    
    def get_direction_to_goal(self):
        if not self.bot.entity: return [0, 0, 0]
        pos = self.bot.entity.position
        
        dx = self.target_pos.x - pos.x
        dy = self.target_pos.y - pos.y
        dz = self.target_pos.z - pos.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < 0.001: return [0.0, 0.0, 0.0]
        
        return [dx/distance, dy/distance, dz/distance]
    
    def get_voxel_grid(self):
        """
        Returns flattened voxel grid.
        0: Air, 1: Solid, 0.5: Partial, 2: Liquid, 3: Climbable, -1: Unloaded
        """
        if not self.bot.entity: return []
        pos = self.bot.entity.position.floored()
        voxel_data = []
        
        for dy in range(-self.scan_radius, self.scan_radius + 1):
            for dz in range(-self.scan_radius, self.scan_radius + 1):
                for dx in range(-self.scan_radius, self.scan_radius + 1):
                    target_pos = pos.offset(dx, dy, dz)
                    block = self.bot.blockAt(target_pos)
                    
                    if block is None:
                        voxel_data.append(-1)
                    elif block.name == 'air':
                        voxel_data.append(0)
                    elif 'ladder' in block.name or 'vine' in block.name:
                        voxel_data.append(3)
                    elif 'water' in block.name or 'lava' in block.name:
                        voxel_data.append(2)
                    elif any(s in block.name for s in ['fence', 'slab', 'wall', 'pane']):
                        voxel_data.append(0.5)
                    else:
                        voxel_data.append(1)
        
        return voxel_data
    
    def get_block_below(self):
        if not self.bot.entity: return -1
        pos = self.bot.entity.position.floored()
        block = self.bot.blockAt(pos.offset(0, -1, 0))
        
        if block is None: return -1
        if block.name == 'air': return 0
        if 'ladder' in block.name: return 3
        if any(s in block.name for s in ['fence', 'slab', 'wall', 'pane']): return 0.5
        return 1
    
    def get_sensor_input_vector(self):
        is_sprinting = 0
        if self.bot.entity and hasattr(self.bot, 'controlState'):
            try:
                is_sprinting = 1 if self.bot.controlState.sprint else 0
            except (AttributeError, TypeError):
                is_sprinting = 0
        
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
        current_distance = self.get_distance_to_goal()
        if self.max_distance_to_goal is None:
            self.max_distance_to_goal = current_distance
        
        # 1. Base Progress
        progress = self.max_distance_to_goal - current_distance
        
        # 2. Time Penalty (Linear)
        time_penalty = self.tick_count * 0.1  # Increased slightly to encourage speed
        
        # 3. Vertical Bonus (Reward for climbing)
        vertical_bonus = 0
        if self.bot.entity:
            vertical_bonus = max(0, self.bot.entity.position.y) * 2.0
            
        # 4. Stagnation / Idleness Penalty (XZ Plane only)
        stagnation_penalty = 0
        if self.bot.entity and self.start_pos:
            curr = self.bot.entity.position
            # Calculate 2D Euclidean distance on XZ plane
            xz_displacement = math.sqrt((curr.x - self.start_pos.x)**2 + (curr.z - self.start_pos.z)**2)
            
            # If moved less than 1.5 blocks horizontally over the whole run
            if xz_displacement < 1.5:
                stagnation_penalty = 150.0  # Massive penalty for camping
        
        # Base Score prevents 0.00 floor for honest attempts
        # Active bots start at 200. Lazy bots drop to ~50. Reaching goal is ~1200+
        BASE_SCORE = 200.0
        
        final_score = BASE_SCORE + progress + vertical_bonus - time_penalty - stagnation_penalty
        
        if self.reached_goal:
            final_score += 1000.0
            
        self.fitness_score = max(0, final_score)
        
        return self.fitness_score

    def check_goal_reached(self, tolerance=1):
        if not self.bot.entity: return False
        return self.get_distance_to_goal() < tolerance