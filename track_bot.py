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
        
        # Sensor configuration
        self.scan_radius = scan_radius

    def reset_for_run(self):
        self.is_active = True
        self.reached_goal = False
        self.tick_count = 0
        self.fitness_score = 0.0
        self.max_distance_to_goal = None
        
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
        Returns flattened voxel grid with specific handling for partial blocks and ladders.
        0: Air
        1: Full Solid Block
        0.5: Partial Block (Slab, Fence, Wall, Glass Pane)
        2: Liquid
        3: Ladder / Vine (Climbable)
        -1: Unloaded
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
                        voxel_data.append(3) #
                    elif 'water' in block.name or 'lava' in block.name:
                        voxel_data.append(2)
                    # Check for partial blocks (fences, slabs, glass panes)
                    elif any(s in block.name for s in ['fence', 'slab', 'wall', 'pane']):
                        voxel_data.append(0.5)
                    else:
                        # Fallback: Check collision shapes for non-standard cubes
                        if hasattr(block, 'shapes') and len(block.shapes) > 0:
                            # If the bounding box is not a full 1x1x1 cube
                            is_full = any(shape[3] == 0 and shape[4] == 0 and shape[5] == 0 and 
                                          shape[0] == 1 and shape[1] == 1 and shape[2] == 1 for shape in block.shapes)
                            voxel_data.append(1 if is_full else 0.5)
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
        current_distance = self.get_distance_to_goal()
        if self.max_distance_to_goal is None:
            self.max_distance_to_goal = current_distance
        
        progress = self.max_distance_to_goal - current_distance
        time_penalty = self.tick_count * 0.01
        
        # Additional reward for vertical progress (for ladders/slabs)
        vertical_bonus = 0
        if self.bot.entity:
            vertical_bonus = max(0, self.bot.entity.position.y) * 0.5

        self.fitness_score = max(0, progress - time_penalty + vertical_bonus)
        
        if self.reached_goal:
            self.fitness_score += 1000.0
        
        return self.fitness_score

    def check_goal_reached(self, tolerance=1):
        if not self.bot.entity: return False
        return self.get_distance_to_goal() < tolerance