"""
Bot tracking and sensor data collection for neuromorphic parkour training.
Handles raycasting (lidar), vector inputs, and fitness calculation.
"""

import math
import time
from javascript import require # type: ignore

Vec3 = require('vec3')

class BotTracker:
    def __init__(self, bot, target_pos, ray_max_dist=6):
        self.bot = bot
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        
        self.is_active = False 
        self.is_alive = True
        self.reached_goal = False
        self.tick_count = 0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None
        self.fitness_score = 0.0
        self.start_pos = None
        self.run_duration = 0.0
        
        self.visited_blocks = set()
        self.ray_max_dist = ray_max_dist

    def reset_for_run(self):
        self.is_active = True
        self.reached_goal = False
        self.tick_count = 0
        self.fitness_score = 0.0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None
        self.start_time = time.time()
        
        self.visited_blocks = set()
        
        if self.bot.entity and self.bot.entity.position:
            try:
                self.start_pos = self.bot.entity.position.clone()
                self.visited_blocks.add((int(self.start_pos.x), int(self.start_pos.z)))
            except:
                pos = self.bot.entity.position
                self.start_pos = Vec3(pos.x, pos.y, pos.z)
                self.visited_blocks.add((int(pos.x), int(pos.z)))
        else:
            self.start_pos = None
        
    def get_position_vector(self):
        if not self.bot.entity: return [0, 0, 0]
        pos = self.bot.entity.position
        try:
            return [float(pos.x), float(pos.y), float(pos.z)]
        except AttributeError:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    
    def get_velocity_vector(self):
        if not self.bot.entity: return [0, 0, 0]
        vel = self.bot.entity.velocity
        try:
            return [float(vel.x), float(vel.y), float(vel.z)]
        except AttributeError:
            return [float(vel[0]), float(vel[1]), float(vel[2])]
    
    def get_yaw_pitch(self):
        if not self.bot.entity: return [0, 0]
        return [self.bot.entity.yaw, self.bot.entity.pitch]
    
    def get_distance_to_goal(self):
        if not self.bot.entity: return 999.0
        try:
            return float(self.bot.entity.position.distanceTo(self.target_pos))
        except:
            p = self.get_position_vector()
            t = [self.target_pos.x, self.target_pos.y, self.target_pos.z]
            return math.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2 + (p[2]-t[2])**2)
            
    def get_direction_to_goal(self):
        if not self.bot.entity: return [0, 0, 0]
        current_pos = self.get_position_vector()
        dx = self.target_pos.x - current_pos[0]
        dy = self.target_pos.y - current_pos[1]
        dz = self.target_pos.z - current_pos[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < 0.001: return [0.0, 0.0, 0.0]
        return [dx/distance, dy/distance, dz/distance]

    def _cast_ray(self, yaw_offset, pitch_offset):
        if not self.bot.entity or not self.bot.entity.position: 
            return 0.0
        try:
            pos = self.get_position_vector()
            start_x = pos[0]
            start_y = pos[1]
            start_z = pos[2]
            
            height = 1.62
            if hasattr(self.bot.entity, 'height'):
                height = float(self.bot.entity.height)
            start_y += height
            
            yaw = self.bot.entity.yaw + yaw_offset
            pitch = self.bot.entity.pitch + pitch_offset
            
            vx = -math.sin(yaw) * math.cos(pitch)
            vy = math.sin(pitch)
            vz = -math.cos(yaw) * math.cos(pitch)
            
            for dist in range(1, self.ray_max_dist + 1):
                tx = start_x + (vx * dist)
                ty = start_y + (vy * dist)
                tz = start_z + (vz * dist)
                point = Vec3(tx, ty, tz)
                block = self.bot.blockAt(point)
                if block and block.boundingBox == 'block':
                    return 1.0 - (dist / self.ray_max_dist)
        except Exception:
            return 0.0
        return 0.0 

    def get_surrounding_rays(self):
        """
        Updated Vision System: "Platform Awareness"
        Now includes rays angled downward to detect valid landing spots 
        to the sides and diagonals, not just forward.
        """
        rays = []
        
        # 1. OBSTACLE AWARENESS (Eye Level - Pitch 0)
        # 8 Rays: Detects walls and barriers 360 degrees around
        for i in range(8):
            angle = i * (math.pi / 4) 
            rays.append(self._cast_ray(angle, 0))
            
        # 2. PLATFORM AWARENESS (Down 30° - Pitch -0.5)
        # 5 Rays: Detects if there is ground to stand/jump on.
        # This helps the bot find "New Blocks" to the side.
        # Angles: Forward(0), Left(90), Right(-90), Diagonal-L(45), Diagonal-R(-45)
        platform_angles = [0, math.pi/2, -math.pi/2, math.pi/4, -math.pi/4]
        for angle in platform_angles:
            rays.append(self._cast_ray(angle, -0.5))
            
        # 3. SPECIALTY RAYS
        rays.append(self._cast_ray(0, -1.2)) # Deep Void (Is there a pit strictly below?)
        rays.append(self._cast_ray(0, 0.5))  # Head Check (Is there a ceiling/overhang?)
        
        # Total Ray Count: 8 + 5 + 2 = 15 Rays
        return rays
    
    def get_on_ground(self):
        if not self.bot.entity: return 0
        return 1 if self.bot.entity.onGround else 0
    
    def get_sensor_input_vector(self):
        """
        Returns NORMALIZED sensor inputs (23 inputs).
        Removed: is_sprinting (always true) and is_sneaking (disabled).
        """
        position = self.get_position_vector()
        norm_position = [
            (position[0] - self.target_pos.x) / 20.0,
            (position[1] - self.target_pos.y) / 10.0,
            (position[2] - self.target_pos.z) / 20.0
        ]
        
        velocity = self.get_velocity_vector()
        norm_velocity = [
            max(-1, min(1, velocity[0])),
            max(-1, min(1, velocity[1])),
            max(-1, min(1, velocity[2]))
        ]
        
        orientation = self.get_yaw_pitch()
        norm_orientation = [
            orientation[0] / math.pi,      
            orientation[1] / (math.pi/2)   
        ]
        
        return {
            'rays': self.get_surrounding_rays(),   # 11 rays
            'position': norm_position,             # 3 values
            'velocity': norm_velocity,             # 3 values
            'orientation': norm_orientation,       # 2 values
            'goal_direction': self.get_direction_to_goal(), # 3 values
            'on_ground': self.get_on_ground(),     # 1 value
            # Total: 23
        }
    
    def update_fitness(self):
        """
        Straight-Line Fitness:
        Optimized for linear speed and obstacle traversal.
        Removes complex 'exploration' logic in favor of raw distance reduction.
        """
        current_distance = self.get_distance_to_goal()
        
        # Capture starting distance once
        if self.max_distance_to_goal is None:
            self.max_distance_to_goal = current_distance
            
        curr_pos = self.get_position_vector()
        
        # === 1. PROGRESS REWARD (The Main Driver) ===
        # Simple: How much closer are we than when we started?
        # If we move 1 block forward -> +150 points.
        # If we move 1 block backward -> -150 points.
        progress = self.max_distance_to_goal - current_distance
        progress_reward = progress * 150.0

        # === 2. ALIGNMENT BONUS (Steering) ===
        # Reward for looking directly at the goal.
        # This prevents them from running sideways or spinning.
        goal_dir = self.get_direction_to_goal()
        bot_yaw = self.bot.entity.yaw
        look_x = -math.sin(bot_yaw)
        look_z = -math.cos(bot_yaw)
        
        # Dot product: 1.0 = Perfect Alignment, 0.0 = 90 degrees off
        alignment = (look_x * goal_dir[0]) + (look_z * goal_dir[2])
        alignment_reward = 0
        if alignment > 0.8: # Only reward if generally facing the right way
            alignment_reward = alignment * 20.0

        # === 3. MOMENTUM REWARD ===
        # Keep moving fast!
        vel = self.get_velocity_vector()
        speed = math.sqrt(vel[0]**2 + vel[2]**2)
        momentum_reward = 0
        if speed > 0.15: 
            momentum_reward = speed * 50.0

        # === 4. PENALTIES ===
        
        # Death Penalty (Falling off the straight track)
        death_penalty = 0
        if curr_pos[1] < -1:
             death_penalty = 200.0
        
        # Stagnation Penalty
        # If we haven't made 1 block of progress after 2 seconds (40 ticks)
        stagnation_penalty = 0
        if self.tick_count > 40 and progress < 1.0:
             stagnation_penalty = self.tick_count * 2.0

        # === TOTAL CALCULATION ===
        fitness = (progress_reward + 
                   alignment_reward + 
                   momentum_reward - 
                   death_penalty - 
                   stagnation_penalty)
        
        # === GOAL COMPLETION ===
        if self.reached_goal:
            # Huge reward + Time Bonus
            fitness += 5000.0
            duration = time.time() - self.start_time
            # Every 0.01s saved is worth 10 points
            time_bonus = max(0, (10.0 - duration) * 1000.0)
            fitness += time_bonus
        
        self.fitness_score = fitness
        return self.fitness_score
    def check_goal_reached(self, tolerance=2):
        if not self.bot.entity: return False
        return self.get_distance_to_goal() < tolerance