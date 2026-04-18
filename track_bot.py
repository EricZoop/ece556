import math
import time
from javascript import require  # type: ignore

Vec3 = require('vec3')

# How many consecutive ticks of the same onGround value before we switch
ON_GROUND_DEBOUNCE = 3

# Quantize ray inputs to prevent float drift from changing hit results
RAY_QUANTIZE_DECIMALS = 3


class BotTracker:
    def __init__(self, bot, target_pos, ray_max_dist=10, spawn_pos=None):
        self.bot = bot
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        self.spawn_pos = (
            Vec3(spawn_pos['x'], spawn_pos['y'], spawn_pos['z']) if spawn_pos else None
        )

        self.is_active            = False
        self.is_settling          = False
        self.settle_ticks         = 0
        self.reached_goal         = False
        self.tick_count           = 0
        self.wall_start           = 0.0
        self.run_duration         = 0.0

        self.fitness_score        = 0.0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None
        self.ray_max_dist         = ray_max_dist

        # Airtime tracking
        self.air_ticks            = 0
        self.ground_ticks         = 0

        # Progress tracking
        self.best_forward_progress = 0.0
        self.closest_euclidean     = 999.0

        # Movement tracking - unique block positions visited
        self.visited_blocks        = set()

        # Facing-goal accumulator
        self.facing_goal_sum       = 0.0
        self.facing_goal_ticks     = 0

        # Debounced onGround state
        self._on_ground_stable     = True
        self._on_ground_counter    = 0

    # ------------------------------------------------------------------
    def set_target_pos(self, new_pos):
        """Update goal during a run or between runs (used by pause/newgoal)."""
        self.target_pos = Vec3(new_pos['x'], new_pos['y'], new_pos['z'])

    # ------------------------------------------------------------------
    def reset_for_run(self):
        self.is_active            = False
        self.is_settling          = False
        self.settle_ticks         = 0
        self.reached_goal         = False
        self.tick_count           = 0
        self.wall_start           = 0.0
        self.fitness_score        = 0.0
        self.run_duration         = 0.0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None

        self.air_ticks             = 0
        self.ground_ticks          = 0
        self.best_forward_progress = 0.0
        self.closest_euclidean     = 999.0
        self.visited_blocks        = set()
        self.facing_goal_sum       = 0.0
        self.facing_goal_ticks     = 0

        self._on_ground_stable     = True
        self._on_ground_counter    = 0

    # ------------------------------------------------------------------
    def _update_on_ground(self):
        if not self.bot.entity:
            return

        raw = bool(self.bot.entity.onGround)

        if raw == self._on_ground_stable:
            self._on_ground_counter = 0
        else:
            self._on_ground_counter += 1
            if self._on_ground_counter >= ON_GROUND_DEBOUNCE:
                self._on_ground_stable = raw
                self._on_ground_counter = 0

    @property
    def on_ground(self):
        return self._on_ground_stable

    # ------------------------------------------------------------------
    def get_distance_to_goal(self):
        if not self.bot.entity:
            return 999.0
        try:
            return float(self.bot.entity.position.distanceTo(self.target_pos))
        except Exception:
            return 999.0

    def _quantized_yaw(self):
        return round(self.bot.entity.yaw, RAY_QUANTIZE_DECIMALS)

    # ------------------------------------------------------------------
    # Rotation-invariant transforms — all use the SAME quantized yaw
    # so that position, velocity, and goal_direction stay internally consistent.
    # ------------------------------------------------------------------
    def _world_to_local_xz(self, wx, wz, cos_y, sin_y):
        """Rotate an (x, z) world-frame vector into the bot's local frame.
        Returns (forward, right). Matches the forward/right basis used by
        Minecraft's yaw convention (yaw=0 faces -Z, yaw increases clockwise
        viewed from above, so yaw=+pi/2 faces +X)."""
        local_forward = -sin_y * wx - cos_y * wz
        local_right   =  cos_y * wx - sin_y * wz
        return local_forward, local_right

    def get_direction_to_goal(self):
        """Unit vector from bot to goal, in the bot's LOCAL frame."""
        if not self.bot.entity:
            return [0.0, 0.0, 0.0]
        pos = self.bot.entity.position
        dx = self.target_pos.x - pos.x
        dy = self.target_pos.y - pos.y
        dz = self.target_pos.z - pos.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.001:
            return [0.0, 0.0, 0.0]

        yaw = self._quantized_yaw()
        fwd, right = self._world_to_local_xz(dx, dz, math.cos(yaw), math.sin(yaw))
        return [fwd / dist, right / dist, dy / dist]

    def _update_facing_goal(self):
        if not self.bot.entity:
            return
        pos = self.bot.entity.position
        dx = self.target_pos.x - pos.x
        dz = self.target_pos.z - pos.z
        dist_xz = math.sqrt(dx * dx + dz * dz)
        if dist_xz < 0.001:
            return

        goal_nx = dx / dist_xz
        goal_nz = dz / dist_xz

        yaw = self.bot.entity.yaw
        fwd_x = -math.sin(yaw)
        fwd_z = -math.cos(yaw)

        dot = fwd_x * goal_nx + fwd_z * goal_nz
        self.facing_goal_sum += dot
        self.facing_goal_ticks += 1

    def _update_visited_blocks(self):
        if not self.bot.entity:
            return
        pos = self.bot.entity.position
        block_key = (int(math.floor(pos.x)), int(math.floor(pos.z)))
        self.visited_blocks.add(block_key)

    # ------------------------------------------------------------------
    def _cast_ray(self, yaw_offset, pitch_offset):
        if not self.bot.entity:
            return 0.0
        try:
            base_yaw   = self._quantized_yaw()
            base_pitch = round(self.bot.entity.pitch, RAY_QUANTIZE_DECIMALS)
            yaw   = base_yaw + yaw_offset
            pitch = base_pitch + pitch_offset

            pos = self.bot.entity.position
            eye_x = round(pos.x, RAY_QUANTIZE_DECIMALS)
            eye_y = round(pos.y + 1.62, RAY_QUANTIZE_DECIMALS)
            eye_z = round(pos.z, RAY_QUANTIZE_DECIMALS)

            vx = -math.sin(yaw) * math.cos(pitch)
            vy =  math.sin(pitch)
            vz = -math.cos(yaw) * math.cos(pitch)

            steps = int(self.ray_max_dist * 2)
            for i in range(1, steps + 1):
                dist = i * 0.5
                check_pos = Vec3(
                    eye_x + vx * dist,
                    eye_y + vy * dist,
                    eye_z + vz * dist,
                )
                block = self.bot.blockAt(check_pos)

                if block and block.boundingBox == 'block':
                    return 1.0 - (dist / self.ray_max_dist)
        except Exception:
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    def get_sensor_input_vector(self, timeout_ticks):
        """
        Returns a dict of sensor channels. All spatial channels are expressed
        in the bot's LOCAL (ego-centric) frame so the policy generalizes
        across courses with different orientations.

        Channel sizes:
          rays           : 19  (unchanged)
          position       :  3  (local-frame offset to goal, [fwd, up, right])
          velocity       :  3  (local-frame velocity, [fwd, up, right])
          self_state     :  2  (pitch, horizontal_speed) - BOTH rotation-invariant
          goal_direction :  3  (local-frame unit vector, [fwd, right, up])
          on_ground          :  1
          distance_to_goal   :  1
          time_remaining     :  1
                          ----
                          total 33
        """
        if not self.bot.entity or not self.is_active:
            return {
                'rays': [0] * 19,
                'position': [0] * 3,
                'velocity': [0] * 3,
                'self_state': [0] * 2,
                'goal_direction': [0] * 3,
                'on_ground': 0,
                'distance_to_goal': 0,
                'time_remaining': 0,
            }

        rays = []
        # Forward hemisphere, horizontal (5)
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:
            rays.append(self._cast_ray(angle, 0))
        # Forward hemisphere, down-angled (5)
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:
            rays.append(self._cast_ray(angle, -0.35))
        # Forward hemisphere, upward (3)
        for angle in [-0.785, 0, 0.785]:
            rays.append(self._cast_ray(angle, 0.6))
        # Steep jump trajectory (3)
        for angle in [-0.524, 0, 0.524]:
            rays.append(self._cast_ray(angle, -0.9))
        # Ground checks (3)
        rays.append(self._cast_ray(0, -1.57))
        rays.append(self._cast_ray(-0.785, -1.2))
        rays.append(self._cast_ray( 0.785, -1.2))

        pos = self.bot.entity.position
        vel = self.bot.entity.velocity

        # One quantized yaw, shared by position / velocity / goal_direction
        yaw = self._quantized_yaw()
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # -- Rotation-invariant position: goal offset in bot's frame --
        gdx = self.target_pos.x - pos.x
        gdy = self.target_pos.y - pos.y
        gdz = self.target_pos.z - pos.z
        pos_fwd, pos_right = self._world_to_local_xz(gdx, gdz, cos_y, sin_y)
        position = [pos_fwd / 25.0, gdy / 10.0, pos_right / 25.0]

        # -- Rotation-invariant velocity: velocity in bot's frame --
        vel_fwd, vel_right = self._world_to_local_xz(vel.x, vel.z, cos_y, sin_y)
        velocity = [
            max(-1.0, min(1.0, vel_fwd)),
            max(-1.0, min(1.0, vel.y)),
            max(-1.0, min(1.0, vel_right)),
        ]

        # -- Rotation-invariant self-state: pitch + horizontal speed --
        # Yaw is deliberately EXCLUDED. It is world-absolute and caused the
        # policy to memorize a single course orientation.
        speed_xz = math.sqrt(vel.x * vel.x + vel.z * vel.z)
        self_state = [
            self.bot.entity.pitch / (math.pi / 2.0),
            min(1.0, speed_xz / 0.5),
        ]

        # -- Goal direction in local frame (already rotation-invariant) --
        gd_dist = math.sqrt(gdx * gdx + gdy * gdy + gdz * gdz)
        if gd_dist < 0.001:
            goal_direction = [0.0, 0.0, 0.0]
        else:
            gd_fwd, gd_right = self._world_to_local_xz(gdx, gdz, cos_y, sin_y)
            goal_direction = [gd_fwd / gd_dist, gd_right / gd_dist, gdy / gd_dist]

        norm_time = max(0.0, 1.0 - (self.tick_count / timeout_ticks))

        self._update_on_ground()
        if self.on_ground:
            self.ground_ticks += 1
        else:
            self.air_ticks += 1

        self._update_facing_goal()
        self._update_visited_blocks()

        return {
            'rays': rays,
            'position': position,
            'velocity': velocity,
            'self_state': self_state,
            'goal_direction': goal_direction,
            'on_ground': 1 if self.on_ground else 0,
            'distance_to_goal': min(1.0, self.get_distance_to_goal() / 30.0),
            'time_remaining': norm_time,
        }

    # ------------------------------------------------------------------
    def check_goal_reached(self):
        if not self.bot.entity:
            return False

        pos = self.bot.entity.position
        dx = pos.x - self.target_pos.x
        dy = pos.y - self.target_pos.y
        dz = pos.z - self.target_pos.z

        current_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.closest_euclidean = min(self.closest_euclidean, current_dist)

        if self.max_distance_to_goal is not None:
            prog = self.max_distance_to_goal - current_dist
            self.best_forward_progress = max(self.best_forward_progress, prog)

        if dy < -1.0 or dy > 4.0:
            return False

        xz_dist = math.sqrt(dx * dx + dz * dz)
        return xz_dist <= 1.25

    # ------------------------------------------------------------------
    def update_fitness(self, timeout_seconds):
        total_ticks   = self.air_ticks + self.ground_ticks
        airtime_ratio = self.air_ticks / max(1, total_ticks)
        num_blocks    = len(self.visited_blocks)

        avg_facing = (self.facing_goal_sum / self.facing_goal_ticks) \
            if self.facing_goal_ticks > 0 else 0.0

        exploration_score = min(num_blocks, 8) * 5.0
        facing_score = max(0.0, (avg_facing + 1.0) / 2.0) * 30.0
        progress_score = self.best_forward_progress * 20.0

        closeness_score = 0.0
        if self.closest_euclidean < 5.0:
            closeness = max(0.0, 5.0 - self.closest_euclidean)
            closeness_score = closeness * closeness * 10.0

        airtime_score = 0.0
        if total_ticks > 10:
            progress_scale = min(1.0, self.best_forward_progress / 2.0)
            airtime_score = airtime_ratio * 20.0 * progress_scale

        survival_score = 0.0
        if self.run_duration > 1.0 and self.best_forward_progress > 0.5:
            survival_score = min(self.run_duration, 6.0) * 2.0

        score = (exploration_score + facing_score + progress_score +
                 closeness_score + airtime_score + survival_score)

        if self.reached_goal:
            score += 500.0
            time_bonus = max(0, (timeout_seconds - self.run_duration) * 50.0)
            score += time_bonus
            score += airtime_ratio * 100.0

        if not self.reached_goal:
            if num_blocks <= 1:
                score *= 0.05
            elif self.best_forward_progress < 0.3:
                score *= 0.3
            elif self.run_duration < 0.5:
                score *= 0.1
            elif self.run_duration >= timeout_seconds - 0.5:
                if self.best_forward_progress < 1.5:
                    score *= 0.15

        self.fitness_score = max(0.0, score)
        return self.fitness_score