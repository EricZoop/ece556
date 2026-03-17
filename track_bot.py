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
        """
        Debounce onGround so single-tick flickers don't change sensor input.
        Only switches state after ON_GROUND_DEBOUNCE consecutive agreeing ticks.
        """
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

    def get_direction_to_goal(self):
        """Goal direction in the bot's LOCAL frame (accounts for yaw)."""
        if not self.bot.entity:
            return [0.0, 0.0, 0.0]
        pos = self.bot.entity.position
        dx = self.target_pos.x - pos.x
        dy = self.target_pos.y - pos.y
        dz = self.target_pos.z - pos.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 0.001:
            return [0.0, 0.0, 0.0]

        yaw = round(self.bot.entity.yaw, RAY_QUANTIZE_DECIMALS)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        local_forward = (-sin_y * dx + -cos_y * dz) / dist
        local_right   = ( cos_y * dx + -sin_y * dz) / dist
        local_up      = dy / dist

        return [local_forward, local_right, local_up]

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
            # Quantize base orientation so tiny yaw/pitch drift
            # doesn't change which block the ray hits
            base_yaw   = round(self.bot.entity.yaw,   RAY_QUANTIZE_DECIMALS)
            base_pitch = round(self.bot.entity.pitch,  RAY_QUANTIZE_DECIMALS)
            yaw   = base_yaw + yaw_offset
            pitch = base_pitch + pitch_offset

            pos = self.bot.entity.position
            # Quantize ray origin so sub-epsilon position drift
            # doesn't shift rays across block boundaries
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
        19 rays + 3 pos + 3 vel + 2 orient + 3 goal_dir + 3 scalars = 33
        """
        if not self.bot.entity or not self.is_active:
            return {
                'rays': [0] * 19,
                'position': [0] * 3,
                'velocity': [0] * 3,
                'orientation': [0] * 2,
                'goal_direction': [0] * 3,
                'on_ground': 0,
                'distance_to_goal': 0,
                'time_remaining': 0,
            }

        rays = []

        # --- Forward hemisphere, horizontal (5) ---
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:
            rays.append(self._cast_ray(angle, 0))

        # --- Forward hemisphere, down-angled (5) ---
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:
            rays.append(self._cast_ray(angle, -0.35))

        # --- Forward hemisphere, upward (3) ---
        for angle in [-0.785, 0, 0.785]:
            rays.append(self._cast_ray(angle, 0.6))

        # --- Steep jump trajectory (3) ---
        for angle in [-0.524, 0, 0.524]:
            rays.append(self._cast_ray(angle, -0.9))

        # --- Ground checks (3) ---
        rays.append(self._cast_ray(0, -1.57))
        rays.append(self._cast_ray(-0.785, -1.2))
        rays.append(self._cast_ray( 0.785, -1.2))

        pos = self.bot.entity.position
        vel = self.bot.entity.velocity

        # Tick-based time remaining
        norm_time = max(0.0, 1.0 - (self.tick_count / timeout_ticks))

        # Update debounced ground state
        self._update_on_ground()

        # Track airtime using debounced value
        if self.on_ground:
            self.ground_ticks += 1
        else:
            self.air_ticks += 1

        # Track facing & movement every tick
        self._update_facing_goal()
        self._update_visited_blocks()

        return {
            'rays': rays,
            'position': [
                (pos.x - self.target_pos.x) / 25,
                (pos.y - self.target_pos.y) / 10,
                (pos.z - self.target_pos.z) / 25,
            ],
            'velocity': [
                max(-1, min(1, vel.x)),
                max(-1, min(1, vel.y)),
                max(-1, min(1, vel.z)),
            ],
            'orientation': [
                self.bot.entity.yaw / math.pi,
                self.bot.entity.pitch / (math.pi / 2),
            ],
            'goal_direction': self.get_direction_to_goal(),
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