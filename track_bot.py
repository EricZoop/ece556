import math
import time
from javascript import require  # type: ignore

Vec3 = require('vec3')


class BotTracker:
    def __init__(self, bot, target_pos, ray_max_dist=10, spawn_pos=None):
        self.bot = bot
        self.target_pos = Vec3(target_pos['x'], target_pos['y'], target_pos['z'])
        self.spawn_pos = (
            Vec3(spawn_pos['x'], spawn_pos['y'], spawn_pos['z']) if spawn_pos else None
        )

        self.is_active            = False
        self.reached_goal         = False
        self.tick_count           = 0
        self.start_time           = 0.0
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

    # ------------------------------------------------------------------
    def reset_for_run(self):
        self.is_active            = False
        self.reached_goal         = False
        self.tick_count           = 0
        self.fitness_score        = 0.0
        self.run_duration         = 0.0
        self.max_distance_to_goal = None
        self.min_distance_to_goal = None

        self.air_ticks             = 0
        self.ground_ticks          = 0
        self.best_forward_progress = 0.0
        self.closest_euclidean     = 999.0

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

        # Rotate the world-space direction into the bot's local frame
        # so the network sees "goal is to my front-left" not "goal is at -X, +Z"
        yaw = self.bot.entity.yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Mineflayer: forward = -sin(yaw)*X component, -cos(yaw)*Z component
        local_forward = (-sin_y * dx + -cos_y * dz) / dist
        local_right   = ( cos_y * dx + -sin_y * dz) / dist
        local_up      = dy / dist

        return [local_forward, local_right, local_up]

    # ------------------------------------------------------------------
    def _cast_ray(self, yaw_offset, pitch_offset):
        if not self.bot.entity:
            return 0.0
        try:
            yaw   = self.bot.entity.yaw + yaw_offset
            pitch = self.bot.entity.pitch + pitch_offset

            pos = self.bot.entity.position.offset(0, 1.62, 0)

            vx = -math.sin(yaw) * math.cos(pitch)
            vy =  math.sin(pitch)
            vz = -math.cos(yaw) * math.cos(pitch)

            steps = int(self.ray_max_dist * 2)
            for i in range(1, steps + 1):
                dist = i * 0.5
                check_pos = pos.offset(vx * dist, vy * dist, vz * dist)
                block = self.bot.blockAt(check_pos)

                if block and block.boundingBox == 'block':
                    return 1.0 - (dist / self.ray_max_dist)
        except Exception:
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    def get_sensor_input_vector(self, timeout_seconds):
        """
        Optimized sensor layout — 19 rays focused on the FORWARD hemisphere
        plus critical ground-check rays. All ray angles are relative to the
        bot's current yaw, so turning changes what they see.

        Ray budget:  19 rays  (down from 27)
        Total input: 33       (down from 41)

        Layout:
          Forward arc, horizontal (5):   -60°, -30°, 0°, +30°, +60°
          Forward arc, down-angled (5):  same yaw offsets, pitch -0.35
          Forward arc, up-angled (3):    -45°, 0°, +45°  pitch +0.6
          Steep jump trajectory (3):     -30°, 0°, +30°  pitch -0.9
          Ground checks (3):            straight-down, slight-left-down, slight-right-down
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
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:  # -60° to +60°
            rays.append(self._cast_ray(angle, 0))

        # --- Forward hemisphere, down-angled for platform detection (5) ---
        for angle in [-1.047, -0.524, 0, 0.524, 1.047]:
            rays.append(self._cast_ray(angle, -0.35))

        # --- Forward hemisphere, upward for ceiling/wall tops (3) ---
        for angle in [-0.785, 0, 0.785]:  # -45°, 0°, +45°
            rays.append(self._cast_ray(angle, 0.6))

        # --- Steep jump trajectory (3) ---
        for angle in [-0.524, 0, 0.524]:  # -30°, 0°, +30°
            rays.append(self._cast_ray(angle, -0.9))

        # --- Ground checks (3): down, down-left, down-right ---
        rays.append(self._cast_ray(0, -1.57))       # straight down
        rays.append(self._cast_ray(-0.785, -1.2))   # down-left
        rays.append(self._cast_ray( 0.785, -1.2))   # down-right

        # Total: 5 + 5 + 3 + 3 + 3 = 19

        pos = self.bot.entity.position
        vel = self.bot.entity.velocity

        elapsed   = max(0, time.time() - self.start_time) if self.start_time > 0 else 0
        norm_time = max(0, 1.0 - (elapsed / timeout_seconds))

        # Track airtime every tick this is called
        if self.bot.entity.onGround:
            self.ground_ticks += 1
        else:
            self.air_ticks += 1

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
            'on_ground': 1 if self.bot.entity.onGround else 0,
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

        # Track best Euclidean progress (distance reduced from start)
        if self.max_distance_to_goal is not None:
            prog = self.max_distance_to_goal - current_dist
            self.best_forward_progress = max(self.best_forward_progress, prog)

        if dy < -1.0 or dy > 4.0:
            return False

        xz_dist = math.sqrt(dx * dx + dz * dz)
        return xz_dist <= 1.25

    # ------------------------------------------------------------------
    def update_fitness(self, timeout_seconds):
        total_ticks = self.air_ticks + self.ground_ticks
        airtime_ratio = self.air_ticks / max(1, total_ticks)

        # 1. Euclidean distance reduction — the primary gradient signal.
        #    best_forward_progress is now computed as (start_dist - closest_dist)
        #    in check_goal_reached, so it works for ANY course shape.
        score = self.best_forward_progress * 20.0

        # 2. Airtime reward — scales with progress so bunny-hopping in place
        #    earns nothing, but sprint-jumping toward the goal earns a lot.
        if total_ticks > 10:
            progress_scale = min(1.0, self.best_forward_progress / 2.0)
            score += airtime_ratio * 20.0 * progress_scale

        # 3. Survival bonus — small reward for staying alive while progressing.
        #    Prevents the GA from treating a 2.7s fall the same as a 0.5s fall.
        if self.run_duration > 1.0 and self.best_forward_progress > 0.5:
            score += min(self.run_duration, 6.0) * 2.0

        # 4. Goal completion
        if self.reached_goal:
            score += 500.0
            time_bonus = max(0, (timeout_seconds - self.run_duration) * 50.0)
            score += time_bonus
            score += airtime_ratio * 100.0
        else:
            if self.run_duration < 0.5:
                score *= 0.1  # instant fall
            elif self.run_duration >= timeout_seconds - 0.5:
                if self.best_forward_progress < 1.5:
                    score *= 0.2  # coward penalty
                else:
                    score *= 0.8

        self.fitness_score = max(0.0, score)
        return self.fitness_score