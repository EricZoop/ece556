"""
Trial run script - Test a trained network on an UNSEEN parkour course.

Variable names match main.py exactly so configs stay in sync:
  SPAWN_POS, GOAL_POS, TEAM_NAME, TIMEOUT_SECONDS, TIMEOUT_TICKS,
  RAY_MAX_DIST, VOID_Y, YAW_RATE, SENSOR_QUANTIZE_DECIMALS.

Sensor composition also matches main.py's get_neural_action() so weights
trained there can be evaluated here without translation.
"""

import sys
import time
import threading
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ======================================================================
# SHARED CONFIG (must match main.py names exactly)
# ======================================================================
TEAM_NAME = "bots"

# --- UNSEEN course: 90-degree RIGHT-hand jump ---
# (mirror of the left-hand training course — goal is now to the bot's right)
SPAWN_POS = {'x': 0.5, 'y': -63, 'z': 0.5}
GOAL_POS  = {'x': 6.5, 'y': -63, 'z': 0.5}

# Trial parameters
TIMEOUT_SECONDS = 30
TIMEOUT_TICKS   = TIMEOUT_SECONDS * 20
RAY_MAX_DIST    = 7
VOID_Y          = -65

# Sensor quantization (must match main.py)
SENSOR_QUANTIZE_DECIMALS = 3

# Control
YAW_RATE = 0.3

# ======================================================================
# LOAD NETWORK
# ======================================================================
if len(sys.argv) < 2:
    print("Usage: python trial_run.py <network_file.json>")
    print("Example: python trial_run.py 2026-04-16_14-22-03_gen50_net3.json")
    sys.exit(1)

NETWORK_FILE = sys.argv[1]


class TrialBot:
    """Single bot running a pre-trained network on an unseen course."""

    def __init__(self, network_file):
        print(f"\n{'='*60}")
        print("TRIAL RUN MODE - Testing Trained Network")
        print(f"{'='*60}")
        print(f"Loading network from: {network_file}")

        self.network = GeneticAlgorithm.load_best_network(network_file)
        print("Network loaded successfully!")
        print(f"  Input size:  {self.network.input_size}")
        print(f"  Hidden size: {self.network.hidden_size}")
        print(f"  Output size: {self.network.output_size}")

        self.tracker      = None
        self.bot          = None
        self.trial_active = False

    # ------------------------------------------------------------------
    def create_bot(self):
        self.bot = mineflayer.createBot({
            'host':     'localhost',
            'port':     25565,
            'username': "TrialBot",
            'version':  '1.19.4',
        })

        self.tracker = BotTracker(
            self.bot,
            GOAL_POS,
            ray_max_dist=RAY_MAX_DIST,
            spawn_pos=SPAWN_POS,
        )

        self.bot._setup_complete     = False
        self.bot._spawn_tp_done      = False
        self.bot._ready_tick_counter = 0

        @Once(self.bot, 'spawn')
        def on_spawn(*args):
            self.bot.chat(f'/team join {TEAM_NAME}')
            print("\n[SPAWN] TrialBot spawned, waiting for position...")

        @On(self.bot, 'move')
        def on_move(*args):
            if (not self.bot._spawn_tp_done
                    and self.bot.entity
                    and self.bot.entity.position):
                self.bot._spawn_tp_done = True
                self.bot.chat(
                    f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}'
                )
                print("[READY] TrialBot teleported to course start")

        @On(self.bot, 'physicsTick')
        def handle_physics(*args):
            if self.bot._spawn_tp_done and not self.bot._setup_complete:
                self.bot._ready_tick_counter += 1
                if self.bot._ready_tick_counter >= 20:
                    self.bot._setup_complete = True
                    self.start_trial()
                return

            if not self.bot._setup_complete or not self.trial_active:
                return

            self.tracker.tick_count += 1

            # Void death
            if self.bot.entity and self.bot.entity.position.y < VOID_Y:
                self.end_trial("FELL INTO VOID")
                return

            # Goal reached
            if self.tracker.check_goal_reached():
                self.tracker.reached_goal = True
                self.end_trial("SUCCESS - REACHED GOAL!")
                return

            # Timeout
            if self.tracker.tick_count >= TIMEOUT_TICKS:
                self.end_trial("TIMEOUT")
                return

            # Act every 2 ticks (matches main.py training cadence)
            if self.tracker.tick_count % 2 == 0:
                action = self.get_neural_action()
                self.execute_action(action)

                if self.tracker.tick_count % 40 == 0:
                    d = self.tracker.get_distance_to_goal()
                    print(f"[{self.tracker.tick_count // 20:>2}s] "
                          f"dist={d:5.2f}  closest={self.tracker.closest_euclidean:5.2f}")

    # ------------------------------------------------------------------
    def start_trial(self):
        print(f"\n{'='*60}")
        print("TRIAL STARTING")
        print(f"{'='*60}")
        self.tracker.reset_for_run()
        self.tracker.is_active = True
        self.tracker.wall_start = time.time()
        self.trial_active = True

    # ------------------------------------------------------------------
    def get_neural_action(self):
        """Must match main.py's get_neural_action() byte-for-byte."""
        sensor = self.tracker.get_sensor_input_vector(TIMEOUT_TICKS)

        inputs = []
        inputs.extend(sensor['rays'])            # 19
        inputs.extend(sensor['position'])        # 3 (local frame)
        inputs.extend(sensor['velocity'])        # 3 (local frame)
        inputs.extend(sensor['self_state'])      # 2 (pitch + horizontal speed)
        inputs.extend(sensor['goal_direction'])  # 3 (local frame)
        inputs.append(sensor['on_ground'])       # 1
        inputs.append(sensor['distance_to_goal'])# 1
        inputs.append(sensor['time_remaining'])  # 1
        # ---------------------------------------  33

        inputs = [round(v, SENSOR_QUANTIZE_DECIMALS) for v in inputs]
        return self.network.forward(inputs)

    # ------------------------------------------------------------------
    def execute_action(self, action):
        """Action layout (must match main.py):
            [0] strafe_left  (bool)
            [1] strafe_right (bool)
            [2] jump         (bool)
            [3] yaw_delta    (float in [-1, +1])"""
        if not self.bot.entity:
            return

        self.bot.setControlState('left',    bool(action[0]))
        self.bot.setControlState('right',   bool(action[1]))
        self.bot.setControlState('jump',    bool(action[2]))
        self.bot.setControlState('forward', True)
        self.bot.setControlState('sprint',  True)

        yaw_delta = action[3] * YAW_RATE
        new_yaw = self.bot.entity.yaw + yaw_delta
        try:
            self.bot.look(new_yaw, self.bot.entity.pitch, True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def end_trial(self, reason):
        self.trial_active      = False
        self.tracker.is_active = False
        self.bot.clearControlStates()

        self.tracker.run_duration = self.tracker.tick_count / 20.0
        fitness = self.tracker.update_fitness(TIMEOUT_SECONDS)

        total_ticks = self.tracker.air_ticks + self.tracker.ground_ticks
        airtime_pct = self.tracker.air_ticks / max(1, total_ticks)

        print(f"\n{'='*60}")
        print("TRIAL COMPLETE")
        print(f"{'='*60}")
        print(f"Result:         {reason}")
        print(f"Time:           {self.tracker.run_duration:.1f}s "
              f"({self.tracker.tick_count} ticks)")
        print(f"Fitness:        {fitness:.2f}")
        print(f"Final dist:     {self.tracker.get_distance_to_goal():.2f} blocks")
        print(f"Closest dist:   {self.tracker.closest_euclidean:.2f} blocks")
        print(f"Blocks visited: {len(self.tracker.visited_blocks)}")
        print(f"Airtime ratio:  {airtime_pct:.1%}")
        print(f"{'='*60}")

        print("\nPress Enter to retry, or Ctrl+C to exit...")
        try:
            input()
            self.retry_trial()
        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye!")
            sys.exit(0)

    # ------------------------------------------------------------------
    def retry_trial(self):
        print("\n[RETRY] Restarting trial in 2 seconds...")

        def restart():
            self.bot.chat(
                f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}'
            )
            self.tracker.reset_for_run()
            self.tracker.is_active = True
            self.tracker.wall_start = time.time()
            self.trial_active = True
            print(f"\n{'='*60}")
            print("TRIAL RESTARTING")
            print(f"{'='*60}")

        threading.Timer(2.0, restart).start()


def main():
    trial = TrialBot(NETWORK_FILE)
    trial.create_bot()
    globals()['trial'] = trial


if __name__ == "__main__":
    main()