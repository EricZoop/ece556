################################################
# Minecraft Neuromorphic Parkour Trainer v2
# GMU ECE 556 - 4/20/2026
#
# Eric Zipor
# G01359507
################################################

import math
import random
import threading
import sys
import time
import json
import datetime
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ======================================================================
# SHARED CONFIG (must match trial_run.py names exactly)
# ======================================================================
TEAM_NAME    = "bots"
ADMIN_PLAYER = "EricZoop"

# Coordinates
SPAWN_POS     = {'x': 0.5,   'y': -63, 'z': 0.5}
GOAL_POS      = {'x': 0.5,   'y': -63, 'z': 6.5}
PURGATORY_POS = {'x': -14.5, 'y': -63, 'z': 0.5}

# Training / trial parameters
TIMEOUT_SECONDS = 15
TIMEOUT_TICKS   = TIMEOUT_SECONDS * 20
RAY_MAX_DIST    = 7
VOID_Y          = -65

# Neural network architecture (must match the shape used by trial_run's loaded nets)
INPUT_SIZE        = 33
HIDDEN_LAYER_SIZE = 32
OUTPUT_SIZE       = 4

# Sensor quantization (must be applied identically in trial_run)
SENSOR_QUANTIZE_DECIMALS = 3

# Control
YAW_RATE = 0.3   # rad per action step

# ======================================================================
# TRAINING-ONLY CONFIG
# ======================================================================
POPULATION_SIZE = 10
SEQUENTIAL_MODE = True
TRIALS_PER_NET  = 3

MUTATION_RATE     = 0.1
MUTATION_STRENGTH = 0.2
ELITE_COUNT       = 2

BOT_COUNT = 1 if SEQUENTIAL_MODE else POPULATION_SIZE

# Bot states
STATE_IDLE        = 'IDLE'
STATE_TP_TO_SPAWN = 'TP_TO_SPAWN'
STATE_VERIFY_POS  = 'VERIFY_POS'
STATE_SETTLING    = 'SETTLING'
STATE_ACTIVE      = 'ACTIVE'
STATE_POST_RUN    = 'POST_RUN'

# Stability thresholds
SPAWN_TOLERANCE     = 1.0
STABLE_POS_EPSILON  = 0.02
STABLE_VEL_EPSILON  = 0.05
TELEPORT_WAIT_TICKS = 10
SETTLE_MIN_TICKS    = 20
SETTLE_STABLE_NEED  = 10
SETTLE_MAX_TICKS    = 120
POST_RUN_TICKS      = 20
STARTUP_TICKS       = 100


# ======================================================================
# HELPERS
# ======================================================================
def safe_stop_bot(bot):
    if not bot:
        return
    try:
        bot.clearControlStates()
    except Exception:
        for key in ('forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'):
            try:
                bot.setControlState(key, False)
            except Exception:
                pass
    try:
        if bot.entity:
            bot.entity.velocity.x = 0
            bot.entity.velocity.y = 0
            bot.entity.velocity.z = 0
    except Exception:
        pass


def _tp_cmd(x, y, z):
    return f'/tp @s {x} {y} {z}'


def _dist_to(bot, target_dict):
    if not bot or not bot.entity:
        return 999.0
    p = bot.entity.position
    dx = p.x - target_dict['x']
    dy = p.y - target_dict['y']
    dz = p.z - target_dict['z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ======================================================================
# MAIN GENERATION MANAGER
# ======================================================================
class BotGeneration:
    def __init__(self):
        self.trackers = []
        self.generation_complete = False
        self.ga = GeneticAlgorithm(
            population_size   = POPULATION_SIZE,
            input_size        = INPUT_SIZE,
            hidden_size       = HIDDEN_LAYER_SIZE,
            output_size       = OUTPUT_SIZE,
            mutation_rate     = MUTATION_RATE,
            mutation_strength = MUTATION_STRENGTH,
            elite_count       = ELITE_COUNT,
        )
        self.auto_evolve              = True
        self.all_time_best_score      = 0.0
        self.all_time_best_network    = None
        self.all_time_best_generation = 0

        # --- Pause / new-goal coordination ---
        # pending_pause: user asked for pause; honor after current gen evolves
        # paused_for_goal: currently paused, CLI thread may prompt for new coords
        self.pending_pause  = False
        self.paused_for_goal = False

        # Sequential-mode state
        self.current_pop_index  = 0
        self.current_trial      = 0
        self.trial_fitness      = []
        self.sequential_fitness = [0.0] * POPULATION_SIZE

        # Tick-driven state machine
        self.bot_state    = {}
        self.state_tick   = {}
        self.tp_retries   = {}
        self.stable_ticks = {}
        self.last_pos     = {}

        self.global_tick = 0
        self.started     = False

    # ------------------------------------------------------------------
    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host':     '192.168.1.118',
            'port':     25565,
            'username': f'Bot_{index}',
            'version':  '1.19.4',
        })

        tracker = BotTracker(bot, GOAL_POS, ray_max_dist=RAY_MAX_DIST, spawn_pos=SPAWN_POS)
        tracker.bot_index = index

        self.bot_state[index]   = STATE_IDLE
        self.state_tick[index]  = 0
        self.tp_retries[index]  = 0
        self.stable_ticks[index] = 0
        self.last_pos[index]    = None

        @Once(bot, 'spawn')
        def on_spawn(*args):
            try:
                bot.chat(f'/team join {TEAM_NAME}')
                bot.chat('/gamemode adventure')
                bot.chat('/attribute @s minecraft:generic.max_health base set 2')
                if index == 0:
                    bot.chat('/gamerule fallDamage false')
                    bot.chat('/gamerule doImmediateRespawn true')
                    bot.chat('/gamerule sendCommandFeedback false')
                    bot.chat('/gamerule keepInventory true')
                    bot.chat('/gamerule spawnRadius 0')
                    bot.chat('/gamerule randomTickSpeed 0')
                    bot.chat(
                        f'/setworldspawn {PURGATORY_POS["x"]} '
                        f'{PURGATORY_POS["y"]} {PURGATORY_POS["z"]}'
                    )
                bot.chat(_tp_cmd(
                    PURGATORY_POS['x'], PURGATORY_POS['y'], PURGATORY_POS['z']
                ))
            except Exception as e:
                print(f'[Bot {index}] Spawn error: {e}')

        @On(bot, 'physicsTick')
        def handle_physics(*args):
            self._tick_state_machine(index)

        self.trackers.append(tracker)

    # ------------------------------------------------------------------
    # STATE MACHINE
    # ------------------------------------------------------------------
    def _set_state(self, index, new_state):
        self.bot_state[index]  = new_state
        self.state_tick[index] = 0

    def _tick_state_machine(self, index):
        tracker = self.trackers[index]
        bot     = tracker.bot
        state   = self.bot_state[index]

        self.state_tick[index] += 1
        tick = self.state_tick[index]

        # Startup delay
        if not self.started:
            self.global_tick += 1
            safe_stop_bot(bot)
            if self.global_tick >= STARTUP_TICKS and index == 0:
                self.started = True
                self.start_new_run()
            return

        if state == STATE_IDLE:
            safe_stop_bot(bot)
            return

        if state == STATE_TP_TO_SPAWN:
            safe_stop_bot(bot)
            if tick >= TELEPORT_WAIT_TICKS:
                self._set_state(index, STATE_VERIFY_POS)
            return

        if state == STATE_VERIFY_POS:
            safe_stop_bot(bot)
            dist = _dist_to(bot, SPAWN_POS)
            on_ground = bot.entity and bot.entity.onGround

            if dist < SPAWN_TOLERANCE and on_ground:
                self.tp_retries[index] = 0
                self._set_state(index, STATE_SETTLING)
                tracker.is_settling  = True
                tracker.settle_ticks = 0
                return

            if tick > 20:
                self.tp_retries[index] += 1
                if self.tp_retries[index] > 5:
                    print(f'  [Warn] Bot {index} failed to reach spawn after 5 retries.')
                    self.tp_retries[index] = 0
                    self._set_state(index, STATE_SETTLING)
                    tracker.is_settling  = True
                    tracker.settle_ticks = 0
                    return
                try:
                    bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
                except Exception:
                    pass
                self._set_state(index, STATE_TP_TO_SPAWN)
            return

        if state == STATE_SETTLING:
            safe_stop_bot(bot)

            if tick == 1:
                try:
                    bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
                except Exception:
                    pass
                self.stable_ticks[index] = 0
                self.last_pos[index] = None
                return

            if tick < SETTLE_MIN_TICKS:
                return

            is_stable = True
            if not bot.entity:
                is_stable = False

            if bot.entity:
                pos = bot.entity.position
                vel = bot.entity.velocity
                vel_mag = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                if vel_mag > STABLE_VEL_EPSILON:
                    is_stable = False

                current_pos = (pos.x, pos.y, pos.z)
                if self.last_pos[index] is not None:
                    lp = self.last_pos[index]
                    drift = math.sqrt(
                        (current_pos[0] - lp[0]) ** 2 +
                        (current_pos[1] - lp[1]) ** 2 +
                        (current_pos[2] - lp[2]) ** 2
                    )
                    if drift > STABLE_POS_EPSILON:
                        is_stable = False
                else:
                    is_stable = False
                self.last_pos[index] = current_pos
            else:
                is_stable = False

            if is_stable:
                self.stable_ticks[index] += 1
            else:
                self.stable_ticks[index] = 0

            if self.stable_ticks[index] >= SETTLE_STABLE_NEED:
                try:
                    bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
                except Exception:
                    pass

                tracker.is_settling = False
                d = tracker.get_distance_to_goal()
                tracker.max_distance_to_goal = d
                tracker.min_distance_to_goal = d
                tracker.is_active    = True
                tracker.tick_count   = 0
                tracker.wall_start   = time.time()
                self._set_state(index, STATE_ACTIVE)
                return

            if tick >= SETTLE_MAX_TICKS:
                try:
                    bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
                except Exception:
                    pass
                self._set_state(index, STATE_SETTLING)
                self.stable_ticks[index] = 0
                self.last_pos[index] = None
            return

        if state == STATE_ACTIVE:
            tracker.tick_count += 1

            if bot.entity and bot.entity.position.y < VOID_Y:
                self._end_run(index, 'DIED')
                return

            if tracker.tick_count >= TIMEOUT_TICKS:
                self._end_run(index, 'TIMEOUT')
                return

            if tracker.check_goal_reached():
                self._end_run(index, 'SUCCESS')
                return

            action = self.get_neural_action(index)
            self.execute_action(bot, action)
            return

        if state == STATE_POST_RUN:
            safe_stop_bot(bot)
            if tick >= POST_RUN_TICKS:
                self._on_post_run_complete(index)
            return

    # ------------------------------------------------------------------
    def _end_run(self, index, reason):
        tracker = self.trackers[index]
        bot     = tracker.bot

        if not tracker.is_active:
            return

        tracker.is_active    = False
        tracker.reached_goal = (reason == 'SUCCESS')
        tracker.run_duration = tracker.tick_count / 20.0

        safe_stop_bot(bot)

        fitness = tracker.update_fitness(TIMEOUT_SECONDS)

        pop_index = self.current_pop_index if SEQUENTIAL_MODE else index
        if fitness > self.all_time_best_score:
            self.all_time_best_score = fitness
            self.all_time_best_generation = self.ga.generation
            self.all_time_best_network = self.ga.population[pop_index].clone()
            print(f'  [Fallback] NEW HIGH SCORE!: {fitness:.2f} | saved.')
            self._save_best(net_index=pop_index)

        wall_elapsed = time.time() - tracker.wall_start if tracker.wall_start else 0
        elapsed_str  = f'{wall_elapsed:.1f}s'

        label = {'SUCCESS': 'PASS', 'TIMEOUT': 'TIMEOUT', 'DIED': 'FELL'}.get(reason, reason)
        prog   = tracker.best_forward_progress
        close  = tracker.closest_euclidean
        blocks = len(tracker.visited_blocks)
        facing = (tracker.facing_goal_sum / tracker.facing_goal_ticks) \
            if tracker.facing_goal_ticks > 0 else 0.0

        display_idx = self.current_pop_index if SEQUENTIAL_MODE else index
        trial_str = f' t{self.current_trial + 1}' if SEQUENTIAL_MODE else ''

        print(
            f'  Gen {self.ga.generation:2d} | Net {display_idx:2d}{trial_str} | {label:7s} | {elapsed_str:>6} | '
            f'prog:{prog:4.1f} euc:{close:4.1f} blk:{blocks:2d} face:{facing:+.2f} | '
            f'fitness: {fitness:8.2f}'
        )

        if SEQUENTIAL_MODE:
            self.trial_fitness.append(fitness)

        try:
            bot.chat('/kill @s')
        except Exception:
            pass

        self._set_state(index, STATE_POST_RUN)

    # ------------------------------------------------------------------
    def _on_post_run_complete(self, index):
        if SEQUENTIAL_MODE:
            self.current_trial += 1
            if self.current_trial < TRIALS_PER_NET:
                self._begin_sequential_run()
            else:
                avg_fitness = sum(self.trial_fitness) / len(self.trial_fitness)
                self.sequential_fitness[self.current_pop_index] = avg_fitness

                scores_str = ', '.join(f'{s:.1f}' for s in self.trial_fitness)
                print(f'         avg: {avg_fitness:.2f}  ({scores_str})')

                self.current_pop_index += 1
                self.current_trial = 0
                self.trial_fitness = []

                if self.current_pop_index >= POPULATION_SIZE:
                    self._finish_sequential_generation()
                else:
                    self._begin_sequential_run()
        else:
            self._set_state(index, STATE_IDLE)
            self.check_generation_complete()

    # ------------------------------------------------------------------
    # SEQUENTIAL MODE
    # ------------------------------------------------------------------
    def _begin_sequential_run(self):
        tracker = self.trackers[0]
        bot     = tracker.bot

        tracker.reset_for_run()
        safe_stop_bot(bot)

        try:
            if bot and bot.entity:
                bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
            elif bot:
                try:
                    bot.respawn()
                except Exception:
                    pass
        except Exception:
            print(f'  [Warn] Bot failed to teleport for net {self.current_pop_index}')

        self._set_state(0, STATE_TP_TO_SPAWN)

    def _finish_sequential_generation(self):
        self.generation_complete = True
        self._set_state(0, STATE_IDLE)
        print()

        self.ga.evaluate_population(self.sequential_fitness)

        best_score = max(self.sequential_fitness)
        best_idx   = self.sequential_fitness.index(best_score)
        avg_score  = sum(self.sequential_fitness) / POPULATION_SIZE
        successes  = sum(1 for s in self.sequential_fitness if s >= 500)

        print(
            f'  Results  | best: {best_score:.2f} | avg: {avg_score:.2f} | '
            f'passed: {successes}/{POPULATION_SIZE}'
        )

        if best_score >= self.all_time_best_score:
            if self.all_time_best_generation != self.ga.generation:
                self.all_time_best_score      = best_score
                self.all_time_best_generation = self.ga.generation
                self.all_time_best_network    = self.ga.population[best_idx].clone()
                self._save_best(net_index=best_idx)
                print(
                    f'  NEW BEST  | score: {best_score:.2f} | '
                    f'generation: {self.ga.generation} | saved.'
                )

        if self.auto_evolve:
            self.ga.evolve()
            if self.pending_pause:
                self._enter_pause()
            else:
                self.start_new_run()

    # ------------------------------------------------------------------
    # PARALLEL MODE
    # ------------------------------------------------------------------
    def start_new_run(self):
        print(f'\n--- Generation {self.ga.generation} ---')
        self.generation_complete = False

        if SEQUENTIAL_MODE:
            self.current_pop_index = 0
            self.current_trial = 0
            self.trial_fitness = []
            self.sequential_fitness = [0.0] * POPULATION_SIZE
            self._begin_sequential_run()
            return

        for tracker in self.trackers:
            tracker.reset_for_run()
            safe_stop_bot(tracker.bot)

        for i, tracker in enumerate(self.trackers):
            bot = tracker.bot
            if bot and bot.entity:
                try:
                    bot.chat(_tp_cmd(SPAWN_POS['x'], SPAWN_POS['y'], SPAWN_POS['z']))
                except Exception:
                    print(f'  [Warn] Bot {i} failed to teleport.')
            else:
                try:
                    bot.respawn()
                except Exception:
                    pass
            self._set_state(i, STATE_TP_TO_SPAWN)

    # ------------------------------------------------------------------
    def get_neural_action(self, bot_index):
        tracker = self.trackers[bot_index]
        sensor  = tracker.get_sensor_input_vector(TIMEOUT_TICKS)

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

        pop_index = self.current_pop_index if SEQUENTIAL_MODE else bot_index
        return self.ga.population[pop_index].forward(inputs)

    def execute_action(self, bot, action):
        if not bot.entity:
            return

        bot.setControlState('left',    bool(action[0]))
        bot.setControlState('right',   bool(action[1]))
        bot.setControlState('jump',    bool(action[2]))
        bot.setControlState('forward', True)
        bot.setControlState('sprint',  True)

        yaw_delta = action[3] * YAW_RATE
        new_yaw = bot.entity.yaw + yaw_delta
        try:
            bot.look(new_yaw, bot.entity.pitch, True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def check_generation_complete(self):
        if self.generation_complete:
            return True

        for i in range(len(self.trackers)):
            if self.bot_state[i] != STATE_IDLE:
                return False

        self.generation_complete = True
        print()

        fitness_scores = [t.fitness_score for t in self.trackers]
        self.ga.evaluate_population(fitness_scores)

        best_score = max(fitness_scores)
        best_idx   = fitness_scores.index(best_score)
        avg_score  = sum(fitness_scores) / len(fitness_scores)
        successes  = sum(1 for t in self.trackers if t.reached_goal)

        print(
            f'  Results  | best: {best_score:.2f} | avg: {avg_score:.2f} | '
            f'passed: {successes}/{BOT_COUNT}'
        )

        if best_score >= self.all_time_best_score:
            if self.all_time_best_generation != self.ga.generation:
                self.all_time_best_score      = best_score
                self.all_time_best_generation = self.ga.generation
                self.all_time_best_network    = self.ga.population[best_idx].clone()
                self._save_best(net_index=best_idx)
                print(
                    f'  NEW BEST  | score: {best_score:.2f} | '
                    f'generation: {self.ga.generation} | saved.'
                )

        if self.auto_evolve:
            self.ga.evolve()
            if self.pending_pause:
                self._enter_pause()
            else:
                self.start_new_run()

        return True

    # ------------------------------------------------------------------
    # PAUSE / NEW-GOAL COORDINATION (called between threads)
    # ------------------------------------------------------------------
    def request_pause(self):
        """Called by CLI thread. Pause will take effect after the current
        generation completes and evolves."""
        if self.paused_for_goal:
            print('  Already paused. Use `resume` or supply new goal coords.')
            return
        if self.pending_pause:
            print('  Pause already queued.')
            return
        self.pending_pause = True
        print('  [Pause] Queued - will take effect after current generation evolves.')

    def _enter_pause(self):
        """Called by the physics thread just after ga.evolve() finishes."""
        self.pending_pause = False
        self.paused_for_goal = True
        self.auto_evolve = False
        print('\n' + '=' * 60)
        print(f'  PAUSED after Gen {self.ga.generation - 1} '
              f'(next gen is {self.ga.generation}).')
        print(f'  All {len(self.trackers)} networks preserved in gene pool.')
        print('  Type `newgoal <x> <y> <z>` or `resume` in the CLI.')
        print('=' * 60 + '\n')

    def apply_new_goal(self, x, y, z):
        """Update goal coords on every tracker and resume training."""
        global GOAL_POS
        GOAL_POS = {'x': x, 'y': y, 'z': z}
        for t in self.trackers:
            t.set_target_pos(GOAL_POS)
        print(f'  [NewGoal] Goal updated to ({x}, {y}, {z}) on all {len(self.trackers)} trackers.')

    def resume(self):
        """Resume training (with whatever GOAL_POS is currently set)."""
        if not self.paused_for_goal:
            self.auto_evolve = True
            print('  Evolution re-enabled (was not paused for goal).')
            return
        self.paused_for_goal = False
        self.auto_evolve = True
        print(f'  [Resume] Continuing training on goal ({GOAL_POS["x"]}, '
              f'{GOAL_POS["y"]}, {GOAL_POS["z"]}).')
        self.start_new_run()

    # ------------------------------------------------------------------
    def _save_best(self, net_index='unknown'):
        if not self.all_time_best_network:
            return

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fn = f'{timestamp}_gen{self.all_time_best_generation}_net{net_index}.json'

        with open(fn, 'w') as f:
            json.dump(
                {
                    'timestamp':  timestamp,
                    'generation': self.all_time_best_generation,
                    'net_index':  net_index,
                    'fitness':    self.all_time_best_score,
                    'network':    self.all_time_best_network.save_to_dict(),
                },
                f,
                indent=2,
            )
        print(f'  Saved     | {fn}')

    def save_best(self):
        self._save_best(net_index='manual')


# ======================================================================
# CLI THREAD
# ======================================================================
def terminal_input_loop(gen):
    print('\nCommands:')
    print('  pause                -> pause after current gen, then prompt for new goal')
    print('  newgoal <x> <y> <z>  -> set goal coords while paused (or queue a pause+prompt)')
    print('  resume               -> resume training with current goal')
    print('  save                 -> save current best network')
    print('  quit                 -> save and exit\n')

    while True:
        try:
            raw = input()
        except (EOFError, KeyboardInterrupt):
            return
        if not raw:
            continue
        parts = raw.strip().split()
        cmd = parts[0].lower()

        if cmd == 'pause':
            # Request pause, then wait for the physics thread to enter
            # paused_for_goal state, then prompt for coords in THIS thread.
            gen.request_pause()
            while not gen.paused_for_goal:
                time.sleep(0.25)

            print('\n  Enter new goal coordinates (or press Enter on any field to keep current):')
            try:
                cx = input(f'    x [{GOAL_POS["x"]}]: ').strip()
                cy = input(f'    y [{GOAL_POS["y"]}]: ').strip()
                cz = input(f'    z [{GOAL_POS["z"]}]: ').strip()
                nx = float(cx) if cx else GOAL_POS['x']
                ny = float(cy) if cy else GOAL_POS['y']
                nz = float(cz) if cz else GOAL_POS['z']
            except ValueError:
                print('  [Error] Invalid number. Goal unchanged. Type `resume` to continue.')
                continue
            except (EOFError, KeyboardInterrupt):
                print('\n  [Cancel] Goal unchanged. Type `resume` to continue.')
                continue

            gen.apply_new_goal(nx, ny, nz)
            print('  Type `resume` when your new course is built and ready.')

        elif cmd == 'newgoal':
            # Inline form: `newgoal 6.5 -63 0.5`
            if len(parts) != 4:
                print('  Usage: newgoal <x> <y> <z>')
                continue
            try:
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                print('  [Error] Invalid numbers.')
                continue

            if not gen.paused_for_goal:
                # Queue a pause and apply coords when pause lands
                gen.request_pause()
                while not gen.paused_for_goal:
                    time.sleep(0.25)
            gen.apply_new_goal(nx, ny, nz)
            print('  Type `resume` when your new course is built and ready.')

        elif cmd == 'resume':
            gen.resume()

        elif cmd == 'save':
            gen.save_best()

        elif cmd == 'quit':
            gen.save_best()
            sys.exit(0)

        else:
            print(f'  Unknown command: {cmd}')


def main():
    mode_str = 'SEQUENTIAL (1 bot)' if SEQUENTIAL_MODE else f'PARALLEL ({POPULATION_SIZE} bots)'
    trials_str = f' | Trials/net: {TRIALS_PER_NET}' if SEQUENTIAL_MODE else ''
    print('ECE 556 - Minecraft Neuromorphic Parkour Trainer')
    print(f'Mode: {mode_str} | Population: {POPULATION_SIZE}{trials_str}')
    print(f'Inputs: {INPUT_SIZE} | Hidden: {HIDDEN_LAYER_SIZE} '
          f'| Outputs: {OUTPUT_SIZE} | Timeout: {TIMEOUT_SECONDS}s')
    print(f'YAW_RATE={YAW_RATE} rad/tick | Goal: ({GOAL_POS["x"]}, {GOAL_POS["y"]}, {GOAL_POS["z"]})')
    print(f'Settle: min {SETTLE_MIN_TICKS}t + {SETTLE_STABLE_NEED}t stable | All delays tick-driven')
    print()

    gen = BotGeneration()
    for i in range(BOT_COUNT):
        gen.create_bot(i)
        time.sleep(0.1)

    globals()['gen'] = gen
    threading.Thread(target=terminal_input_loop, args=(gen,), daemon=True).start()

    print(f'\nWaiting {STARTUP_TICKS} ticks before first generation...')


if __name__ == '__main__':
    main()