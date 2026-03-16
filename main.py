# GMU ECE 556 PROJECT
# ERIC ZIPOR 3/14/2026

import math
import random
import threading
import sys
import time
import json
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== CONFIGURATION ==========
BOT_COUNT    = 7
TEAM_NAME    = "bots"
ADMIN_PLAYER = "EricZoop"

# Coordinates
SPAWN_POS     = {'x': 0.5,   'y': -63, 'z': 0.5}
GOAL_POS      = {'x': 5.5,   'y': -63, 'z': 5.5}
PURGATORY_POS = {'x': -14.5, 'y': -63, 'z': 0.5}

# Training Parameters
TIMEOUT_SECONDS  = 15
RAY_MAX_DIST     = 7
VOID_Y           = -65
SPAWN_RADIUS     = 2.0
TELEPORT_DELAY   = 0.5
ACTIVATE_DELAY   = 2.5

# Neural Network Architecture — UPDATED for v2
#   19 rays + 3 pos + 3 vel + 2 orient + 3 goal + 3 scalars = 33 inputs
#   4 outputs: strafe_left, strafe_right, jump, yaw_delta (continuous)
INPUT_SIZE        = 33
HIDDEN_LAYER_SIZE = 32
OUTPUT_SIZE       = 4     # left, right, jump, yaw_delta

# Yaw control parameters
YAW_RATE          = 0.15  # max radians per tick (~8.6° per tick, ~170°/s at 20 TPS)

# Genetic Algorithm Parameters
MUTATION_RATE     = 0.1
MUTATION_STRENGTH = 0.15
ELITE_COUNT       = 2


def safe_clear_controls(bot):
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


def _dist2d(pos, target):
    dx = pos.x - target['x']
    dz = pos.z - target['z']
    return (dx * dx + dz * dz) ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
class BotGeneration:
    def __init__(self):
        self.trackers = []
        self.generation_complete = False
        self.ga = GeneticAlgorithm(
            population_size   = BOT_COUNT,
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

    # ------------------------------------------------------------------
    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host':     '192.168.1.118', # localhost
            'port':     25565,
            'username': f'Bot_{index}',
            'version':  '1.19.4',
        })

        tracker = BotTracker(bot, GOAL_POS, ray_max_dist=RAY_MAX_DIST, spawn_pos=SPAWN_POS)
        tracker.bot_index = index
        bot._ready_for_physics = False

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
                    bot.chat(
                        f'/setworldspawn {PURGATORY_POS["x"]} '
                        f'{PURGATORY_POS["y"]} {PURGATORY_POS["z"]}'
                    )
                    bot.chat(
                        f'/tp @a[name=!{ADMIN_PLAYER},team={TEAM_NAME}] '
                        f'{PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}'
                    )
            except Exception as e:
                print(f'[Bot {index}] Spawn error: {e}')

        @On(bot, 'physicsTick')
        def handle_physics(*args):
            if not tracker.is_active:
                safe_clear_controls(bot)
                return

            if time.time() < tracker.start_time:
                safe_clear_controls(bot)
                return

            tracker.tick_count += 1

            # 1. Void check
            if bot.entity and bot.entity.position.y < VOID_Y:
                self.end_bot_run(index, 'DIED')
                return

            # 2. Timeout check
            if (time.time() - tracker.start_time) > TIMEOUT_SECONDS:
                self.end_bot_run(index, 'TIMEOUT')
                return

            # 3. Goal check
            if tracker.check_goal_reached():
                self.end_bot_run(index, 'SUCCESS')
                return

            # 4. Neural network action — every tick for consistency
            #    (staggering by index caused identical networks to score
            #    differently when their index changed between generations)
            action = self.get_neural_action(index)
            self.execute_action(bot, action)

        self.trackers.append(tracker)

    # ------------------------------------------------------------------
    def end_bot_run(self, index, reason):
        tracker = self.trackers[index]
        bot     = tracker.bot

        if not tracker.is_active:
            return

        tracker.is_active    = False
        tracker.reached_goal = (reason == 'SUCCESS')
        tracker.run_duration = time.time() - tracker.start_time

        safe_clear_controls(bot)

        if reason != 'DIED':
            try:
                if bot.entity:
                    bot.chat(
                        f'/tp @s {PURGATORY_POS["x"]} '
                        f'{PURGATORY_POS["y"]} {PURGATORY_POS["z"]}'
                    )
            except Exception:
                pass

        fitness = tracker.update_fitness(TIMEOUT_SECONDS)
        elapsed = f'{tracker.run_duration:.1f}s'

        label = {
            'SUCCESS': 'PASS', 'TIMEOUT': 'TIMEOUT', 'DIED': 'FELL'
        }.get(reason, reason)
        print(f'  Bot {index:2d} | {label:7s} | {elapsed:>6} | fitness: {fitness:8.2f}')

        self.check_generation_complete()

    # ------------------------------------------------------------------
    def start_new_run(self):
        print(f'\n--- Generation {self.ga.generation} ---')
        self.generation_complete = False

        for tracker in self.trackers:
            tracker.reset_for_run()
            tracker.bot._ready_for_physics = False
            safe_clear_controls(tracker.bot)

        def teleport_bots():
            for tracker in self.trackers:
                if tracker.bot and tracker.bot.entity:
                    try:
                        tracker.bot.chat(
                            f'/tp @s {SPAWN_POS["x"]} '
                            f'{SPAWN_POS["y"]} {SPAWN_POS["z"]}'
                        )
                    except Exception:
                        print(f'  [Warn] Bot {tracker.bot_index} failed to teleport.')

        def enable_bots():
            future_start = time.time() + 1.5
            active_count = 0
            skipped      = 0

            for tracker in self.trackers:
                if not (tracker.bot and tracker.bot.entity):
                    try:
                        tracker.bot.respawn()
                    except Exception:
                        pass
                    skipped += 1
                    continue

                dist_to_spawn = _dist2d(tracker.bot.entity.position, SPAWN_POS)
                if dist_to_spawn > SPAWN_RADIUS:
                    try:
                        tracker.bot.chat(
                            f'/tp @s {SPAWN_POS["x"]} '
                            f'{SPAWN_POS["y"]} {SPAWN_POS["z"]}'
                        )
                    except Exception:
                        pass
                    print(
                        f'  [Skip] Bot {tracker.bot_index} not at spawn '
                        f'(dist={dist_to_spawn:.1f}), re-teleporting.'
                    )
                    skipped += 1
                    continue

                d = tracker.get_distance_to_goal()
                tracker.max_distance_to_goal = d
                tracker.min_distance_to_goal = d
                tracker.start_time           = future_start
                tracker.is_active            = True
                tracker.bot._ready_for_physics = True
                active_count += 1

            msg = f'  {active_count}/{BOT_COUNT} bots active | timeout: {TIMEOUT_SECONDS}s'
            if skipped:
                msg += f'  ({skipped} skipped - not at spawn)'
            print(msg)

            if active_count == 0:
                print('  [Warn] No bots activated - retrying in 3 s...')
                threading.Timer(3.0, self.start_new_run).start()
                return

            self._status_timer()

        threading.Timer(TELEPORT_DELAY, teleport_bots).start()
        threading.Timer(ACTIVATE_DELAY, enable_bots).start()

    # ------------------------------------------------------------------
    def _status_timer(self):
        def tick():
            if self.generation_complete:
                return
            active = sum(1 for t in self.trackers if t.is_active)
            if active > 0:
                ref     = self.trackers[0].start_time
                elapsed = max(0.0, time.time() - ref) if ref > 0 else 0.0
                print(
                    f'  [{elapsed:5.1f}s] {active} bots still running...',
                    end='\r',
                )
                threading.Timer(1.0, tick).start()

        threading.Timer(2.0, tick).start()

    # ------------------------------------------------------------------
    def get_neural_action(self, bot_index):
        tracker     = self.trackers[bot_index]
        sensor_data = tracker.get_sensor_input_vector(TIMEOUT_SECONDS)

        inputs = []
        inputs.extend(sensor_data['rays'])             # 19
        inputs.extend(sensor_data['position'])         # 3
        inputs.extend(sensor_data['velocity'])         # 3
        inputs.extend(sensor_data['orientation'])      # 2
        inputs.extend(sensor_data['goal_direction'])   # 3
        inputs.append(sensor_data['on_ground'])        # 1
        inputs.append(sensor_data['distance_to_goal']) # 1
        inputs.append(sensor_data['time_remaining'])   # 1
        # total: 33

        return self.ga.population[bot_index].forward(inputs)

    def execute_action(self, bot, action):
        """
        action[0]: strafe left  (bool)
        action[1]: strafe right (bool)
        action[2]: jump         (bool)
        action[3]: yaw delta    (continuous, mapped from sigmoid 0..1 to -YAW_RATE..+YAW_RATE)
        """
        if not bot.entity:
            return

        bot.setControlState('left',    bool(action[0]))
        bot.setControlState('right',   bool(action[1]))
        bot.setControlState('jump',    bool(action[2]))
        bot.setControlState('forward', True)
        bot.setControlState('sprint',  True)

        # Continuous yaw control
        # action[3] is a sigmoid value in [0, 1]; map to [-YAW_RATE, +YAW_RATE]
        yaw_delta = (action[3] - 0.5) * 2.0 * YAW_RATE
        new_yaw = bot.entity.yaw + yaw_delta
        try:
            bot.look(new_yaw, bot.entity.pitch, True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def check_generation_complete(self):
        if self.generation_complete:
            return True
        if any(t.is_active for t in self.trackers):
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

        if best_score > self.all_time_best_score:
            self.all_time_best_score      = best_score
            self.all_time_best_generation = self.ga.generation
            self.all_time_best_network    = self.ga.population[best_idx].clone()
            self._save_best()
            print(
                f'  NEW BEST  | score: {best_score:.2f} | '
                f'generation: {self.ga.generation} | saved.'
            )

        if self.auto_evolve:
            self.ga.evolve()
            threading.Timer(2.0, self.start_new_run).start()

        return True

    # ------------------------------------------------------------------
    def _save_best(self):
        if not self.all_time_best_network:
            return
        fn = f'best_gen_{self.all_time_best_generation}.json'
        with open(fn, 'w') as f:
            json.dump(
                {
                    'generation': self.all_time_best_generation,
                    'fitness':    self.all_time_best_score,
                    'network':    self.all_time_best_network.save_to_dict(),
                },
                f,
                indent=2,
            )
        print(f'  Saved     | {fn}')

    def save_best(self):
        self._save_best()


# ──────────────────────────────────────────────────────────────────────────────
def terminal_input_loop(gen):
    print('\nCommands: pause | resume | save | quit\n')
    while True:
        try:
            cmd = input().strip().lower()
            if   cmd == 'pause':  gen.auto_evolve = False; print('  Evolution paused.')
            elif cmd == 'resume': gen.auto_evolve = True;  print('  Evolution resumed.')
            elif cmd == 'save':   gen.save_best()
            elif cmd == 'quit':
                gen.save_best()
                sys.exit(0)
        except Exception:
            break


def main():
    print('ECE 556 - Minecraft Neuromorphic Parkour Trainer')
    print(
        f'Bots: {BOT_COUNT} | Inputs: {INPUT_SIZE} | Hidden: {HIDDEN_LAYER_SIZE} '
        f'| Outputs: {OUTPUT_SIZE} | Timeout: {TIMEOUT_SECONDS}s'
    )
    print()

    gen = BotGeneration()
    for i in range(BOT_COUNT):
        gen.create_bot(i)
        time.sleep(0.1)

    globals()['gen'] = gen
    threading.Thread(target=terminal_input_loop, args=(gen,), daemon=True).start()

    print(f'\nStarting first gen in 5 seconds...')
    threading.Timer(5.0, gen.start_new_run).start()


if __name__ == '__main__':
    main()