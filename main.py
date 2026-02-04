import random
import threading
import sys
import time
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

### MAKE SURE INPUT_SIZE IS CONSISTENT

# ========== CONFIGURATION ==========
BOT_COUNT = 6
TEAM_NAME = "bots"
SPAWN_POS = {'x': 0.5, 'y': 0, 'z': 0.5}     
PURGATORY_POS = {'x': -15, 'y': 0, 'z': 0.5} 
GOAL_POS = {'x': 10.5, 'y': 0, 'z': 0.5}       

# --- Training Parameters ---
TIMEOUT_SECONDS = 4       
DEATH_Y = -1               
RAY_MAX_DIST = 10         

# --- Neural Network Architecture ---
HIDDEN_LAYER_SIZE = 32     
INPUT_SIZE = 27            # UPDATED: Removed 'is_sprinting' input

# --- Genetic Algorithm Parameters ---
MUTATION_RATE = 0.1       
MUTATION_STRENGTH = 0.2    
ELITE_COUNT = 2            
AUTO_SAVE_INTERVAL = 10    

class BotGeneration:
    def __init__(self):
        self.trackers = []
        self.generation_complete = False
        self.ga = GeneticAlgorithm(
            population_size=BOT_COUNT,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_LAYER_SIZE,
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH,
            elite_count=ELITE_COUNT
        )
        self.auto_evolve = True
        
    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host': 'localhost',
            'port': 51376,
            'username': f"Bot_{index}",
            'version': '1.19.4'
        })
        
        tracker = BotTracker(bot, GOAL_POS, ray_max_dist=RAY_MAX_DIST)
        tracker.bot_index = index
        tracker.run_duration = 0.0
        
        # Setup flags
        bot._setup_complete = False
        bot._purgatory_tp_done = False
        bot._ready_tick_counter = 0
        
        @Once(bot, 'spawn')
        def on_spawn(*args):
            bot.chat(f'/team join {TEAM_NAME}')

        @On(bot, 'move')
        def on_move(*args):
            if not bot._purgatory_tp_done and bot.entity and bot.entity.position:
                bot._purgatory_tp_done = True
                bot.chat(f'/tp {PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}')

        @On(bot, 'physicsTick')
        def handle_physics(*args):
            if bot._purgatory_tp_done and not bot._setup_complete:
                bot._ready_tick_counter += 1
                if bot._ready_tick_counter >= 20:
                    bot._setup_complete = True
                return
            
            if not bot._setup_complete:
                return
            
            if not tracker.is_active or tracker.reached_goal:
                return
            
            tracker.tick_count += 1

            if bot.entity and bot.entity.position.y < DEATH_Y:
                self.move_to_purgatory(index, "FELL")
                return
            
            if tracker.check_goal_reached():
                tracker.reached_goal = True
                self.move_to_purgatory(index, "SUCCESS")
                return
            
            if (time.time() - tracker.start_time) > TIMEOUT_SECONDS:
                self.move_to_purgatory(index, "TIMEOUT")
                return
            
            # Execute Action
            if tracker.tick_count % 2 == 0:
                action = self.get_neural_action(index)
                self.execute_action(bot, action)
        
        self.trackers.append(tracker)

    def move_to_purgatory(self, index, reason):
        tracker = self.trackers[index]
        bot = tracker.bot
        
        if hasattr(tracker, 'start_time'):
            tracker.run_duration = time.time() - tracker.start_time
        else:
            tracker.run_duration = 0.0

        tracker.is_active = False
        
        bot.clearControlStates()
        bot.chat(f'/tp {PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}')
        
        fitness = tracker.update_fitness()
        self.check_generation_complete()

    def start_new_run(self):
        print(f"\n{'='*60}")
        print(f"Generation {self.ga.generation} - Starting") 
        print(f"{'='*60}")
        
        self.generation_complete = False
        
        for i, tracker in enumerate(self.trackers):
            tracker.reset_for_run()
            tracker.run_duration = 0.0
            tracker.bot.chat(f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}')

    def get_neural_action(self, bot_index):
        tracker = self.trackers[bot_index]
        sensor_data = tracker.get_sensor_input_vector()
        
        # Flatten inputs (Total 23)
        inputs = []
        inputs.extend(sensor_data['rays'])           # 11 values
        inputs.extend(sensor_data['position'])       # 3 values
        inputs.extend(sensor_data['velocity'])       # 3 values
        inputs.extend(sensor_data['orientation'])    # 2 values
        inputs.extend(sensor_data['goal_direction']) # 3 values
        inputs.append(sensor_data['on_ground'])      # 1 value
        # REMOVED: inputs.append(sensor_data['is_sprinting'])
        
        network = self.ga.population[bot_index]
        action = network.forward(inputs)
        return action
    
    def execute_action(self, bot, action):
        """
        UPDATED: 4-Dimensional Action Vector
        [0]: Forward
        [1]: Left
        [2]: Right
        [3]: Jump
        """
        if not bot.entity: 
            return 
        
        # Neural Decisions
        bot.setControlState('forward', bool(action[0]))
        bot.setControlState('left', bool(action[1]))
        bot.setControlState('right', bool(action[2]))
        bot.setControlState('jump', bool(action[3]))
        
        # HARDCODED CONSTANTS
        bot.setControlState('sprint', True)  # Always Sprint
        bot.setControlState('back', False)   # Never Back
        bot.setControlState('sneak', False)  # Never Sneak
    
    def check_generation_complete(self):
        all_finished = all(not t.is_active for t in self.trackers)
        
        if all_finished and not self.generation_complete:
            self.generation_complete = True
            
            fitness_scores = [t.fitness_score for t in self.trackers]
            self.ga.evaluate_population(fitness_scores)
            
            stats = self.ga.get_statistics()
            print(f"\n{'='*60}")
            print(f"Generation {self.ga.generation} - Complete")
            print(f"{'='*60}")
            print(f"Best:  {stats['best_fitness']:8.2f}")
            print(f"Avg:   {stats['avg_fitness']:8.2f}")
            
            sorted_bots = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
            for rank, (idx, fitness) in enumerate(sorted_bots, 1):
                status = "✔" if self.trackers[idx].reached_goal else "✘"
                duration = self.trackers[idx].run_duration
                print(f"#{rank} Bot_{idx}: {fitness:7.2f} {status} ({duration:5.2f}s)")
            
            if self.ga.generation % AUTO_SAVE_INTERVAL == 0:
                self.save_checkpoint()
            
            if self.auto_evolve:
                self.ga.evolve()
                print(f"\n[EVOLVING] Creating generation {self.ga.generation}...") 
                threading.Timer(3.0, self.start_new_run).start()
            else:
                print("\n[PAUSED] Auto-evolve disabled. Type 'evolve' to continue.")
        
        return all_finished
    
    def save_checkpoint(self):
        filename = f'checkpoint_gen_{self.ga.generation}.json'
        self.ga.save_to_file(filename)
        print(f"[SAVE] {filename}")
    
    def save_best(self):
        filename = f'best_network_gen_{self.ga.generation}.json'
        self.ga.save_best_network(filename)
        best_net, best_fitness, best_idx = self.ga.get_best_network()
        print(f"[SAVE] {filename}")
        print(f"       Fitness: {best_fitness:.2f} (Bot_{best_idx})")
        return filename

def print_help():
    print("\n" + "="*60)
    print("COMMANDS:")
    print("="*60)
    print("  start      - Start a new generation run")
    print("  evolve     - Evolve and start next generation")
    print("  pause      - Disable auto-evolution")
    print("  resume     - Enable auto-evolution")
    print("  save       - Save current checkpoint")
    print("  save_best  - Save best network to file")
    print("  stats      - Show current statistics")
    print("  help       - Show this help")
    print("  quit       - Exit program")
    print("="*60)

def terminal_input_loop(gen):
    print("\n[TERMINAL] Type 'help' for commands")
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if cmd == 'start':
                gen.start_new_run()
            elif cmd == 'evolve':
                gen.ga.evolve()
                print(f"[EVOLVE] Created generation {gen.ga.generation}")
                gen.start_new_run()
            elif cmd == 'pause':
                gen.auto_evolve = False
                print("[PAUSE] Auto-evolution disabled")
            elif cmd == 'resume':
                gen.auto_evolve = True
                print("[RESUME] Auto-evolution enabled")
            elif cmd == 'save':
                gen.save_checkpoint()
            elif cmd == 'save_best':
                gen.save_best()
            elif cmd == 'stats':
                stats = gen.ga.get_statistics()
                print(f"\nGeneration {gen.ga.generation}:")
                print(f"  Best:  {stats['best_fitness']:.2f}")
                print(f"  Avg:   {stats['avg_fitness']:.2f}")
                print(f"  Worst: {stats['worst_fitness']:.2f}")
                print(f"  Mutation Rate: {stats['mutation_rate']:.3f}")
            elif cmd == 'help':
                print_help()
            elif cmd == 'quit':
                print("[EXIT] Saving final checkpoint...")
                gen.save_checkpoint()
                gen.save_best()
                print("[EXIT] Goodbye!")
                sys.exit(0)
            else:
                print(f"Unknown: '{cmd}'. Type 'help' for commands.")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted. Saving...")
            gen.save_checkpoint()
            gen.save_best()
            sys.exit(0)

def main():
    print("\n" + "="*60)
    print("NEUROMORPHIC MINECRAFT PARKOUR TRAINER")
    print("="*60)
    print(f"Population:       {BOT_COUNT}")
    print(f"Hidden Layer:     {HIDDEN_LAYER_SIZE} neurons")
    print(f"Input Size:       {INPUT_SIZE}")
    print(f"Mutation Rate:    {MUTATION_RATE}")
    print(f"Mutation Strength: {MUTATION_STRENGTH}")
    print(f"Elite Count:      {ELITE_COUNT}")
    print(f"Timeout:          {TIMEOUT_SECONDS}s")
    print("="*60)
    
    gen = BotGeneration()
    
    print(f"\n[INIT] Creating {BOT_COUNT} bots...")
    for i in range(BOT_COUNT):
        gen.create_bot(i)
    
    globals()['gen'] = gen
    
    terminal_thread = threading.Thread(target=terminal_input_loop, args=(gen,), daemon=True)
    terminal_thread.start()
    
    print(f"\n[AUTO] Starting first generation in 5 seconds...")
    threading.Timer(5.0, gen.start_new_run).start()

if __name__ == "__main__":
    main()