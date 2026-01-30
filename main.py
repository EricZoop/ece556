import random
import threading
import sys
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== CONFIGURATION ==========
BOT_COUNT = 5
TEAM_NAME = "bots"
SPAWN_POS = {'x': 0.5, 'y': 0, 'z': 0.5}   # Where they start the course
PURGATORY_POS = {'x': -15, 'y': 0, 'z': 0.5} # Where they wait after failing
GOAL_POS = {'x': 10, 'y': 0, 'z': 0.5}

# --- Training Parameters ---
TIMEOUT_SECONDS = 20
MAX_TICKS = TIMEOUT_SECONDS * 20
DEATH_Y = -10  
SCAN_RADIUS = 2

# --- Genetic Algorithm Parameters ---
HIDDEN_LAYER_SIZE = 64
MUTATION_RATE = 0.15        # 15% chance to mutate each weight
MUTATION_STRENGTH = 0.3     # Standard deviation of mutation
ELITE_COUNT = 1             # Keep best performer unchanged
AUTO_SAVE_INTERVAL = 10     # Save every N generations

# Calculate input size: voxel_grid + other sensors
# voxel_grid: (2*SCAN_RADIUS+1)^3 = (2*5+1)^3 = 11^3 = 1331
# position: 3, velocity: 3, orientation: 2, goal_direction: 3, goal_distance: 1, block_below: 1, is_sprinting: 1
INPUT_SIZE = (5 * SCAN_RADIUS + 1) ** 3 + 3 + 3 + 2 + 3 + 1 + 1 + 1  # = 1345

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
        self.auto_evolve = True  # Can be toggled via terminal
        
    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host': 'localhost',
            'port': 51376,
            'username': f"Bot_{index}",
            'version': '1.19.4'
        })
        
        tracker = BotTracker(bot, GOAL_POS, scan_radius=SCAN_RADIUS)
        tracker.bot_index = index
        
        # Flag to track if initial setup is complete
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
            # Handle initial setup delay
            if bot._purgatory_tp_done and not bot._setup_complete:
                bot._ready_tick_counter += 1
                if bot._ready_tick_counter >= 20:
                    bot._setup_complete = True
                return
            
            if not bot._setup_complete:
                return
            
            # Only process if bot is active in a run
            if not tracker.is_active or tracker.reached_goal:
                return
            
            tracker.tick_count += 1

            # 1. Death Check
            if bot.entity and bot.entity.position.y < DEATH_Y:
                self.move_to_purgatory(index, "FELL")
                return
            
            # 2. Goal Check
            if tracker.check_goal_reached():
                tracker.reached_goal = True
                self.move_to_purgatory(index, "SUCCESS")
                return
            
            # 3. Timeout Check
            if tracker.tick_count >= MAX_TICKS:
                self.move_to_purgatory(index, "TIMEOUT")
                return
            
            # 4. Execute Neural Action every 2 ticks
            if tracker.tick_count % 2 == 0:
                action = self.get_neural_action(index)
                self.execute_action(bot, action)
        
        self.trackers.append(tracker)

    def move_to_purgatory(self, index, reason):
        """Move bot to purgatory after completing their run"""
        tracker = self.trackers[index]
        bot = tracker.bot
        tracker.is_active = False
        
        bot.clearControlStates()
        bot.chat(f'/tp {PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}')
        
        # Calculate final fitness
        fitness = tracker.update_fitness()
        
        # Check if all bots are done
        self.check_generation_complete()

    def start_new_run(self):
        """Teleport all bots to spawn and begin the course"""
        print(f"\n{'='*60}")
        print(f"Generation {self.ga.generation + 1} - Starting")
        print(f"{'='*60}")
        
        self.generation_complete = False
        
        for i, tracker in enumerate(self.trackers):
            tracker.reset_for_run()
            tracker.bot.chat(f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}')

    def get_neural_action(self, bot_index):
        """Get action from neural network based on sensor input"""
        tracker = self.trackers[bot_index]
        sensor_data = tracker.get_sensor_input_vector()
        
        # Flatten all inputs into a single vector
        inputs = []
        inputs.extend(sensor_data['voxel_grid'])
        inputs.extend(sensor_data['position'])
        inputs.extend(sensor_data['velocity'])
        inputs.extend(sensor_data['orientation'])
        inputs.extend(sensor_data['goal_direction'])
        inputs.append(sensor_data['goal_distance'])
        inputs.append(sensor_data['block_below'])
        inputs.append(sensor_data['is_sprinting'])
        
        # Get neural network for this bot
        network = self.ga.population[bot_index]
        
        # Get action from network
        action = network.forward(inputs)
        
        return action
    
    def execute_action(self, bot, action):
        """Execute the action vector"""
        if not bot.entity: 
            return 
        
        bot.setControlState('forward', bool(action[0]))
        bot.setControlState('back', bool(action[1]))
        bot.setControlState('left', bool(action[2]))
        bot.setControlState('right', bool(action[3]))
        bot.setControlState('jump', bool(action[4]))
        bot.setControlState('sprint', bool(action[5]))

    def check_generation_complete(self):
        """Check if all bots finished and handle evolution"""
        all_finished = all(not t.is_active for t in self.trackers)
        
        if all_finished and not self.generation_complete:
            self.generation_complete = True
            
            # Collect fitness scores
            fitness_scores = [t.fitness_score for t in self.trackers]
            
            # Update GA with fitness scores
            self.ga.evaluate_population(fitness_scores)
            
            # Print statistics
            stats = self.ga.get_statistics()
            print(f"\n{'='*60}")
            print(f"Generation {self.ga.generation} - Complete")
            print(f"{'='*60}")
            print(f"Best Fitness:  {stats['best_fitness']:8.2f}")
            print(f"Avg Fitness:   {stats['avg_fitness']:8.2f}")
            print(f"Worst Fitness: {stats['worst_fitness']:8.2f}")
            
            # Show individual results
            sorted_bots = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
            for rank, (idx, fitness) in enumerate(sorted_bots, 1):
                status = "✓" if self.trackers[idx].reached_goal else "✗"
                print(f"  #{rank} Bot_{idx}: {status} {fitness:7.2f}")
            
            # Auto-save periodically
            if self.ga.generation % AUTO_SAVE_INTERVAL == 0:
                self.save_checkpoint()
            
            # Evolve to next generation
            if self.auto_evolve:
                self.ga.evolve()
                print(f"\n[EVOLVING] Creating generation {self.ga.generation + 1}...")
                threading.Timer(3.0, self.start_new_run).start()
            else:
                print("\n[PAUSED] Auto-evolve disabled. Type 'evolve' to continue.")
        
        return all_finished
    
    def save_checkpoint(self):
        """Save current population state"""
        filename = f'checkpoint_gen_{self.ga.generation}.json'
        self.ga.save_to_file(filename)
        print(f"[SAVE] Checkpoint saved: {filename}")
    
    def save_best(self):
        """Save the best network to a file"""
        filename = f'best_network_gen_{self.ga.generation}.json'
        self.ga.save_best_network(filename)
        best_net, best_fitness, best_idx = self.ga.get_best_network()
        print(f"[SAVE] Best network saved: {filename}")
        print(f"       Fitness: {best_fitness:.2f} (Bot_{best_idx})")
        return filename

def print_help():
    """Print available terminal commands"""
    print("\n" + "="*60)
    print("TERMINAL COMMANDS:")
    print("="*60)
    print("  start          - Start a new generation run")
    print("  evolve         - Evolve and start next generation")
    print("  pause          - Disable auto-evolution")
    print("  resume         - Enable auto-evolution")
    print("  save           - Save current checkpoint")
    print("  save_best      - Save best network to file")
    print("  stats          - Show current statistics")
    print("  help           - Show this help message")
    print("  quit           - Exit program")
    print("="*60)

def terminal_input_loop(gen):
    """Handle terminal input commands"""
    print("\n[TERMINAL] Type 'help' for available commands")
    
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
                filename = gen.save_best()
                print(f"[SAVED] File is in current directory: {filename}")
            
            elif cmd == 'stats':
                stats = gen.ga.get_statistics()
                print(f"\nGeneration {gen.ga.generation}:")
                print(f"  Best:  {stats['best_fitness']:.2f}")
                print(f"  Avg:   {stats['avg_fitness']:.2f}")
                print(f"  Worst: {stats['worst_fitness']:.2f}")
            
            elif cmd == 'help':
                print_help()
            
            elif cmd == 'quit':
                print("[EXIT] Saving final checkpoint...")
                gen.save_checkpoint()
                gen.save_best()
                print("[EXIT] Goodbye!")
                sys.exit(0)
            
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for commands.")
        
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
    print(f"Population Size:  {BOT_COUNT}")
    print(f"Hidden Layer:     {HIDDEN_LAYER_SIZE} neurons")
    print(f"Input Size:       {INPUT_SIZE}")
    print(f"Mutation Rate:    {MUTATION_RATE}")
    print(f"Mutation Strength: {MUTATION_STRENGTH}")
    print(f"Elite Count:      {ELITE_COUNT}")
    print(f"Timeout:          {TIMEOUT_SECONDS}s ({MAX_TICKS} ticks)")
    print("="*60)
    
    gen = BotGeneration()
    
    # Create all bots
    print(f"\n[INIT] Creating {BOT_COUNT} bots...")
    for i in range(BOT_COUNT):
        gen.create_bot(i)
    
    # Store gen globally
    globals()['gen'] = gen
    
    # Start terminal input thread
    terminal_thread = threading.Thread(target=terminal_input_loop, args=(gen,), daemon=True)
    terminal_thread.start()
    
    # Auto-start first generation
    print(f"\n[AUTO] Starting first generation in 5 seconds...")
    threading.Timer(5.0, gen.start_new_run).start()

if __name__ == "__main__":
    main()