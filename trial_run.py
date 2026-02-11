"""
Trial run script - Test a trained network on a new parkour course.
Loads a saved neural network and runs it without training/evolution.
Supports deployment in new environments via JSON configuration.
"""

import sys
import json
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== DEFAULT CONFIGURATION ==========
DEFAULT_CONFIG = {
    'team_name': 'bots',
    'spawn_pos': {'x': 0.5, 'y': 0, 'z': 0.5},
    'goal_pos': {'x': 10, 'y': 0, 'z': 0.5},
    'timeout_seconds': 30,
    'death_y': -10,
    'ray_max_dist': 10
}

def load_config(config_file=None):
    """Load environment configuration from JSON file or use defaults"""
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG] Loaded environment from: {config_file}")
            return config
        except FileNotFoundError:
            print(f"[WARNING] Config file '{config_file}' not found, using defaults")
            return DEFAULT_CONFIG
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON in '{config_file}', using defaults")
            return DEFAULT_CONFIG
    else:
        return DEFAULT_CONFIG

class TrialBot:
    """Single bot running a pre-trained network"""
    
    def __init__(self, network_file, config):
        print(f"\n{'='*60}")
        print("TRIAL RUN MODE - Testing Trained Network")
        print(f"{'='*60}")
        print(f"Loading network from: {network_file}")
        
        # Load the trained network
        self.network = GeneticAlgorithm.load_best_network(network_file)
        print(f"Network loaded successfully!")
        print(f"  Input size:  {self.network.input_size}")
        print(f"  Hidden size: {self.network.hidden_size}")
        print(f"  Output size: {self.network.output_size}")
        
        # Store configuration
        self.config = config
        self.spawn_pos = config['spawn_pos']
        self.goal_pos = config['goal_pos']
        self.timeout_seconds = config['timeout_seconds']
        self.death_y = config['death_y']
        self.ray_max_dist = config['ray_max_dist']
        self.team_name = config['team_name']
        
        self.max_ticks = self.timeout_seconds * 20
        
        print(f"\nEnvironment Configuration:")
        print(f"  Spawn:   ({self.spawn_pos['x']}, {self.spawn_pos['y']}, {self.spawn_pos['z']})")
        print(f"  Goal:    ({self.goal_pos['x']}, {self.goal_pos['y']}, {self.goal_pos['z']})")
        print(f"  Timeout: {self.timeout_seconds}s")
        print(f"  Death Y: {self.death_y}")
        
        self.tracker = None
        self.bot = None
        self.trial_active = False
        
    def create_bot(self):
        """Create the trial bot"""
        self.bot = mineflayer.createBot({
            'host': 'localhost',
            'port': 51376,
            'username': "TrialBot",
            'version': '1.19.4'
        })
        
        self.tracker = BotTracker(self.bot, self.goal_pos, ray_max_dist=self.ray_max_dist)
        
        # Setup flags
        self.bot._setup_complete = False
        self.bot._spawn_tp_done = False
        self.bot._ready_tick_counter = 0
        
        @Once(self.bot, 'spawn')
        def on_spawn(*args):
            self.bot.chat(f'/team join {self.team_name}')
            print(f"\n[SPAWN] TrialBot spawned, joining team...")

        @On(self.bot, 'move')
        def on_move(*args):
            # Wait for valid position, then teleport to spawn
            if not self.bot._spawn_tp_done and self.bot.entity and self.bot.entity.position:
                self.bot._spawn_tp_done = True
                self.bot.chat(f'/tp {self.spawn_pos["x"]} {self.spawn_pos["y"]} {self.spawn_pos["z"]}')
                print(f"[READY] TrialBot teleported to course start")

        @On(self.bot, 'physicsTick')
        def handle_physics(*args):
            # Handle initial setup delay
            if self.bot._spawn_tp_done and not self.bot._setup_complete:
                self.bot._ready_tick_counter += 1
                if self.bot._ready_tick_counter >= 20:
                    self.bot._setup_complete = True
                    self.start_trial()
                return
            
            if not self.bot._setup_complete:
                return
            
            # Only process if trial is active
            if not self.trial_active:
                return
            
            self.tracker.tick_count += 1

            # 1. Death Check
            if self.bot.entity and self.bot.entity.position.y < self.death_y:
                self.end_trial("FELL INTO VOID")
                return
            
            # 2. Goal Check
            if self.tracker.check_goal_reached():
                self.tracker.reached_goal = True
                self.end_trial("SUCCESS - REACHED GOAL!")
                return
            
            # 3. Timeout Check
            if self.tracker.tick_count >= self.max_ticks:
                self.end_trial("TIMEOUT")
                return
            
            # 4. Execute Neural Action every 2 ticks
            if self.tracker.tick_count % 2 == 0:
                action = self.get_neural_action()
                self.execute_action(action)
                
                # Progress update every 2 seconds
                if self.tracker.tick_count % 40 == 0:
                    distance = self.tracker.get_distance_to_goal()
                    elapsed = self.tracker.tick_count // 20
                    print(f"[{elapsed}s] Distance to goal: {distance:.2f} blocks")
    
    def start_trial(self):
        """Begin the trial run"""
        print(f"\n{'='*60}")
        print("TRIAL STARTING")
        print(f"{'='*60}")
        self.trial_active = True
        self.tracker.reset_for_run()
    
    def get_neural_action(self):
        """Get action from the trained neural network"""
        sensor_data = self.tracker.get_sensor_input_vector()
        
        # Flatten all inputs into a single vector (33 inputs)
        inputs = []
        inputs.extend(sensor_data['rays'])           # 21 values
        inputs.extend(sensor_data['position'])       # 3 values
        inputs.extend(sensor_data['velocity'])       # 3 values
        inputs.extend(sensor_data['orientation'])    # 2 values
        inputs.extend(sensor_data['goal_direction']) # 3 values
        inputs.append(sensor_data['on_ground'])      # 1 value
        
        # Get action from trained network
        action = self.network.forward(inputs)
        return action
    
    def execute_action(self, action):
        """
        Execute the 3-dimensional action vector:
        [0]: Left
        [1]: Right
        [2]: Jump
        
        Forward and Sprint are ALWAYS ON.
        """
        if not self.bot.entity:
            return
        
        # Neural network controls (LEFT, RIGHT, JUMP only)
        self.bot.setControlState('left', bool(action[0]))
        self.bot.setControlState('right', bool(action[1]))
        self.bot.setControlState('jump', bool(action[2]))
        
        # HARDCODED - ALWAYS ON
        self.bot.setControlState('forward', True)   # ALWAYS moving forward
        self.bot.setControlState('sprint', True)    # ALWAYS sprinting
        
        # HARDCODED - ALWAYS OFF
        self.bot.setControlState('back', False)     # Never back up
        self.bot.setControlState('sneak', False)    # Never sneak
    
    def end_trial(self, reason):
        """End the trial and show results"""
        self.trial_active = False
        self.bot.clearControlStates()
        
        # Calculate final fitness
        fitness = self.tracker.update_fitness()
        
        print(f"\n{'='*60}")
        print("TRIAL COMPLETE")
        print(f"{'='*60}")
        print(f"Result:   {reason}")
        print(f"Time:     {self.tracker.tick_count/20:.1f}s ({self.tracker.tick_count} ticks)")
        print(f"Fitness:  {fitness:.2f}")
        print(f"Distance: {self.tracker.get_distance_to_goal():.2f} blocks")
        if self.tracker.max_height_reached:
            print(f"Max Height: {self.tracker.max_height_reached:.2f}")
        print(f"{'='*60}")
        
        # Ask if user wants to retry
        print("\nPress Enter to retry, or Ctrl+C to exit...")
        try:
            input()
            self.retry_trial()
        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye!")
            sys.exit(0)
    
    def retry_trial(self):
        """Reset and run the trial again"""
        print("\n[RETRY] Restarting trial in 2 seconds...")
        import threading
        
        def restart():
            self.bot.chat(f'/tp {self.spawn_pos["x"]} {self.spawn_pos["y"]} {self.spawn_pos["z"]}')
            self.tracker.reset_for_run()
            self.trial_active = True
            print(f"\n{'='*60}")
            print("TRIAL RESTARTING")
            print(f"{'='*60}")
        
        threading.Timer(2.0, restart).start()

def print_usage():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("USAGE:")
    print("="*60)
    print("python trial_run.py <network_file.json> [environment_config.json]")
    print("\nExamples:")
    print("  python trial_run.py best_network_gen_50.json")
    print("  python trial_run.py best_network.json new_course.json")
    print("\nEnvironment JSON format:")
    print(json.dumps(DEFAULT_CONFIG, indent=2))
    print("="*60)

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    network_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load environment configuration
    config = load_config(config_file)
    
    # Create and run trial
    trial = TrialBot(network_file, config)
    trial.create_bot()
    
    # Store globally for debugging
    globals()['trial'] = trial

if __name__ == "__main__":
    main()