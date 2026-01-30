"""
Trial run script - Test a trained network on a new parkour course.
Loads a saved neural network and runs it without training/evolution.
"""

import sys
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== CONFIGURATION ==========
TEAM_NAME = "bots"
SPAWN_POS = {'x': 0.5, 'y': 0, 'z': 0.5}
GOAL_POS = {'x': 10, 'y': 0, 'z': 0.5}  # Update for new course

# --- Trial Parameters ---
TIMEOUT_SECONDS = 30
MAX_TICKS = TIMEOUT_SECONDS * 20
DEATH_Y = -10
SCAN_RADIUS = 5

# --- Load Network ---
if len(sys.argv) < 2:
    print("Usage: python trial_run.py <network_file.json>")
    print("Example: python trial_run.py best_network_gen_50.json")
    sys.exit(1)

NETWORK_FILE = sys.argv[1]

class TrialBot:
    """Single bot running a pre-trained network"""
    
    def __init__(self, network_file):
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
        
        self.tracker = BotTracker(self.bot, GOAL_POS, scan_radius=SCAN_RADIUS)
        
        # Setup flags
        self.bot._setup_complete = False
        self.bot._spawn_tp_done = False
        self.bot._ready_tick_counter = 0
        
        @Once(self.bot, 'spawn')
        def on_spawn(*args):
            self.bot.chat(f'/team join {TEAM_NAME}')
            print(f"\n[SPAWN] TrialBot spawned, waiting for position...")

        @On(self.bot, 'move')
        def on_move(*args):
            # Wait for valid position, then teleport to spawn
            if not self.bot._spawn_tp_done and self.bot.entity and self.bot.entity.position:
                self.bot._spawn_tp_done = True
                self.bot.chat(f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}')
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
            if self.bot.entity and self.bot.entity.position.y < DEATH_Y:
                self.end_trial("FELL INTO VOID")
                return
            
            # 2. Goal Check
            if self.tracker.check_goal_reached():
                self.tracker.reached_goal = True
                self.end_trial("SUCCESS - REACHED GOAL!")
                return
            
            # 3. Timeout Check
            if self.tracker.tick_count >= MAX_TICKS:
                self.end_trial("TIMEOUT")
                return
            
            # 4. Execute Neural Action every 2 ticks
            if self.tracker.tick_count % 2 == 0:
                action = self.get_neural_action()
                self.execute_action(action)
                
                # Progress update every 2 seconds
                if self.tracker.tick_count % 40 == 0:
                    distance = self.tracker.get_distance_to_goal()
                    print(f"[{self.tracker.tick_count//20}s] Distance to goal: {distance:.2f}")
    
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
        
        # Get action from trained network
        action = self.network.forward(inputs)
        return action
    
    def execute_action(self, action):
        """Execute the action vector"""
        if not self.bot.entity:
            return
        
        self.bot.setControlState('forward', bool(action[0]))
        self.bot.setControlState('back', bool(action[1]))
        self.bot.setControlState('left', bool(action[2]))
        self.bot.setControlState('right', bool(action[3]))
        self.bot.setControlState('jump', bool(action[4]))
        self.bot.setControlState('sprint', bool(action[5]))
    
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
            self.bot.chat(f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}')
            self.tracker.reset_for_run()
            self.trial_active = True
            print(f"\n{'='*60}")
            print("TRIAL RESTARTING")
            print(f"{'='*60}")
        
        threading.Timer(2.0, restart).start()

def main():
    trial = TrialBot(NETWORK_FILE)
    trial.create_bot()
    
    # Store globally for debugging
    globals()['trial'] = trial

if __name__ == "__main__":
    main()
