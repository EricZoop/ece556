import random
from javascript import require, On, Once
from track_bot import BotTracker

mineflayer = require('mineflayer')

# ========== CONFIGURATION ==========
BOT_COUNT = 5
TEAM_NAME = "bots"
SPAWN_POS = {'x': 0.5, 'y': 0, 'z': 0.5}   # Where they start the course
PURGATORY_POS = {'x': -15, 'y': 1, 'z': 0.5} # Where they wait after failing
GOAL_POS = {'x': 25.5, 'y': 0, 'z': 0.5}

# --- Training Parameters ---
TIMEOUT_SECONDS = 20
MAX_TICKS = TIMEOUT_SECONDS * 20
DEATH_Y = -10  
SCAN_RADIUS = 5    

class BotGeneration:
    def __init__(self):
        self.trackers = []
        self.generation_complete = False
        
    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host': 'localhost',
            'port': 51376,
            'username': f"Bot_{index}",
            'version': '1.19.4'
        })
        
        tracker = BotTracker(bot, GOAL_POS, scan_radius=SCAN_RADIUS)
        tracker.bot_index = index  # Store index for logging
        
        # Flag to track if initial setup is complete
        bot._setup_complete = False
        bot._purgatory_tp_done = False
        bot._ready_tick_counter = 0
        
        @Once(bot, 'spawn')
        def on_spawn(*args):
            # Join team immediately
            bot.chat(f'/team join {TEAM_NAME}')
            print(f"[SPAWN] Bot_{index} spawned, waiting for position...")

        @On(bot, 'move')
        def on_move(*args):
            # Wait until we have a valid position, then teleport to purgatory
            if not bot._purgatory_tp_done and bot.entity and bot.entity.position:
                bot._purgatory_tp_done = True
                bot.chat(f'/tp {PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}')
                print(f"[GEN] Bot_{index} teleported to purgatory")

        @On(bot, 'physicsTick')
        def handle_physics(*args):
            # Handle initial setup delay (wait 20 ticks after purgatory TP)
            if bot._purgatory_tp_done and not bot._setup_complete:
                bot._ready_tick_counter += 1
                if bot._ready_tick_counter >= 20:  # 1 second delay
                    bot._setup_complete = True
                    print(f"[READY] Bot_{index} ready in purgatory")
                return
            
            # Wait for initial setup to complete
            if not bot._setup_complete:
                return
            
            # Only process if bot is active in a run
            if not tracker.is_active or tracker.reached_goal:
                return
            
            tracker.tick_count += 1
            
            # Safety: Wait for 20 ticks to stabilize after TP to spawn position
            if tracker.tick_count < 20: 
                return

            # 1. Death Check - fell into void
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
                self.execute_action(bot, self.get_random_action())
        
        self.trackers.append(tracker)

    def move_to_purgatory(self, index, reason):
        """Move bot to purgatory after completing their run (success or failure)"""
        tracker = self.trackers[index]
        bot = tracker.bot
        tracker.is_active = False # Stop processing physics logic
        
        # Clear all movement keys immediately
        bot.clearControlStates()
        
        # Teleport to purgatory
        bot.chat(f'/tp {PURGATORY_POS["x"]} {PURGATORY_POS["y"]} {PURGATORY_POS["z"]}')
        
        # Calculate and log fitness
        fitness = tracker.update_fitness()
        print(f"[{reason}] Bot_{index} | Ticks: {tracker.tick_count} | Fitness: {fitness:.2f}")
        
        # Check if all bots are done
        self.check_generation_complete()

    def start_new_run(self):
        """Teleport all bots to the spawn position and begin the parkour course"""
        print("\n" + "="*20 + " STARTING NEW RUN " + "="*20)
        self.generation_complete = False
        
        for i, tracker in enumerate(self.trackers):
            # Reset tracker stats
            tracker.reset_for_run()
            
            # Teleport to spawn position
            tracker.bot.chat(f'/tp {SPAWN_POS["x"]} {SPAWN_POS["y"]} {SPAWN_POS["z"]}')
            print(f"[START] Bot_{i} teleported to course start")

    def get_random_action(self):
        """Generate random action vector for placeholder neural network"""
        return [
            random.random() > 0.2,   # forward (80% chance)
            False,                    # back (never)
            random.random() > 0.9,   # left (10% chance)
            random.random() > 0.9,   # right (10% chance)
            random.random() > 0.8,   # jump (20% chance)
            random.random() > 0.5    # sprint (50% chance)
        ]
    
    def execute_action(self, bot, action):
        """Execute the action vector on the bot"""
        if not bot.entity: 
            return 
        
        bot.setControlState('forward', bool(action[0]))
        bot.setControlState('back', bool(action[1]))
        bot.setControlState('left', bool(action[2]))
        bot.setControlState('right', bool(action[3]))
        bot.setControlState('jump', bool(action[4]))
        bot.setControlState('sprint', bool(action[5]))
    
    def check_generation_complete(self):
        """Check if all bots have finished their runs and auto-restart."""
        all_finished = all(not t.is_active for t in self.trackers)
        
        if all_finished and not self.generation_complete:
            self.generation_complete = True
            print("\n" + "="*20 + " GENERATION COMPLETE " + "="*20)
            self.print_summary()
            
            # --- AUTOMATION ADDITION ---
            import threading
            print("\n[AUTO] Starting next generation in 3 seconds...")
            threading.Timer(3.0, self.start_new_run).start()
            # ---------------------------
            
        return all_finished
    
    def print_summary(self):
        """Print fitness summary for all bots"""
        for i, tracker in enumerate(self.trackers):
            status = "SUCCESS" if tracker.reached_goal else "FAILED"
            print(f"Bot_{i}: {status:7} | Fitness: {tracker.fitness_score:6.2f} | Ticks: {tracker.tick_count:4}")

def main():

    print("NEUROMORPHIC MINECRAFT PARKOUR TRAINER")
    print(f"Bots: {BOT_COUNT} | Timeout: {TIMEOUT_SECONDS}s")
    print(f"Spawn: ({SPAWN_POS['x']}, {SPAWN_POS['y']}, {SPAWN_POS['z']})")
    print(f"Goal: ({GOAL_POS['x']}, {GOAL_POS['y']}, {GOAL_POS['z']})")
    
    gen = BotGeneration()
    
    # Create all bots
    for i in range(BOT_COUNT):
        gen.create_bot(i)
    
    print(f"\n[INIT] Creating {BOT_COUNT} bots...")
    print("[INFO] Bots will spawn in purgatory and wait")
    print("[INFO] To start first run, wait ~5 seconds then run in console:")
    print("       >>> gen.start_new_run()")
    
    # Store gen globally so you can call gen.start_new_run() from console
    globals()['gen'] = gen
    
    # Optional: Auto-start after 5 seconds
    import threading
    threading.Timer(5.0, gen.start_new_run).start()

if __name__ == "__main__":
    main()