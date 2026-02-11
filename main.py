import random
import threading
import sys
import time
from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== CONFIGURATION ==========
BOT_COUNT = 10
TEAM_NAME = "bots"
SPAWN_POS = {'x': 0.5, 'y': 1, 'z': 0.5}     
GOAL_POS = {'x': 20.5, 'y': 1, 'z': 0.5}       

# Training Parameters
TIMEOUT_SECONDS = 15      
DEATH_Y = -1.0             # FAIL IMMEDIATELY if they fall below the platform level
RAY_MAX_DIST = 10         

# Neural Network Architecture
HIDDEN_LAYER_SIZE = 32     
INPUT_SIZE = 33            
OUTPUT_SIZE = 3            

# Genetic Algorithm Parameters
MUTATION_RATE = 0.1       
MUTATION_STRENGTH = 0.2    
ELITE_COUNT = 5            
AUTO_SAVE_INTERVAL = 10    

# ignore me in all server commands
BOT_SELECTOR = f"@a[team={TEAM_NAME},name=!EricZoop]"

class BotGeneration:
    def __init__(self):
        self.trackers = []
        self.generation_complete = False
        self.ga = GeneticAlgorithm(
            population_size=BOT_COUNT,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_LAYER_SIZE,
            output_size=OUTPUT_SIZE,
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH,
            elite_count=ELITE_COUNT
        )
        self.auto_evolve = True
        self.master_bot = None 
        self.all_time_top_scores = [] 

    def create_bot(self, index):
        bot = mineflayer.createBot({
            'host': 'localhost',  # 192.168.1.118
            'port': 25565,
            'username': f"Bot_{index}",
            'version': '1.19.4'
        })
        
        if index == 0:
            self.master_bot = bot
            
        tracker = BotTracker(bot, GOAL_POS, ray_max_dist=RAY_MAX_DIST)
        tracker.bot_index = index
        
        bot._is_spectator = True  # Start in spectator mode
        bot._ready_for_physics = False
        
        @Once(bot, 'spawn')
        def on_spawn(*args):
            bot.chat(f'/team join {TEAM_NAME}')
            # Start in spectator mode
            bot.chat('/gamemode spectator')
            
            if index == 0:
                # Essential Gamerules to reduce lag/chat spam
                bot.chat('/gamerule fallDamage false')
                bot.chat('/gamerule doImmediateRespawn true')
                bot.chat('/gamerule sendCommandFeedback false')
                bot.chat('/gamerule keepInventory true')

        @On(bot, 'physicsTick')
        def handle_physics(*args):
            if not bot._ready_for_physics:
                return
            
            # If bot is not active, ensure they're in spectator and not moving
            if not tracker.is_active:
                if not bot._is_spectator:
                    bot.clearControlStates()
                    bot.chat('/gamemode spectator')
                    bot._is_spectator = True
                return
            
            tracker.tick_count += 1
            
            # --- Fail Conditions ---
            
            # 1. Check Death (Y Level) - Switch to spectator BEFORE void damage
            if bot.entity:
                if bot.entity.position.y < DEATH_Y:
                    # Immediately switch to spectator to prevent actual death
                    bot.chat('/gamemode spectator')
                    self.end_bot_run(index, "FELL")
                    return
            
            # 2. Check Timeout
            if (time.time() - tracker.start_time) > TIMEOUT_SECONDS:
                self.end_bot_run(index, "TIMEOUT")
                return
            
            # 3. Check Success
            if tracker.check_goal_reached():
                # SPAWN FIREWORK IMMEDIATELY when goal is detected
                try:
                    if bot.entity:
                        x, y, z = bot.entity.position.x, bot.entity.position.y, bot.entity.position.z
                        bot.chat(f'/summon firework_rocket {x} {y+1} {z} {{LifeTime:12,FireworksItem:{{id:"minecraft:firework_rocket",Count:1,tag:{{Fireworks:{{Explosions:[{{Type:2,Colors:[I;16776960]}}]}}}}}}}}')
                        print(f"🎆 Bot_{index} COMPLETED THE COURSE! Fireworks launched! 🎆")
                except Exception as e:
                    print(f"[WARNING] Could not spawn firework for Bot_{index}: {e}")
                
                self.end_bot_run(index, "SUCCESS")
                return
            
            # --- Neural Network Control ---
            if tracker.tick_count % 2 == 0:
                action = self.get_neural_action(index)
                self.execute_action(bot, action)
        
        self.trackers.append(tracker)

    def end_bot_run(self, index, reason):
        tracker = self.trackers[index]
        bot = tracker.bot
        
        if not tracker.is_active: 
            return  # Already ended
        
        tracker.is_active = False
        tracker.reached_goal = (reason == "SUCCESS")
        
        # STOP MOVING IMMEDIATELY
        bot.clearControlStates()
        bot.setControlState('forward', False)
        bot.setControlState('sprint', False)
        bot.setControlState('jump', False)
        
        # Switch to Spectator (stay here until next generation)
        bot.chat('/gamemode spectator')
        bot._is_spectator = True
        
        tracker.update_fitness()
        self.check_generation_complete()

    def start_new_run(self):
        print(f"\nGeneration {self.ga.generation} - Initializing...")
        self.generation_complete = False
        
        # 1. Reset Trackers
        for tracker in self.trackers:
            tracker.reset_for_run()
            tracker.bot._ready_for_physics = False
            tracker.bot._is_spectator = False
            tracker.bot.clearControlStates()
        
        # 2. PRECISE TELEPORT SEQUENCE
        # First: Switch to spectator to "reset" their state
        if self.master_bot:
            self.master_bot.chat(f'/gamemode spectator {BOT_SELECTOR}')
        
        def switch_to_adventure():
            if self.master_bot:
                # Switch to adventure mode
                self.master_bot.chat(f'/gamemode adventure {BOT_SELECTOR}')
                print(f"Generation {self.ga.generation} - Switched to Adventure mode")
        
        def teleport_bots():
            if self.master_bot:
                # Teleport to spawn position
                self.master_bot.chat(f'/tp {BOT_SELECTOR} {SPAWN_POS["x"]} {SPAWN_POS["y"]+0.5} {SPAWN_POS["z"]}')
                print(f"Generation {self.ga.generation} - Teleported to spawn")
        
        def enable_bots():
            print(f"Generation {self.ga.generation} - GO!")
            current_time = time.time()
            for tracker in self.trackers:
                # Initialize distance metrics based on spawn position
                if tracker.bot.entity:
                    tracker.max_distance_to_goal = tracker.get_distance_to_goal()
                    tracker.min_distance_to_goal = tracker.max_distance_to_goal
                
                tracker.start_time = current_time
                tracker.is_active = True
                tracker.bot._ready_for_physics = True
        
        # Staggered timing for clean state transitions
        threading.Timer(0.5, switch_to_adventure).start()  # Spectator -> Adventure
        threading.Timer(1.5, teleport_bots).start()        # TP to spawn
        threading.Timer(3.5, enable_bots).start()          # Enable physics


    def get_neural_action(self, bot_index):
        tracker = self.trackers[bot_index]
        sensor_data = tracker.get_sensor_input_vector()
        
        inputs = []
        inputs.extend(sensor_data['rays'])           
        inputs.extend(sensor_data['position'])       
        inputs.extend(sensor_data['velocity'])       
        inputs.extend(sensor_data['orientation'])    
        inputs.extend(sensor_data['goal_direction']) 
        inputs.append(sensor_data['on_ground'])      
        
        network = self.ga.population[bot_index]
        return network.forward(inputs)
    
    def execute_action(self, bot, action):
        if not bot.entity: return 
        
        bot.setControlState('left', bool(action[0]))
        bot.setControlState('right', bool(action[1]))
        bot.setControlState('jump', bool(action[2]))
        
        # Always Forward/Sprint
        bot.setControlState('forward', True)   
        bot.setControlState('sprint', True)    
        
        # Never Back/Sneak
        bot.setControlState('back', False)     
        bot.setControlState('sneak', False)    
    
    def check_generation_complete(self):
        all_finished = all(not t.is_active for t in self.trackers)
        
        if all_finished and not self.generation_complete:
            self.generation_complete = True
            
            fitness_scores = [t.fitness_score for t in self.trackers]
            self.ga.evaluate_population(fitness_scores)
            
            # Track All-Time Stats
            for i, score in enumerate(fitness_scores):
                self.all_time_top_scores.append({
                    'score': score,
                    'gen': self.ga.generation,
                    'bot': i,
                    'status': "✓" if self.trackers[i].reached_goal else "✗"
                })
            
            self.all_time_top_scores.sort(key=lambda x: x['score'], reverse=True)
            self.all_time_top_scores = self.all_time_top_scores[:5]
            
            # --- PRINT ALL RESULTS ---
            print("\n" + "=" * 60)
            print(f"Generation {self.ga.generation} COMPLETE")
            print("=" * 60)
            
            sorted_bots = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
            print("RUN RESULTS:")
            for rank, (idx, fitness) in enumerate(sorted_bots, 1):
                status = "✓" if self.trackers[idx].reached_goal else "✗"
                print(f"  #{rank:02d} Bot_{idx:02d}: {fitness:7.2f} {status}")

            print("-" * 60)
            print("ALL-TIME TOP 5:")
            for rank, rec in enumerate(self.all_time_top_scores, 1):
                print(f"  #{rank} [Gen {rec['gen']}] Bot_{rec['bot']}: {rec['score']:.2f} {rec['status']}")
            print("=" * 60)

            if self.ga.generation % AUTO_SAVE_INTERVAL == 0:
                self.save_checkpoint()
            
            if self.auto_evolve:
                self.ga.evolve()
                print(f"\n[EVOLVING] Next generation starting in 5s...")
                # Increased delay between generations to 5s for better stability
                threading.Timer(5.0, self.start_new_run).start()
        
        return all_finished

    def save_checkpoint(self):
        filename = f'checkpoint_gen_{self.ga.generation}.json'
        self.ga.save_to_file(filename)
        print(f"[SAVE] {filename}")
    
    def save_best(self):
        filename = f'best_network_gen_{self.ga.generation}.json'
        self.ga.save_best_network(filename)
        print(f"[SAVE] {filename}")

def terminal_input_loop(gen):
    print("\n[TERMINAL] Type 'help' for commands")
    while True:
        try:
            cmd = input("> ").strip().lower()
            if cmd == 'start': 
                gen.start_new_run()
            elif cmd == 'evolve': 
                gen.ga.evolve()
                gen.start_new_run()
            elif cmd == 'pause':
                gen.auto_evolve = False
                print("[PAUSED]")
            elif cmd == 'resume':
                gen.auto_evolve = True
                print("[RESUMED]")
            elif cmd == 'save': 
                gen.save_checkpoint()
            elif cmd == 'save_best': 
                gen.save_best()
            elif cmd == 'help':
                print("""
Available Commands:
  start      - Start a new generation run
  evolve     - Evolve population and start new run
  pause      - Stop auto-evolution after current gen
  resume     - Resume auto-evolution
  save       - Save checkpoint
  save_best  - Save best network only
  quit       - Save and exit
  help       - Show this message
                """)
            elif cmd == 'quit': 
                print("Saving...")
                gen.save_checkpoint()
                sys.exit(0)
        except: break

def main():
    print("=" * 60)
    print("NEUROMORPHIC PARKOUR TRAINER - v2.0")
    print("Optimized: Minimal TP, Spectator Persistence, Fireworks!")
    print("=" * 60)
    
    gen = BotGeneration()
    
    print(f"\n[INIT] Connecting {BOT_COUNT} bots...")
    for i in range(BOT_COUNT):
        gen.create_bot(i)
        time.sleep(0.1) 
    
    globals()['gen'] = gen
    
    threading.Thread(target=terminal_input_loop, args=(gen,), daemon=True).start()
    
    print(f"\n[AUTO] Starting in 10 seconds...")
    threading.Timer(10.0, gen.start_new_run).start()

if __name__ == "__main__":
    main()