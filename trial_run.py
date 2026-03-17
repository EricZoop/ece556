"""
Trial run – test a trained network on a parkour course.
Loads a saved neural network and runs it without training/evolution.
Shows a live matplotlib window visualising neuron activations each tick.

Architecture: matplotlib MUST own the main thread (Windows/macOS requirement).
The mineflayer bot runs in a daemon thread.  Activations are passed to the
main thread via a thread-safe queue.
"""

import sys
import json
import time
import threading
import queue
import numpy as np

import matplotlib
matplotlib.use('TkAgg')   # works on Windows; swap to 'QtAgg' if TkAgg is missing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from javascript import require, On, Once
from track_bot import BotTracker
from genetic_algorithm import GeneticAlgorithm

mineflayer = require('mineflayer')

# ========== DEFAULT CONFIGURATION ==========
DEFAULT_CONFIG = {
    'host':             '192.168.1.118',   # LAN IP - override via config.json or CLI arg
    'port':             25565,
    'team_name':        'bots',
    'spawn_pos':        {'x': 0.5,  'y': -63, 'z': 0.5},
    'goal_pos':         {'x': 0.5,  'y': -63, 'z': 12.5},
    'timeout_seconds':  30,
    'death_y':          -65,
    'ray_max_dist':     10,
}

# Input slice labels / colours  (must sum to 41)
INPUT_GROUPS = [
    ('Rays H',   8,  '#4FC3F7'),
    ('Rays D',   8,  '#29B6F6'),
    ('Rays U',   4,  '#0288D1'),
    ('Jump arc', 3,  '#0277BD'),
    ('Void',     3,  '#01579B'),
    ('Down',     1,  '#003c6e'),
    ('Position', 3,  '#81C784'),
    ('Velocity', 3,  '#AED581'),
    ('Orient',   2,  '#FFF176'),
    ('Goal dir', 3,  '#FFB74D'),
    ('Ground',   1,  '#EF9A9A'),
    ('Dist',     1,  '#E57373'),
    ('Time',     1,  '#EF5350'),
]
OUTPUT_LABELS = ['LEFT', 'RIGHT', 'JUMP']
OUTPUT_COLORS = ['#F48FB1', '#80CBC4', '#FFD54F']

# Shared inter-thread state
_viz_queue   = queue.Queue(maxsize=2)
_retry_event = threading.Event()
_quit_flag   = threading.Event()
_trial_ref   = [None]   # list so the nested thread can write to it


# ------------------------------------------------------------------------------
def load_config(config_file=None):
    if config_file:
        try:
            with open(config_file) as f:
                cfg = {**DEFAULT_CONFIG, **json.load(f)}
            print(f'Config loaded: {config_file}')
            return cfg
        except FileNotFoundError:
            print(f"Config '{config_file}' not found – using defaults.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in '{config_file}' – using defaults.")
    return DEFAULT_CONFIG.copy()


def safe_clear(bot):
    try:
        bot.clearControlStates()
    except Exception:
        for k in ('forward','back','left','right','jump','sprint','sneak'):
            try: bot.setControlState(k, False)
            except: pass


def _push_viz(activations: dict, meta: dict):
    """Non-blocking push; drops the oldest frame if the queue is full."""
    try:
        _viz_queue.put_nowait((activations, meta))
    except queue.Full:
        try: _viz_queue.get_nowait()
        except queue.Empty: pass
        try: _viz_queue.put_nowait((activations, meta))
        except queue.Full: pass


# ------------------------------------------------------------------------------
class TrialBot:
    """Runs entirely inside a background daemon thread."""

    def __init__(self, network_file: str, config: dict):
        print('=== Trial Run Mode ===')
        print(f'Network:  {network_file}')
        self.network = GeneticAlgorithm.load_best_network(network_file)
        print(f'Inputs:   {self.network.input_size}')
        print(f'Hidden:   {self.network.hidden_size}')
        print(f'Outputs:  {self.network.output_size}')

        self.config          = config
        self.spawn_pos       = config['spawn_pos']
        self.goal_pos        = config['goal_pos']
        self.timeout_seconds = config['timeout_seconds']
        self.death_y         = config['death_y']
        self.ray_max_dist    = config['ray_max_dist']
        self.team_name       = config['team_name']
        self.max_ticks       = self.timeout_seconds * 20

        sp, gp = self.spawn_pos, self.goal_pos
        print(f"Spawn:   ({sp['x']}, {sp['y']}, {sp['z']})")
        print(f"Goal:    ({gp['x']}, {gp['y']}, {gp['z']})")
        print(f"Timeout: {self.timeout_seconds}s  |  Death Y: {self.death_y}")
        print(f"Host:    {config['host']}:{config['port']}")

        self.tracker      = None
        self.bot          = None
        self.trial_active = False

    # ------------------------------------------------------------------
    def start(self):
        self.bot = mineflayer.createBot({
            'host':     self.config['host'],
            'port':     self.config['port'],
            'username': 'TrialBot',
            'version':  '1.19.4',
        })

        self.tracker = BotTracker(
            self.bot, self.goal_pos,
            ray_max_dist = self.ray_max_dist,
            spawn_pos    = self.config['spawn_pos'],
        )

        self.bot._spawn_tp_done      = False
        self.bot._setup_complete     = False
        self.bot._ready_tick_counter = 0

        @Once(self.bot, 'spawn')
        def on_spawn(*args):
            self.bot.chat(f'/team join {self.team_name}')
            self.bot.chat('/gamerule fallDamage false')
            self.bot.chat('/gamerule doImmediateRespawn true')
            self.bot.chat('/gamerule sendCommandFeedback false')
            print('Spawned – teleporting to course start…')

        @On(self.bot, 'move')
        def on_move(*args):
            if not self.bot._spawn_tp_done and self.bot.entity and self.bot.entity.position:
                self.bot._spawn_tp_done = True
                sp = self.spawn_pos
                self.bot.chat(f'/tp {sp["x"]} {sp["y"]} {sp["z"]}')

        @On(self.bot, 'physicsTick')
        def handle_physics(*args):
            # Wait 20 ticks after teleport to stabilise physics
            if self.bot._spawn_tp_done and not self.bot._setup_complete:
                self.bot._ready_tick_counter += 1
                if self.bot._ready_tick_counter >= 20:
                    self.bot._setup_complete = True
                    self._start_trial()
                return

            if not self.bot._setup_complete or not self.trial_active:
                return

            self.tracker.tick_count += 1

            if self.bot.entity and self.bot.entity.position.y < self.death_y:
                self._end_trial('FELL INTO VOID')
                return

            if self.tracker.check_goal_reached():
                self.tracker.reached_goal = True
                try:
                    if self.bot.entity:
                        x = self.bot.entity.position.x
                        y = self.bot.entity.position.y
                        z = self.bot.entity.position.z
                        self.bot.chat('/effect give @s minecraft:resistance 3 255 true')
                        self.bot.chat(
                            f'/summon firework_rocket {x} {y+8} {z} '
                            '{{LifeTime:12,FireworksItem:{{id:"minecraft:firework_rocket",'
                            'Count:1,tag:{{Fireworks:{{Explosions:[{{Type:2,Colors:[I;16776960]}}'
                            ']}}}}}}}}'
                        )
                except Exception:
                    pass
                self._end_trial('REACHED GOAL')
                return

            if self.tracker.tick_count >= self.max_ticks:
                self._end_trial('TIMEOUT')
                return

            if self.tracker.tick_count % 2 == 0:
                action, activations = self._neural_action()
                self._execute(action)

                elapsed = time.time() - self.tracker.start_time
                dist    = self.tracker.get_distance_to_goal()
                _push_viz(activations, {
                    'distance': dist,
                    'elapsed':  elapsed,
                    'ticks':    self.tracker.tick_count,
                })

                if self.tracker.tick_count % 40 == 0:
                    print(f'  [{int(elapsed):3d}s]  dist={dist:.2f}  '
                          f'act=[L:{action[0]} R:{action[1]} J:{action[2]}]')

    # ------------------------------------------------------------------
    def _start_trial(self):
        print('\n=== Trial Starting ===')
        self.tracker.reset_for_run()
        self.tracker.start_time = time.time()
        self.tracker.is_active  = True
        self.trial_active       = True

    def _build_inputs(self):
        sd = self.tracker.get_sensor_input_vector(self.timeout_seconds)
        inp = []
        inp.extend(sd['rays'])
        inp.extend(sd['position'])
        inp.extend(sd['velocity'])
        inp.extend(sd['orientation'])
        inp.extend(sd['goal_direction'])
        inp.append(sd['on_ground'])
        inp.append(sd['distance_to_goal'])
        inp.append(sd['time_remaining'])
        return inp

    def _neural_action(self):
        return self.network.forward_with_activations(self._build_inputs())

    def _execute(self, action):
        if not self.bot.entity:
            return
        self.bot.setControlState('left',    bool(action[0]))
        self.bot.setControlState('right',   bool(action[1]))
        self.bot.setControlState('jump',    bool(action[2]))
        self.bot.setControlState('forward', True)
        self.bot.setControlState('sprint',  True)
        self.bot.setControlState('back',    False)
        self.bot.setControlState('sneak',   False)

    def _end_trial(self, reason: str):
        self.trial_active         = False
        self.tracker.is_active    = False
        self.tracker.run_duration = time.time() - self.tracker.start_time
        safe_clear(self.bot)

        fitness = self.tracker.update_fitness(self.timeout_seconds)
        dist    = self.tracker.get_distance_to_goal()

        try:
            _, last_act = self._neural_action()
            _push_viz(last_act, {
                'distance': dist, 'elapsed': self.tracker.run_duration,
                'ticks': self.tracker.tick_count,
                'result': reason, 'fitness': fitness,
            })
        except Exception:
            pass

        print(f'\n=== Trial Complete ===')
        print(f'Result:   {reason}')
        print(f'Time:     {self.tracker.run_duration:.1f}s  ({self.tracker.tick_count} ticks)')
        print(f'Distance: {dist:.2f} blocks')
        print(f'Fitness:  {fitness:.2f}')
        print('======================')
        print('\nPress Enter to retry, or type "quit" to exit…')
        _retry_event.set()

    def do_retry(self):
        _retry_event.clear()
        def restart():
            sp = self.spawn_pos
            self.bot.chat(f'/tp {sp["x"]} {sp["y"]} {sp["z"]}')
            time.sleep(1.5)
            self.tracker.reset_for_run()
            self.tracker.start_time = time.time()
            self.tracker.is_active  = True
            self.trial_active       = True
            print('\n=== Trial Restarting ===')
        threading.Timer(2.0, restart).start()


# ------------------------------------------------------------------------------
#  Matplotlib drawing helpers  (always called from main thread)
# ------------------------------------------------------------------------------
def _build_figure(input_size, hidden_size, output_size):
    fig = plt.figure(figsize=(17, 8), facecolor='#1a1a2e')
    fig.canvas.manager.set_window_title('Neural Activations – Trial Run')

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.04, right=0.97, top=0.88, bottom=0.07,
        wspace=0.35, hspace=0.55,
        width_ratios=[2.2, 0.18, 1.4, 0.7],
    )

    ax_inp  = fig.add_subplot(gs[:, 0])
    ax_arr1 = fig.add_subplot(gs[:, 1])
    ax_hid  = fig.add_subplot(gs[:, 2])
    ax_arr2 = fig.add_subplot(gs[0, 3])
    ax_out  = fig.add_subplot(gs[1, 3])

    BG = '#0d0d1a'
    for ax in [ax_inp, ax_arr1, ax_hid, ax_arr2, ax_out]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color('#333366')

    for ax in [ax_arr1, ax_arr2]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#6666aa', lw=2))

    kw = dict(color='#aaaacc', fontsize=10, pad=5)
    ax_inp.set_title(f'Inputs  ({input_size})',  **kw)
    ax_hid.set_title(f'Hidden  ({hidden_size})', **kw)
    ax_out.set_title(f'Output  ({output_size})', **kw)

    meta_txt = fig.text(0.5, 0.95, 'Waiting for bot connection…',
                         ha='center', va='top', color='#ccccff',
                         fontsize=11, fontweight='bold')

    return fig, ax_inp, ax_hid, ax_out, meta_txt


def _redraw(ax_inp, ax_hid, ax_out, meta_txt, activations, meta, hidden_size):
    inp_vals = activations['input']
    hid_vals = activations['hidden']
    out_vals = activations['output']

    # -- inputs --
    ax_inp.cla()
    ax_inp.set_facecolor('#0d0d1a')
    ax_inp.set_title(f'Inputs  ({len(inp_vals)})', color='#aaaacc', fontsize=10, pad=5)
    y = 0
    yticks, ylabels = [], []
    for label, size, color in INPUT_GROUPS:
        group = inp_vals[y:y + size]
        for k, v in enumerate(group):
            norm_v = max(0.0, min(1.0, (float(v) + 1.0) / 2.0))
            ax_inp.barh(y + k, norm_v, color=color, alpha=0.4 + 0.6 * norm_v, height=0.8)
            ax_inp.text(norm_v + 0.02, y + k, f'{v:.2f}',
                        va='center', color='#888899', fontsize=5.5)
        yticks.append(y + size / 2 - 0.5)
        ylabels.append(label)
        y += size
    ax_inp.set_yticks(yticks)
    ax_inp.set_yticklabels(ylabels, color='#aaaacc', fontsize=7)
    ax_inp.set_xlim(0, 1.35)
    ax_inp.set_ylim(-0.5, len(inp_vals) - 0.5)
    ax_inp.invert_yaxis()
    ax_inp.tick_params(axis='x', colors='#555577', labelsize=7)
    ax_inp.set_xlabel('Normalised value', color='#555577', fontsize=7)
    for sp in ax_inp.spines.values(): sp.set_color('#333366')

    # -- hidden heatmap --
    ax_hid.cla()
    ax_hid.set_facecolor('#0d0d1a')
    ax_hid.set_title(f'Hidden  ({hidden_size})', color='#aaaacc', fontsize=10, pad=5)
    cols = 8
    rows = hidden_size // cols
    grid = hid_vals[:rows * cols].reshape(rows, cols)
    ax_hid.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1,
                  aspect='auto', interpolation='nearest')
    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            ax_hid.text(c, r, f'{v:.2f}', ha='center', va='center',
                        color='#000' if abs(v) > 0.5 else '#ccc', fontsize=6)
    ax_hid.tick_params(colors='#555577', labelsize=7)
    ax_hid.set_xlabel('neuron col', color='#555577', fontsize=7)
    ax_hid.set_ylabel('neuron row', color='#555577', fontsize=7)
    for sp in ax_hid.spines.values(): sp.set_color('#333366')

    # -- outputs --
    ax_out.cla()
    ax_out.set_facecolor('#0d0d1a')
    ax_out.set_title('Output  (3)', color='#aaaacc', fontsize=10, pad=5)
    for i, (label, color) in enumerate(zip(OUTPUT_LABELS, OUTPUT_COLORS)):
        v     = float(out_vals[i])
        fired = v > 0.5
        ax_out.barh(i, v, color=color, alpha=0.35 + 0.65 * v, height=0.65)
        marker = '◆ ON ' if fired else '   OFF'
        ax_out.text(1.05, i, f'{marker}  {v:.2f}', va='center',
                    color=color if fired else '#666688',
                    fontsize=9, fontweight='bold' if fired else 'normal')
        ax_out.text(-0.04, i, label, va='center', ha='right',
                    color='#aaaacc', fontsize=8)
    ax_out.set_xlim(0, 1.7)
    ax_out.set_ylim(-0.6, 2.4)
    ax_out.invert_yaxis()
    ax_out.set_yticks([])
    ax_out.set_xticks([0, 0.5, 1.0])
    ax_out.tick_params(axis='x', colors='#555577', labelsize=7)
    ax_out.axvline(0.5, color='#444466', ls='--', lw=0.8)
    for sp in ax_out.spines.values(): sp.set_color('#333366')

    # -- meta --
    parts = [
        f"t={meta.get('elapsed', 0):.1f}s",
        f"tick={meta.get('ticks', 0)}",
        f"dist={meta.get('distance', 0):.2f}",
    ]
    if meta.get('fitness') is not None:
        parts.append(f"fitness={meta['fitness']:.1f}")
    if meta.get('result'):
        parts.append(f"[{meta['result']}]")
    meta_txt.set_text('   |   '.join(parts))


# ------------------------------------------------------------------------------
def _bot_thread_main(network_file: str, config: dict):
    trial = TrialBot(network_file, config)
    _trial_ref[0] = trial
    trial.start()

    # Handle retry/quit prompts from end_trial
    while not _quit_flag.is_set():
        if _retry_event.wait(timeout=0.5):
            try:
                ans = input().strip().lower()
            except EOFError:
                ans = ''
            if ans == 'quit':
                _quit_flag.set()
            else:
                trial.do_retry()


# ------------------------------------------------------------------------------
def print_usage():
    print('\nUsage: python trial_run.py <network.json> [config.json]')
    print('\nConfig defaults:')
    print(json.dumps(DEFAULT_CONFIG, indent=2))


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    network_file = sys.argv[1]
    config_file  = sys.argv[2] if len(sys.argv) > 2 else None
    config       = load_config(config_file)

    # Load network once just to get sizes for the figure
    tmp_net = GeneticAlgorithm.load_best_network(network_file)

    # Launch bot in background daemon thread
    bot_t = threading.Thread(
        target=_bot_thread_main,
        args=(network_file, config),
        daemon=True,
    )
    bot_t.start()

    # Build matplotlib figure on the MAIN thread (required on Windows/macOS)
    fig, ax_inp, ax_hid, ax_out, meta_txt = _build_figure(
        tmp_net.input_size, tmp_net.hidden_size, tmp_net.output_size,
    )

    plt.ion()
    plt.show(block=False)

    # Main loop: drain viz queue and redraw at ~20 fps
    try:
        while not _quit_flag.is_set():
            latest = None
            while True:
                try:
                    latest = _viz_queue.get_nowait()
                except queue.Empty:
                    break

            if latest is not None:
                activations, meta = latest
                _redraw(ax_inp, ax_hid, ax_out, meta_txt,
                        activations, meta, tmp_net.hidden_size)
                fig.canvas.draw_idle()

            plt.pause(0.05)   # 20 fps; also pumps the Tk event loop

    except KeyboardInterrupt:
        pass
    finally:
        _quit_flag.set()
        plt.close('all')
        print('\nGoodbye!')


if __name__ == '__main__':
    main()