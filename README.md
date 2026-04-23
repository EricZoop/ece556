# GMU ECE 556 - Minecraft Neuromorphic Parkour

Set up a 1.19.4 vanilla minecraft server

https://www.minecraft.net/en-us/article/minecraft-java-edition-1-19-4#:~:text=Minecraft%20server%20jar

Replace world with zip file in server folder, apply server.properties located in this repo and start server.jar

As admin run enter following commands:

```
/setworldspawn -15 -63 0
/gamerule spawnRadius 0
/team add bots "bots"
/team modify bots collisionRule never
/team modify bots friendlyFire false
/op @a
/gamerule randomTickSpeed 0

/gamerule fallDamage false
/gamerule doImmediateRespawn true
/gamerule sendCommandFeedback false
/kill @e[type=player]
```

Clone repository
```cmd
git clone https://github.com/EricZoop/ece556
```

Change main.py target host IP to your LAN setup
```python
def create_bot(self, index):
    bot = mineflayer.createBot({
        'host':     '192.168.1.118', # CHANGE ME
        'port':     25565,
        'username': f'Bot_{index}',
        'version':  '1.19.4',
    })
```

Create enviorment
```cmd
python -m venv venv
```

Activate
```cmd
venv\Scripts\activate
```

Install dependencies
```cmd
pip install -r requirements.txt
```

Run
```cmd
python main.py
```



### dice setup
```minecraft
/summon marker ~ ~ ~ {Tags:["rng_source","lime_source"]}
/scoreboard players set @e[tag=lime_source] rng 0
/summon marker ~ ~ ~ {Tags:["rng_source","red_source"]}
/scoreboard players set @e[tag=red_source] rng 1
```

### string of obstacle locations
```minecraft
/summon marker 0 -64 1 {Tags:["block_pos"]}
/summon marker 0 -64 2 {Tags:["block_pos"]}
/summon marker 0 -64 3 {Tags:["block_pos"]}
/summon marker 0 -64 4 {Tags:["block_pos"]}
/summon marker 0 -64 5 {Tags:["block_pos"]}
/summon marker 0 -64 6 {Tags:["block_pos"]}
/summon marker 0 -64 7 {Tags:["block_pos"]}
/summon marker 0 -64 8 {Tags:["block_pos"]}
/summon marker 0 -64 9 {Tags:["block_pos"]}
```

### command block chain
change `deaths=1` depending on your training setup
```yaml
block_1:
    - Type: Repeating
    - Condition: Unconditional
    - Redstone: Always Active
    - Command: execute if entity @a[scores={deaths=30..}] as @e[tag=block_pos] at @s as @e[type=marker,tag=rng_source,sort=random,limit=1] run scoreboard players operation @e[tag=block_pos,distance=..0.5,limit=1] rng = @s rng

block_2
    - Type: Chain
    - Condition: Unconditional
    - Redstone: Always Active
    - Command: execute if entity @a[scores={deaths=30..}] as @e[tag=block_pos,scores={rng=0}] at @s run setblock ~ ~ ~ lime_wool

block_3
    - Type: Chain
    - Condition: Unconditional
    - Redstone: Always Active
    - Command: execute if entity @a[scores={deaths=30..}] as @e[tag=block_pos,scores={rng=1}] at @s run setblock ~ ~ ~ air

block_4
    - Type: Chain
    - Condition: Unconditional
    - Redstone: Always Active
    - Command: scoreboard players reset @a[scores={deaths=30..}] deaths

```

### cleanup
```minecraft
/kill @e[type=marker,tag=block_pos]
/kill @e[type=marker,tag=rng_source]
```

#### bonus:
```minecraft
/summon firework_rocket ~ ~1 ~ {LifeTime:12,FireworksItem:{id:"minecraft:firework_rocket",Count:1,tag:{Fireworks:{Explosions:[{Type:2,Colors:[I;16776960]}]}}}}
```