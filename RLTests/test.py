
from gym_examples2.gym_examples.envs.AlbertEnv import AlbertEnv
import keyboard

def action_debug():
    # Dictionary to map arrow keys to their corresponding index in the output array
    action = [0, 0, 0]

    # Check for arrow key events
    if keyboard.is_pressed("up"):
        action[1] = 2

    if keyboard.is_pressed("down"):
        action[1] = 1

    if keyboard.is_pressed("left"):
        action[0] = 1

    if keyboard.is_pressed("right"):
        action[0] = 2

    if keyboard.is_pressed("space"):
        action[2] = 1
    return action

env = AlbertEnv()
env.reset()



episodes=10

for ep in range(episodes):
    env.reset()
    done = False
    while not done:
        env.render()
        current_obs, reward, done, info = env.step(action_debug())
        #current_obs, reward, done, info = env.step(env.action_space.sample())



env.close()