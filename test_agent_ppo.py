import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import ale_py
import os

# Charger le modèle
model = PPO.load("ppo_mspacman_5M")

# Environnement avec affichage visuel (human mode)
def make_env():
    def _init():
        env = gym.make("ALE/MsPacman-v5", render_mode="human")
        env = AtariWrapper(env, terminal_on_life_loss=True, clip_reward=False)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Faire jouer l'agent jusqu’à 1M de steps
max_steps = 5_000_000
total_steps = 0
total_reward = 0
episode_count = 0
start_time = time.time()

obs = env.reset()
done = False

while total_steps < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    total_steps += 1
    total_reward += reward[0]

    if done:
        episode_count += 1
        print(f"Épisode {episode_count} terminé | Score : {total_reward:.2f} | Total steps : {total_steps}")
        obs = env.reset()
        total_reward = 0

env.close()
duration = time.time() - start_time
print(f"\nExécution terminée après {total_steps} steps")
print(f"Durée totale : {duration:.1f} secondes ({duration / 60:.2f} minutes)")
print(f"Nombre total d’épisodes joués : {episode_count}")
