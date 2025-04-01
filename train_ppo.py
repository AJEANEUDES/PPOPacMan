import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
import os
import ale_py

# Créer les environnements Atari prétraités automatiquement (frame skip, grayscale, resize, etc.)
env = make_atari_env("ALE/MsPacman-v5", n_envs=8, seed=42)
env = VecFrameStack(env, n_stack=4)

# Créer un environnement d'évaluation (même prétraitement)
eval_env = make_atari_env("ALE/MsPacman-v5", n_envs=1, seed=43)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Définir le callback pour sauvegarder le meilleur modèle
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs",
    eval_freq=25000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1
)

# Créer le modèle PPO avec des hyperparamètres adaptés à l'apprentissage rapide
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=1024,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs",
    verbose=1
)

# Entraîner pour 5 millions de timesteps (en ~2h)
total_timesteps = 5_000_000
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Sauvegarder le modèle final
model.save("ppo_mspacman_5M")
