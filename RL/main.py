from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import gym

# Optionale Zeile, um die Gym-Umgebung zu testen oder zu explorieren (kann entfernt werden, wenn nicht benötigt)
# env_test = gym.make("ALE/Pong-v5", render_mode='human')

# Erstelle die Umgebung und stacke die Frames für eine bessere zeitliche Auflösung für das DQN-Training
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

# Erstelle den DQN-Agenten
model = DQN('CnnPolicy', env, verbose=1, learning_rate=1e-4, buffer_size=10000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.01, max_grad_norm=10)

# Trainiere den Agenten
model.learn(total_timesteps=int(2e6))

# Speichere das trainierte Modell
model.save("dqn_pong")

# Lade ein trainiertes Modell (optional)
model = DQN.load("dqn_pong")

# Evaluierung des Agenten
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Optional: Schließe die Testumgebung, wenn verwendet
# env_test.close()
