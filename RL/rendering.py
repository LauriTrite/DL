import gym

# Initialisiere die Pong-Umgebung
env = gym.make("ALE/Pong-v5", render_mode='human')

# Setze die Umgebung zurück
observation = env.reset()

while True:
    action = env.action_space.sample()  # Wähle eine zufällige Aktion
    result = env.step(action)  # Führe die Aktion aus und erhalte Feedback
    observation, reward, done, info = result[:4]


    #observation, reward, done, info = env.step(action)  # Führe die Aktion aus und erhalte Feedback
    
    if done:
        observation = env.reset()  # Setze die Umgebung zurück, wenn das Spiel vorbei ist

env.close()  # Schließe die Umgebung, wenn du fertig bist
