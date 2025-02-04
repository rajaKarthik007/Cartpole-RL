import gym
from cartpole_crossentropy import Policy
import torch

env = gym.make("CartPole-v1", render_mode="human")
model = Policy(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load("cartpole_cross_entropy.pth", weights_only=True))
num_episodes = 1

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        with torch.no_grad():
            logits = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        action = torch.argmax(logits, dim=1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        done = terminated or truncated

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()