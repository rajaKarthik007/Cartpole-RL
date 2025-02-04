import gym
import torch
import torch.nn as nn
import numpy as np

sm = nn.Softmax(dim=1)

HIDDEN_NEURONS = 128
LEARNING_RATE = 0.01
BATCHSIZE = 16

class Policy(nn.Module):
    def __init__(self, observation_size, n_actions):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(observation_size, HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_NEURONS, n_actions)
        )
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight)
    #         nn.init.zeros_(m.bias)
              
    def forward(self, x):
        return self.pipe(x)
    
def iterate_batches(env=gym.make("CartPole-v1"), policynet=Policy(0, 0), batch_size=BATCHSIZE):
    batch = []
    episode = []
    actions = []
    reward = 0.0
    
    obs, _ = env.reset()
    while True:
        with torch.no_grad():
            action_logits = policynet(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        # print(action_logits.shape)
        probs = sm(action_logits)
        # print(probs)
        action = np.random.choice(list(range(env.action_space.n)), p=probs.squeeze(0).numpy())
        actions.append(action)
        obs, r, terminated, truncated, _ = env.step(action)
        reward += r
        episode.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
            batch.append({"episode": episode, "actions": actions, "reward": reward})
            episode, actions, reward = [], [], 0.0
            if len(batch) == batch_size:
                yield batch
                batch = []
                
def filter_batches(batches, percentile):
    pass
    
if __name__ == "__main__":
    #defining the environment
    env = gym.make("CartPole-v1")
    
    #defining the policy network
    policynet = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(params = policynet.parameters(), lr = LEARNING_RATE)
    
    #training loop
    for batch in iterate_batches(env, policynet):
        print(batch[0])
        break
    