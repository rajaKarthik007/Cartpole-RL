import gym
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torchsummary import summary

sm = nn.Softmax(dim=1)
objective = nn.CrossEntropyLoss()

HIDDEN_NEURONS = 128
LEARNING_RATE = 0.01
BATCHSIZE = 16
PERCENTILE = 70

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Running on {device}")

class Policy(nn.Module):
    def __init__(self, observation_size, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.pipe = nn.Sequential(
            nn.Linear(observation_size, HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_NEURONS, n_actions)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, nn.Linear) and m.out_features == self.n_actions:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')  # For final layer
            torch.nn.init.constant_(m.bias, 0.1)
              
    def forward(self, x):
        return self.pipe(x)
    
def iterate_batches(env=gym.make("CartPole-v1"), policynet=Policy(4, 2), batch_size=BATCHSIZE):
    batch = []
    episode = []
    actions = []
    reward = 0.0
    
    obs, _ = env.reset()
    while True:
        with torch.no_grad():
            action_logits = policynet(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
        # print(action_logits.shape)
        probs = sm(action_logits.cpu())
        # print(probs)
        # action = np.random.choice(list(range(env.action_space.n)), p=probs.squeeze(0).numpy())
        action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        actions.append(action)
        episode.append(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        reward += r
        
        if terminated or truncated:
            obs, _ = env.reset()
            batch.append({"episode": episode, "actions": actions, "reward": reward})
            episode, actions, reward = [], [], 0.0
            if len(batch) == batch_size:
                yield batch
                batch = []
                
def filter_batch(batches, percentile=PERCENTILE):
    batches.sort(key= lambda x: x["reward"], reverse=True)
    ret_idx = len(batches) * percentile // 100
    del batches[ret_idx:]
    X, y = [], []
    for batch in batches:
        X.extend(batch["episode"])   
        y.extend(batch["actions"])
    rewards = [batch["reward"] for batch in batches]
    return torch.tensor(np.array(X),dtype=torch.float32, device=device), torch.tensor(np.array(y), dtype=torch.float32, device=device), np.array(rewards).mean()
    
    
    
if __name__ == "__main__":
    #defining the environment
    env = gym.make("CartPole-v1")
    
    #defining the policy network
    policynet = Policy(env.observation_space.shape[0], env.action_space.n)
    policynet = policynet.to(device)
    optimizer = torch.optim.Adam(params = policynet.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter(comment="-cartpole-crossentropy")
    
    #training loop
    iter_no = 0
    for batch in iterate_batches(env, policynet):
        iter_no += 1
        X, y, rewardMean = filter_batch(batch)
        
        optimizer.zero_grad()
        action_logits = policynet(X)
        loss = objective(action_logits, y)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("loss", loss.item(), iter_no)
        writer.add_scalar("reward mean", rewardMean, iter_no)
        print("%d: loss=%.3f, reward_mean=%.1f" % (iter_no, loss.item(), rewardMean))
        if rewardMean > 499:
            print("Solved")
            break
    
    torch.save(policynet.state_dict(), "cartpole_cross_entropy.pth")        
    writer.close()
        

    