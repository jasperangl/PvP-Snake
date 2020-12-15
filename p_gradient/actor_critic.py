import torch as T
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(T.nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = T.nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = T.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = T.nn.Linear(self.fc2_dims, n_actions)
        self.v = T.nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)


class ActorCriticAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """

    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=4):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                               layer2_size, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs
        return action.item()

    def learn(self, state, reward, new_state, done):
        # Getting rid of intermediate steps
        self.actor_critic.optimizer.zero_grad()

        _, critic_value = self.actor_critic.forward(state)
        _, new_critic_value = self.actor_critic.forward(new_state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma * new_critic_value - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = 0.5 * delta ** 2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

        self.actor_critic.eval()
