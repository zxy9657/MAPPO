import numpy as np
from gym import spaces


class PettingZooWrapper:
    def __init__(self, env):
        self.env = env
        self.n = len(self.env.possible_agents)

        # Image obs space
        if len(self.env.observation_space(self.env.possible_agents[0]).shape) == 3:
            w, h, c = self.env.observation_space(self.env.possible_agents[0]).shape
            self.share_observation_space = {
                agent: spaces.Box(
                low=0, high=255, shape=(self.n * w, h, c), dtype=np.uint8) for agent in self.env.possible_agents}
        # Vector obs space
        else:
            if self.env.observation_space(self.env.possible_agents[0]).__class__.__name__ == 'Discrete':
                share_obs_dim = int(sum([self.env.observation_space(agent).n for agent in self.env.possible_agents]))
            else:
                share_obs_dim = sum([self.env.observation_space(agent).shape[0] for agent in self.env.possible_agents])
            self.share_observation_space = {
                agent: spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for agent in self.env.possible_agents}

        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.env.possible_agents}
        self.action_space = {agent: self.env.action_space(agent) for agent in self.env.possible_agents}

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, actions):
        obs, reward, done, truncation, info = self.env.step(actions)

        done = done if done else {agent: True for agent in self.env.possible_agents}

        return obs, reward, done, truncation, info

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
