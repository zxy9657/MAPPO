    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class PettingZooRunner(Runner):
    def __init__(self, config):
        super(PettingZooRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_idx, agent in enumerate(self.envs.agents):
                    self.trainer[agent_idx].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, truncs, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_idx, agent in enumerate(self.envs.agents):
                        train_infos[agent_idx].update({"average_episode_rewards": np.mean(self.buffer[agent_idx].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, infos = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(np.concatenate(list(o.values())))
        share_obs = np.array(share_obs)

        for agent_idx, agent in enumerate(self.envs.agents):
            if not self.use_centralized_V:
                share_obs = np.array([o[agent] for o in obs])
            self.buffer[agent_idx].share_obs[0] = share_obs.copy()
            self.buffer[agent_idx].obs[0] = np.array([o[agent] for o in obs]).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_idx, agent in enumerate(self.envs.agents):
            self.trainer[agent_idx].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_idx].policy.get_actions(self.buffer[agent_idx].share_obs[step],
                                                            self.buffer[agent_idx].obs[step],
                                                            self.buffer[agent_idx].rnn_states[step],
                                                            self.buffer[agent_idx].rnn_states_critic[step],
                                                            self.buffer[agent_idx].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            action_env = {}
            for agent_idx, agent in enumerate(self.envs.agents):
                action_env[agent] = actions[i, agent_idx].squeeze()
            actions_env.append(action_env)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = np.array([list(d.values()) for d in dones])

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(np.concatenate(list(o.values())))
        share_obs = np.array(share_obs)

        for agent_idx, agent in enumerate(self.envs.agents):
            if not self.use_centralized_V:
                share_obs = np.array([o[agent] for o in obs])

            self.buffer[agent_idx].insert(share_obs,
                                        np.array([o[agent] for o in obs]),
                                        rnn_states[:, agent_idx],
                                        rnn_states_critic[:, agent_idx],
                                        actions[:, agent_idx],
                                        action_log_probs[:, agent_idx],
                                        values[:, agent_idx],
                                        np.array([r[agent] for r in rewards]).reshape(-1, 1),
                                        masks[:, agent_idx])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_infos = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_actions = []
            for agent_idx, agent in enumerate(self.envs.agents):
                self.trainer[agent_idx].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_idx].policy.act(np.array([o[agent] for o in eval_obs]),
                                                                                eval_rnn_states[:, agent_idx],
                                                                                eval_masks[:, agent_idx],
                                                                                deterministic=True)

                eval_actions.append(eval_action.detach().cpu().numpy())
                eval_rnn_states[:, agent_idx] = _t2n(eval_rnn_state)

            eval_actions = np.array(eval_actions).transpose(1, 0, 2)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_action_env = {}
                for agent_idx, agent in enumerate(self.envs.agents):
                    eval_action_env[agent] = eval_actions[i, agent_idx].squeeze()
                eval_actions_env.append(eval_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_truncs, eval_infos = self.eval_envs.step(eval_actions_env)

            eval_rewards = np.array([list(r.values()) for r in eval_rewards])
            eval_dones = np.array([list(d.values()) for d in eval_dones])

            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_idx, agent in enumerate(self.envs.agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_idx], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_idx + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  
