# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment (0 = pick randomly)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo-humanoid-baselines"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_group: str = ""
    """the group name for wandb runs"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    sparse: bool = False
    """if toggled, rewards are accumulated and delivered only at episode end"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class SparseRewardWrapper(gym.Wrapper):
    """Accumulates dense rewards and delivers the sum only at episode end."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._accumulated = np.zeros(self.env.num_envs, dtype=float)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        self._accumulated += np.asarray(reward, dtype=float)
        sparse = np.where(done, self._accumulated, 0.0)
        self._accumulated[done] = 0.0
        return obs, sparse, terminated, truncated, info


class EnvPoolAdapter(gym.Wrapper):
    """Adapts an envpool env to the gymnasium VectorEnv interface expected by the training loop."""

    def __init__(self, env, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs
        self.is_vector_env = True
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self._ep_returns = np.zeros(num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self._ep_returns[:] = 0
        self._ep_lengths[:] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_returns += reward
        self._ep_lengths += 1
        done = terminated | truncated
        if done.any():
            final_info = [None] * self.num_envs
            for i in np.where(done)[0]:
                final_info[i] = {
                    "episode": {
                        "r": float(self._ep_returns[i]),
                        "l": int(self._ep_lengths[i]),
                    }
                }
            info["final_info"] = final_info
            self._ep_returns[done] = 0
            self._ep_lengths[done] = 0
        return obs, reward, terminated, truncated, info



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.seed == 0:
        args.seed = random.randint(1, 2**31 - 1)
    episode_mode = args.num_steps == 0
    if not episode_mode:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group if args.wandb_group else None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(args.env_id, env_type="gymnasium", num_envs=args.num_envs, seed=args.seed)
    envs = EnvPoolAdapter(envs, args.num_envs)
    if args.sparse:
        envs = SparseRewardWrapper(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Fixed-step mode: pre-allocate rollout buffers once and reuse across iterations.
    if not episode_mode:
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    iteration = 0
    while global_step < args.total_timesteps:
        iteration += 1

        if args.anneal_lr:
            frac = max(0.0, 1.0 - global_step / args.total_timesteps)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        if episode_mode:
            # Collect exactly one complete episode per env.
            # Steps from already-finished envs are discarded so the buffer
            # contains no partial episodes.
            per_env_obs      = [[] for _ in range(args.num_envs)]
            per_env_actions  = [[] for _ in range(args.num_envs)]
            per_env_logprobs = [[] for _ in range(args.num_envs)]
            per_env_rewards  = [[] for _ in range(args.num_envs)]
            per_env_values   = [[] for _ in range(args.num_envs)]
            finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

            while not finished.all():
                global_step += int((~finished).sum().item())
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminations, truncations)
                reward_t = torch.tensor(reward, device=device).view(-1)

                for i in range(args.num_envs):
                    if not finished[i]:
                        per_env_obs[i].append(next_obs[i])
                        per_env_actions[i].append(action[i])
                        per_env_logprobs[i].append(logprob[i])
                        per_env_rewards[i].append(reward_t[i])
                        per_env_values[i].append(value.flatten()[i])
                        if done_np[i]:
                            finished[i] = True

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                next_obs = torch.Tensor(next_obs_np).to(device)

            # Explicitly reset all envs so the next iteration starts from clean
            # episode boundaries for every env.
            next_obs, _ = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs, device=device)

            # Per-episode GAE with no value bootstrap (episodes are complete).
            b_obs_chunks, b_act_chunks, b_lp_chunks = [], [], []
            b_adv_chunks, b_ret_chunks, b_val_chunks = [], [], []
            for i in range(args.num_envs):
                R = torch.stack(per_env_rewards[i])
                V = torch.stack(per_env_values[i])
                H_i = len(R)
                adv = torch.zeros(H_i, device=device)
                lastgaelam = 0
                for t in reversed(range(H_i)):
                    nextnonterminal = 0.0 if t == H_i - 1 else 1.0
                    nextvalue       = 0.0 if t == H_i - 1 else V[t + 1]
                    delta = R[t] + args.gamma * nextvalue * nextnonterminal - V[t]
                    adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                b_obs_chunks.append(torch.stack(per_env_obs[i]))
                b_act_chunks.append(torch.stack(per_env_actions[i]))
                b_lp_chunks.append(torch.stack(per_env_logprobs[i]))
                b_adv_chunks.append(adv)
                b_ret_chunks.append(adv + V)
                b_val_chunks.append(V)

            b_obs        = torch.cat(b_obs_chunks)
            b_actions    = torch.cat(b_act_chunks)
            b_logprobs   = torch.cat(b_lp_chunks)
            b_advantages = torch.cat(b_adv_chunks)
            b_returns    = torch.cat(b_ret_chunks)
            b_values     = torch.cat(b_val_chunks)
            H = max(len(per_env_obs[i]) for i in range(args.num_envs))
            batch_size = b_obs.shape[0]
            minibatch_size = max(1, batch_size // args.num_minibatches)
        else:
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

            # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            H = args.num_steps
            batch_size = args.batch_size
            minibatch_size = args.minibatch_size

            # GAE with bootstrap for partial trajectories at rollout boundary.
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(H)):
                    if t == H - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs   = logprobs.reshape(-1)
            b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns    = returns.reshape(-1)
            b_values     = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
