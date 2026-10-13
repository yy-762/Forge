"""MAPPO updates with training-only semantic reward and future-risk GAE."""
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from .env import UAVSwarmEnv
from .policy import Actor, Critic
from .semantic import Memory, Writer, context, guidance, serialize


@dataclass
class Config:
    episodes: int = 1000
    steps: int = 20
    seed: int = 7
    method: str = 'forge'
    backend: str = 'template'
    model: str | None = None
    device: str = 'cpu'
    hidden: int = 192
    lr: float = .0007
    gamma: float = .99
    gae_lambda: float = .95
    ppo_epochs: int = 5
    clip: float = .2
    entropy: float = .004
    risk_weight: float = .08
    semantic_bound: float = .5
    memory_every: int = 20
    capacity: int = 5000
    top_k: int = 3
    deadline: float = 50.
    delay_scale: float = .5
    output: str = 'outputs/forge_seed7'


def base_metrics(info, deadline, delay_scale):
    delay = float(info['avg_vehicle_max_delay_ms']) * delay_scale
    pressure = float(info['avg_peak_power_usage'])
    severity = float(np.clip((delay - deadline) / deadline, 0, 1))
    reward = float(np.exp(-delay / deadline) + .3 * (1 - pressure) - severity)
    return reward, severity, delay, pressure


def gae(rewards, values, dones, gamma, trace):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running = 0.
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.
        mask = 1. - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running = delta + gamma * trace * mask * running
        advantages[t] = running
    return advantages + np.asarray(values, dtype=np.float32), advantages


def normalize(vector):
    return (vector - vector.mean()) / (vector.std(unbiased=False) + 1e-8)


def discounted_costs(costs, dones, gamma):
    targets = np.zeros(len(costs), dtype=np.float32)
    future = 0.
    for t in reversed(range(len(costs))):
        future = costs[t] + gamma * future * (1. - float(dones[t]))
        targets[t] = future
    return targets


def update(actor, critic, optimizer, rollout, cfg):
    tensors = {key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=cfg.device)
               for key, value in rollout.items()}
    returns, advantages = gae(rollout['reward'], rollout['value'], rollout['done'], cfg.gamma, cfg.gae_lambda)
    _, risk_adv = gae(rollout['risk'], rollout['risk_value'], rollout['done'], cfg.gamma, cfg.gae_lambda)
    risk_returns = discounted_costs(rollout['risk'], rollout['done'], cfg.gamma)
    advantages = normalize(torch.as_tensor(advantages, device=cfg.device))
    risk_adv = normalize(torch.as_tensor(risk_adv, device=cfg.device))
    weight = cfg.risk_weight if cfg.method == 'forge' else 0.
    joint_advantage = advantages - weight * risk_adv
    returns = torch.as_tensor(returns, device=cfg.device)
    risk_returns = torch.as_tensor(risk_returns, device=cfg.device)
    for _ in range(cfg.ppo_epochs):
        obs = tensors['obs'].flatten(0, 1)
        packet_dist, power_dist = actor.distributions(obs)
        logp = (packet_dist.log_prob(tensors['packets'].flatten(0, 1)) +
                power_dist.log_prob(tensors['power'].flatten(0, 1))).reshape_as(tensors['logp'])
        ratio = (logp - tensors['logp']).exp()
        advantage = joint_advantage[:, None].expand_as(ratio)
        policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
        entropy = (packet_dist.entropy() + power_dist.entropy()).mean()
        value, risk_value = critic(tensors['state'])
        value_loss = .5 * F.mse_loss(value, returns)
        risk_loss = .5 * F.mse_loss(risk_value, risk_returns) if weight else value_loss.new_zeros(())
        loss = policy_loss - cfg.entropy * entropy + value_loss + risk_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PPO objective')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.)
        optimizer.step()
    return dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                risk_loss=float(risk_loss.detach()))


def run(cfg):
    if min(cfg.episodes, cfg.steps, cfg.memory_every, cfg.ppo_epochs, cfg.top_k, cfg.capacity) < 1:
        raise ValueError('Episode, horizon, memory and update counts must be positive')
    if cfg.deadline <= 0 or cfg.delay_scale <= 0 or cfg.semantic_bound < 0 or cfg.risk_weight < 0:
        raise ValueError('Invalid reward configuration')
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    env = UAVSwarmEnv(seed=cfg.seed)
    env.episode_length = cfg.steps
    actor_cfg = dict(obs_dim=env.obs_dim, links=env.n_selected_bs, hidden=cfg.hidden, packets=env.total_packet_budget)
    actor = Actor(**actor_cfg).to(cfg.device)
    critic = Critic(env.state_dim, cfg.hidden).to(cfg.device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)
    semantic_enabled = cfg.method != 'mappo' and cfg.semantic_bound > 0
    bank = Memory(cfg.capacity, cfg.top_k) if semantic_enabled else None
    writer = Writer(cfg.backend, cfg.model) if semantic_enabled else None
    metrics, semantic_log = [], []
    for episode in range(cfg.episodes):
        obs, state = env.reset()
        rollout = {key: [] for key in ['obs', 'state', 'packets', 'power', 'logp', 'reward', 'risk', 'done', 'value', 'risk_value']}
        histories = [[] for _ in obs]
        delays, pressures, bases, bonuses = [], [], [], []
        previous_delay = 0.
        for step in range(cfg.steps):
            obs = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                packets, power, logp = actor.act(torch.as_tensor(obs, device=cfg.device))
                value, risk_value = critic(torch.as_tensor(state, device=cfg.device))
            packets_np, power_np = packets.cpu().numpy(), power.cpu().numpy()
            agent_bonuses, details, contexts = [], [], []
            for agent in range(len(obs)):
                text, tags = context(obs[agent], previous_delay, cfg.deadline)
                contexts.append((text, tags))
                retrieved = bank.retrieve(text, tags, episode) if bank is not None else []
                bonus, detail = guidance(packets_np[agent], power_np[agent], obs[agent], retrieved, cfg.semantic_bound)
                agent_bonuses.append(bonus); details.append(detail)
            next_obs, next_state, _, done, info = env.step(power_np, packets_np.astype(int))
            base, risk, delay, pressure = base_metrics(info, cfg.deadline, cfg.delay_scale)
            bonus = float(np.mean(agent_bonuses))
            transition = dict(obs=obs, state=state, packets=packets_np, power=power_np,
                              logp=logp.cpu().numpy(), reward=base + bonus, risk=risk,
                              done=float(done), value=float(value), risk_value=float(risk_value))
            for key in rollout:
                rollout[key].append(transition[key])
            for agent, (text, tags) in enumerate(contexts):
                evidence = dict(context=text, tags=tags, packets=packets_np[agent].tolist(),
                                power=power_np[agent].tolist(), delay=delay, severity=risk)
                histories[agent].append(evidence)
                if semantic_enabled:
                    semantic_log.append(dict(episode=episode, step=step, agent=agent,
                        base_reward=base, training_reward=base + bonus, joint_bonus=bonus,
                        agent_bonus=agent_bonuses[agent], packets=evidence['packets'], power=evidence['power'],
                        **details[agent]))
            delays.append(delay); pressures.append(pressure); bases.append(base); bonuses.append(bonus)
            obs, state, previous_delay = next_obs, next_state, delay
            if done:
                break
        losses = update(actor, critic, optimizer, rollout, cfg)
        if writer is not None and (episode + 1) % cfg.memory_every == 0:
            for history in histories:
                bank.add(writer.write(history, episode))
        metrics.append(dict(episode=episode, base_reward=float(np.mean(bases)),
                            training_reward=float(np.mean(bases) + np.mean(bonuses)),
                            semantic_bonus=float(np.mean(bonuses)), delay=float(np.mean(delays)),
                            power_pressure=float(np.mean(pressures)),
                            threshold_event=int(np.mean(delays) > cfg.deadline),
                            memory_size=len(bank.records) if bank else 0, **losses))
        if episode == 0 or (episode + 1) % 20 == 0 or episode + 1 == cfg.episodes:
            print(json.dumps(metrics[-1]), flush=True)
    with (out / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=metrics[0])
        csv_writer.writeheader(); csv_writer.writerows(metrics)
    if bank is not None:
        (out / 'memory.json').write_text(json.dumps([serialize(r) for r, _ in bank.records], indent=2), encoding='utf-8')
        with (out / 'semantic_trace.jsonl').open('w', encoding='utf-8') as stream:
            for row in semantic_log:
                stream.write(json.dumps(row) + '\n')
    # Only the actor and numeric metadata cross the deployment boundary.
    torch.save(dict(actor={key: value.detach().cpu() for key, value in actor.state_dict().items()},
                    actor_config=actor_cfg, evaluation_config=dict(steps=cfg.steps, deadline=cfg.deadline,
                    delay_scale=cfg.delay_scale)), out / 'actor.pt')
    return out
