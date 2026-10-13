"""Frozen evaluation imports neither semantic memory nor training critics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from forge.env import UAVSwarmEnv
from forge.policy import load_actor


def evaluate(checkpoint, episodes=100, seed=10007, device='cpu', deterministic=False, output='outputs/evaluation.csv'):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    torch.manual_seed(seed); torch.set_num_threads(1)
    actor, cfg = load_actor(checkpoint, device)
    env = UAVSwarmEnv(seed=seed)
    env.episode_length = cfg['steps']
    rows = []
    with torch.inference_mode():
        for episode in range(episodes):
            obs, _ = env.reset()
            delays, pressures = [], []
            for _ in range(cfg['steps']):
                obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
                packets, power, _ = actor.act(obs_t, deterministic)
                obs, _, _, done, info = env.step(power.cpu().numpy(), packets.cpu().numpy().astype(int))
                delays.append(info['avg_vehicle_max_delay_ms'] * cfg['delay_scale'])
                pressures.append(info['avg_peak_power_usage'])
                if done:
                    break
            rows.append(dict(episode=episode, delay=float(np.mean(delays)),
                             power_pressure=float(np.mean(pressures)),
                             threshold_event=int(np.mean(delays) > cfg['deadline'])))
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = dict(episodes=episodes, delay=float(np.mean([r['delay'] for r in rows])),
                  power_pressure=float(np.mean([r['power_pressure'] for r in rows])),
                  events=sum(r['threshold_event'] for r in rows))
    result['violation_rate'] = result['events'] / episodes
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the deployed FORGE actor')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10007)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output', default='outputs/evaluation.csv')
    evaluate(**vars(parser.parse_args()))
