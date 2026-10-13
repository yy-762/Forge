# FORGE

Minimal implementation of training-time semantic reward guidance for UAV
swarm communication. The decentralized actor receives local observations;
the return and risk critics receive centralized state during training.
Retrieved semantic records contribute a bounded reward bonus. Deployment
loads only the actor checkpoint.

## Install

Use Python 3.11 or newer in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python train.py --episodes 3 --steps 4 --memory-every 1 --ppo-epochs 1 --output outputs/smoke
python evaluate.py --checkpoint outputs/smoke/actor.pt --episodes 5 --output outputs/smoke/evaluation.csv
```

The quick start uses an explicit template writer. For local Qwen generation,
install the optional backend and supply a GGUF model obtained separately:

```bash
python -m pip install -r requirements-llm.txt
python train.py --backend qwen --model models/Qwen_Qwen3.5-9B-Q4_K_M.gguf --device cuda --output outputs/qwen_seed7
```

The Qwen backend requires a working local llama-cpp-python installation and
model. Invalid JSON/schema records are rejected. Template and Qwen runs are
different writer settings, recorded in `config.json`.

## Comparisons

```bash
python train.py --method mappo --output outputs/mappo_seed7
python train.py --method no_risk --output outputs/no_risk_seed7
python train.py --method forge --semantic-bound 0.3 --output outputs/forge_beta03_seed7
```

Use `--seed` for independently trained policies. Evaluate each frozen checkpoint
separately using held-out seeds. Aggregate delay and power within training seeds;
the episode violation rate counts episodes whose mean delay exceeds the deadline.
Default evaluation samples actions. Use `--deterministic` for a separate
deterministic-action evaluation protocol.

## Outputs

- `actor.pt`: observation-only actor weights and numeric deployment metadata.
- `training.csv`: base/training reward, semantic bonus, delay, power, and losses.
- `memory.json`: validated training records.
- `semantic_trace.jsonl`: record IDs, alignment, reference, and reward terms.
- `config.json`: run settings.

The simulator has eight agents, sixteen access/relay nodes, three links per
agent, ten packets per decision and a 200 mW power budget. Packet allocation
uses a Multinomial head and power allocation a Dirichlet head. The numeric
delay scale and threshold are configurable with `--delay-scale` and `--deadline`.

Defaults: hidden width 192, Adam learning rate 0.0007, five PPO epochs,
discount 0.99, GAE 0.95, entropy coefficient 0.004, semantic bound 0.5,
risk weight 0.08, top-3 retrieval and capacity 5,000. Records refresh every
20 episodes. Retrieval uses normalized hashed text keys, constraint-tag
overlap and an age penalty. Candidate ranks are ordered by current link quality.
Finite vocabulary tendencies map to fixed normalized allocation references.

## Files

```text
train.py          Training command
evaluate.py       Actor-only evaluation command
forge/policy.py   Actor and centralized training critics
forge/semantic.py Writer, validation, retrieval and reward alignment
forge/train.py    Rollout collection, reward construction and PPO updates
forge/env.py      Communication simulator
tests/            Mechanism and deployment checks
```

```bash
python -m unittest discover -s tests -v
```
