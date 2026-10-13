# FORGE

Code for **Real-Time and Risk-Aware UAV Swarm Coordination at the Web Edge:
Distilling Offline Semantic Reasoning into Decentralized Policies**.

FORGE brings trajectory-derived semantic guidance into multi-agent policy
learning without placing a language model on the real-time control path.
The framework is instantiated with MAPPO for cooperative packet routing
and transmit-power allocation in UAV swarm communication.

## Research overview

UAV swarms can provide mobile relay and gateway infrastructure for
Web-connected sensing and latency-sensitive services. Their controllers
must coordinate traffic and power under changing link quality, traffic
pressure, and delivery deadlines. A locally attractive allocation can
create congestion and delayed failures that are difficult to attribute
to individual decisions.

Our paper investigates whether semantic descriptions of past trajectories
can improve this learning process while preserving lightweight decentralized
execution. FORGE uses an offline language-model writer to organize selected
trajectories into structured records of state patterns, outcomes, and
allocation recommendations. During training, relevant records define a
bounded semantic reward bonus; an auxiliary risk critic estimates future
deadline-exceedance severity. The resulting optimization signals influence
the learned policy through MAPPO updates.

The key distinction is where semantic knowledge is used: **memory shapes
the training reward, not the actor input or the executed action**. After
training, each UAV acts from its local observation using only the frozen
actor. No LLM, memory bank, retrieval index, or centralized critic is
required at deployment.

## Method at a glance

1. **Translate trajectories.** Express link quality, traffic pressure, and
   deadline conditions as compact semantic descriptions.
2. **Write and validate records.** Use a local Qwen writer or the template
   backend to produce structured recommendations; reject invalid schemas
   and exclude low-confidence records from guidance.
3. **Retrieve training context.** Match records using semantic similarity,
   constraint-tag overlap, and an age penalty.
4. **Guide policy optimization.** Compare sampled packet/power allocations
   with the retrieved reference to compute a bounded reward bonus, then
   update the observation-only actor and centralized training critics.
5. **Deploy the actor alone.** Load the actor checkpoint and generate feasible
   packet and power allocations directly from local observations.

The implementation supports FORGE, a no-risk ablation, and a MAPPO control,
as well as configurable semantic reward strength and held-out actor-only
evaluation. Training traces retain the retrieved record IDs and reward
components so that semantic guidance can be inspected.

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
