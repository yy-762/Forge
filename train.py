import argparse
from forge.train import Config, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FORGE with an observation-only actor')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--method', choices=['forge', 'no_risk', 'mappo'], default='forge')
    parser.add_argument('--backend', choices=['template', 'qwen'], default='template')
    parser.add_argument('--model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden', type=int, default=192)
    parser.add_argument('--semantic-bound', type=float, default=.5)
    parser.add_argument('--risk-weight', type=float, default=.08)
    parser.add_argument('--memory-every', type=int, default=20)
    parser.add_argument('--ppo-epochs', type=int, default=5)
    parser.add_argument('--deadline', type=float, default=50)
    parser.add_argument('--delay-scale', type=float, default=.5)
    parser.add_argument('--output', default='outputs/forge_seed7')
    args = parser.parse_args()
    run(Config(**vars(args)))
