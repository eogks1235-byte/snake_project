"""A: 포식자 PPO 학습 — prey 정책과 함께 arms race.

학습 옵션 1) 포식자만 학습 (prey는 진화 baseline):
  python -m neural.ppo_predator --steps 2000000

학습 옵션 2) prey도 동시에 PPO (양쪽 arms race):
  python -m neural.ppo_predator --steps 2000000 --prey-policy recordings/neural/ppo_world_policy.pt

평가:
  python -m neural.ppo_predator --eval --eval-seeds 10
"""
import sys
import time
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.optim as optim

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neural.world import (World, PPO_PRED_OBS_DIM, PPO_PRED_ACT_DIM)
    from neural.torch_brain import TorchBrain
    from neural.ppo_world import (compute_per_creature_gae, ppo_update,
                                   ROLLOUT_TICKS, HIDDEN, LR)
    from neural.ppo_world import load_policy as load_prey_policy
else:
    from .world import (World, PPO_PRED_OBS_DIM, PPO_PRED_ACT_DIM)
    from .torch_brain import TorchBrain
    from .ppo_world import (compute_per_creature_gae, ppo_update,
                             ROLLOUT_TICKS, HIDDEN, LR)
    from .ppo_world import load_policy as load_prey_policy


def predator_policy_path() -> Path:
    p = Path(__file__).resolve().parent.parent / 'recordings' / 'neural' / 'ppo_predator_policy.pt'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def make_predator_world(seed: int, pred_policy: TorchBrain,
                         prey_policy=None, arch='mlp'):
    w = World(seed)
    w.predator_policy = pred_policy
    w.disable_pred_reproduction = True
    if prey_policy is not None:
        w.shared_policy = prey_policy
        w.disable_reproduction = True
    w.ppo_buffer_max = None
    return w


def collect_predator_rollout(world, ticks):
    world._pred_ppo_buffer = []
    for _ in range(ticks):
        if not world.predators:
            world._maintain_predator_population()
        world.step()
    transitions = world._pred_ppo_buffer
    world._pred_ppo_buffer = []
    return transitions


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[PPO-PRED] device={device} steps={args.steps} arch={args.arch}')

    pred_policy = TorchBrain(PPO_PRED_OBS_DIM, args.hidden, PPO_PRED_ACT_DIM,
                              arch=args.arch).to(device)
    optimizer = optim.Adam(pred_policy.parameters(), lr=args.lr)

    prey_policy = None
    if args.prey_policy:
        prey_policy = load_prey_policy(Path(args.prey_policy), device)
        print(f'[PPO-PRED] arms race mode — prey policy loaded')

    world = make_predator_world(args.seed, pred_policy, prey_policy, args.arch)
    total_steps = 0
    iteration = 0
    t0 = time.perf_counter()
    while total_steps < args.steps:
        transitions = collect_predator_rollout(world, ROLLOUT_TICKS)
        obs, act, logp, adv, ret = compute_per_creature_gae(transitions)
        if obs.shape[0] == 0:
            world = make_predator_world(args.seed + iteration, pred_policy,
                                          prey_policy, args.arch)
            continue
        loss = ppo_update(pred_policy, optimizer, obs, act, logp, adv, ret)
        total_steps += obs.shape[0]
        iteration += 1
        mean_r = float(np.mean([np.mean(b['rewards']) for b in transitions
                                if b['rewards'] is not None]))
        dt = time.perf_counter() - t0
        if iteration % 5 == 0 or total_steps >= args.steps:
            print(f'[PPO-PRED] iter={iteration:4d} steps={total_steps:7d} '
                  f'mean_r={mean_r:+5.2f} kills={world.kills} '
                  f'preds={len(world.predators)} '
                  f'pl={loss["policy"]:+.3f} ent={loss["entropy"]:.2f} ({dt:.0f}s)')
        if iteration % 20 == 0:
            world = make_predator_world(args.seed + iteration * 7, pred_policy,
                                          prey_policy, args.arch)

    torch.save({
        'state_dict': pred_policy.state_dict(),
        'config': {'arch': args.arch, 'hidden': args.hidden,
                   'obs_dim': PPO_PRED_OBS_DIM, 'act_dim': PPO_PRED_ACT_DIM},
    }, predator_policy_path())
    print(f'[PPO-PRED] saved → {predator_policy_path()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden', type=int, default=HIDDEN)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--arch', choices=['mlp', 'gru'], default='mlp')
    parser.add_argument('--prey-policy', type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
