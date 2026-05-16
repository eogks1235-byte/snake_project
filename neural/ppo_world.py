"""다중 에이전트 PPO — 전체 World에서 모든 prey가 공유 정책으로 동작.

진화 baseline과 같은 환경(World)에서 PPO 정책 학습 → apples-to-apples 비교.

학습:
  python -m neural.ppo_world --steps 200000

평가 (Held-out):
  python -m neural.ppo_world --eval --eval-seeds 10

비교 (진화 vs PPO, 같은 환경):
  python -m neural.ppo_world --compare --n-seeds 10 --ticks 1500

핵심 아이디어:
  - 50~150마리가 매 틱 동시 행동
  - 모두 같은 정책 (parameter sharing) — 개체별 차이는 위치/에너지/시야만
  - 각자 trajectory 수집 (사망 시 종료, 새 spawn으로 교체)
  - 모든 transition을 한 PPO 업데이트에 사용 (independent learners + shared policy)
"""
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neural.world import World, PPO_OBS_DIM, PPO_ACT_DIM, PPO_TARGET_POPULATION
    from neural.torch_brain import TorchBrain
else:
    from .world import World, PPO_OBS_DIM, PPO_ACT_DIM, PPO_TARGET_POPULATION
    from .torch_brain import TorchBrain


# 하이퍼파라미터
ROLLOUT_TICKS = 128        # 한 번 업데이트당 월드 틱 수
HIDDEN = 64
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 256
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


def make_world(seed: int, policy: TorchBrain, for_training: bool = False,
                curriculum: bool = False, dynamic_env: bool = False,
                multi_resource: bool = False) -> World:
    w = World(seed, curriculum=curriculum, dynamic_env=dynamic_env,
              multi_resource=multi_resource)
    w.shared_policy = policy
    w.disable_reproduction = True
    if for_training:
        w.ppo_buffer_max = None
    return w


def collect_rollout(world: World, ticks: int) -> List[dict]:
    """world.step()을 ticks회 실행, ppo_buffer 회수 후 비움."""
    world._ppo_buffer = []
    for _ in range(ticks):
        if world.population() == 0:
            break
        world.step()
    transitions = world._ppo_buffer
    world._ppo_buffer = []
    return transitions


def compute_per_creature_gae(transitions: List[dict],
                              gamma: float = GAMMA,
                              lam: float = GAE_LAMBDA
                              ) -> Tuple[np.ndarray, ...]:
    """각 creature_id별 trajectory를 만들어 GAE 계산, 마지막에 평탄화."""
    # cid -> list of dict(obs, action, log_prob, value, reward, done)
    trajs: Dict[int, List[dict]] = {}
    for batch in transitions:
        if batch['rewards'] is None:
            continue
        for i, cid in enumerate(batch['creature_ids']):
            trajs.setdefault(cid, []).append({
                'obs': batch['obs'][i],
                'action': batch['actions'][i],
                'log_prob': batch['log_probs'][i],
                'value': batch['values'][i],
                'reward': batch['rewards'][i],
                'done': batch['dones'][i],
            })

    all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []
    for cid, traj in trajs.items():
        if not traj:
            continue
        rewards = np.array([t['reward'] for t in traj], dtype=np.float32)
        values = np.array([t['value'] for t in traj], dtype=np.float32)
        dones = np.array([t['done'] for t in traj], dtype=np.float32)
        T = len(traj)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        for i, t in enumerate(traj):
            all_obs.append(t['obs'])
            all_act.append(t['action'])
            all_logp.append(t['log_prob'])
            all_adv.append(advantages[i])
            all_ret.append(returns[i])

    if not all_obs:
        return (np.empty(0),) * 5
    return (np.stack(all_obs).astype(np.float32),
            np.stack(all_act).astype(np.float32),
            np.array(all_logp, dtype=np.float32),
            np.array(all_adv, dtype=np.float32),
            np.array(all_ret, dtype=np.float32))


def ppo_update(policy: TorchBrain, optimizer: optim.Optimizer,
               obs: np.ndarray, actions: np.ndarray,
               old_log_probs: np.ndarray, advantages: np.ndarray,
               returns: np.ndarray) -> dict:
    """표준 PPO 업데이트 — clipped surrogate + value loss + entropy bonus."""
    device = next(policy.parameters()).device
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    olp_t = torch.from_numpy(old_log_probs).to(device)
    adv_t = torch.from_numpy(advantages).to(device)
    ret_t = torch.from_numpy(returns).to(device)
    # advantage 정규화
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    losses = {'policy': [], 'value': [], 'entropy': []}
    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            mb = idx[start:start + BATCH_SIZE]
            new_logp, ent, value = policy.evaluate(obs_t[mb], act_t[mb])
            ratio = (new_logp - olp_t[mb]).exp()
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (value - ret_t[mb]).pow(2).mean()
            ent_bonus = ent.mean()
            loss = (policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * ent_bonus)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses['policy'].append(policy_loss.item())
            losses['value'].append(value_loss.item())
            losses['entropy'].append(ent_bonus.item())
    return {k: float(np.mean(v)) for k, v in losses.items()}


def policy_path() -> Path:
    p = Path(__file__).resolve().parent.parent / 'recordings' / 'neural' / 'ppo_world_policy.pt'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save_checkpoint(policy, optimizer, total_steps: int, iteration: int,
                      learning_curve: list, args) -> None:
    """전체 학습 상태를 저장 — resume 가능. policy_path()에 덮어씀."""
    out = policy_path()
    torch.save({
        'state_dict': policy.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'total_steps': total_steps,
        'iteration': iteration,
        'learning_curve': learning_curve,
        'config': {'arch': getattr(args, 'arch', 'mlp'),
                   'hidden': args.hidden,
                   'obs_dim': PPO_OBS_DIM, 'act_dim': PPO_ACT_DIM},
    }, out)


def _save_learning_curve_plot(learning_curve: list) -> None:
    """학습 곡선 — mean_reward, entropy, value_loss, policy_loss 4-panel PNG."""
    if not learning_curve:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = [e['total_steps'] for e in learning_curve]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('PPO Training Curve', fontsize=13, fontweight='bold')
    for ax, key, color, label in [
        (axes[0][0], 'mean_reward', '#3a7', 'Mean Reward'),
        (axes[0][1], 'entropy', '#d62', 'Policy Entropy'),
        (axes[1][0], 'value_loss', '#69c', 'Value Loss'),
        (axes[1][1], 'policy_loss', '#a37', 'Policy Loss'),
    ]:
        ys = [e[key] for e in learning_curve]
        ax.plot(steps, ys, color=color, linewidth=1.4)
        ax.fill_between(steps, ys, alpha=0.18, color=color)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('total transitions')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = policy_path().parent / 'ppo_training_curve.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[PPO-WORLD] device={device}  steps={args.steps}  '
          f'target_pop={PPO_TARGET_POPULATION}  arch={args.arch}')

    policy = TorchBrain(PPO_OBS_DIM, args.hidden, PPO_ACT_DIM,
                        arch=args.arch).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    world = make_world(args.seed, policy, for_training=True,
                        curriculum=args.curriculum,
                        dynamic_env=args.dynamic_env,
                        multi_resource=args.multi_resource)

    total_steps = 0
    iteration = 0
    learning_curve: List[dict] = []

    if args.resume:
        ckpt_path = policy_path()
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            policy.load_state_dict(ckpt['state_dict'])
            if 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            total_steps = int(ckpt.get('total_steps', 0))
            iteration = int(ckpt.get('iteration', 0))
            learning_curve = ckpt.get('learning_curve', [])
            print(f'[RESUME] from {ckpt_path}  '
                  f'iter={iteration} steps={total_steps}')
        else:
            print(f'[RESUME] no checkpoint at {ckpt_path}, starting fresh')

    t0 = time.perf_counter()
    while total_steps < args.steps:
        transitions = collect_rollout(world, ROLLOUT_TICKS)
        obs, act, logp, adv, ret = compute_per_creature_gae(transitions)
        if obs.shape[0] == 0:
            print('no transitions, resetting world')
            world = make_world(args.seed + iteration, policy)
            continue
        loss = ppo_update(policy, optimizer, obs, act, logp, adv, ret)
        total_steps += obs.shape[0]
        iteration += 1

        mean_reward = float(np.mean([
            np.mean(b['rewards']) for b in transitions
            if b['rewards'] is not None and len(b['rewards']) > 0
        ])) if transitions else 0.0
        dt = time.perf_counter() - t0
        learning_curve.append({
            'iteration': iteration, 'total_steps': total_steps,
            'mean_reward': mean_reward, 'kills': world.kills,
            'policy_loss': loss['policy'], 'value_loss': loss['value'],
            'entropy': loss['entropy'], 'wallclock': dt,
        })

        if iteration % 5 == 0 or total_steps >= args.steps:
            print(f'[PPO-WORLD] iter={iteration:4d} '
                  f'transitions={total_steps:7d}  '
                  f'mean_r={mean_reward:+5.2f}  pop={world.population():3d}  '
                  f'kills={world.kills}  '
                  f'pl={loss["policy"]:+.3f} vl={loss["value"]:.2f} '
                  f'ent={loss["entropy"]:.2f}  ({dt:.0f}s)')

        # 주기적 체크포인트 (resume용)
        if iteration % 25 == 0 or total_steps >= args.steps:
            _save_checkpoint(policy, optimizer, total_steps, iteration,
                              learning_curve, args)
            if args.live_plot:
                _save_learning_curve_plot(learning_curve)

        # 가끔 월드 리셋해서 다양한 환경 보게
        if iteration % 20 == 0:
            world = make_world(args.seed + iteration * 7, policy,
                                for_training=True,
                                curriculum=args.curriculum,
                                dynamic_env=args.dynamic_env,
                                multi_resource=args.multi_resource)

    _save_checkpoint(policy, optimizer, total_steps, iteration,
                      learning_curve, args)
    if args.live_plot:
        _save_learning_curve_plot(learning_curve)
    print(f'[PPO-WORLD] saved → {policy_path()}')


def load_policy(path: Path, device) -> TorchBrain:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    policy = TorchBrain(cfg['obs_dim'], cfg['hidden'], cfg['act_dim'],
                        arch='mlp').to(device)
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()
    return policy


def evaluate(args):
    """학습된 PPO 정책을 N개 시드의 World에서 평가."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = policy_path()
    if not path.exists():
        print(f'no policy at {path}'); return
    policy = load_policy(path, device)
    print(f'[EVAL] loaded {path}')

    results = []
    test_seeds = list(range(10_000, 10_000 + args.eval_seeds))
    for s in test_seeds:
        world = make_world(s, policy)
        for _ in range(args.ticks):
            if world.population() == 0:
                break
            world.step()
        if world.creatures:
            avg_eaten = sum(c.lifetime_eaten for c in world.creatures) / world.population()
        else:
            avg_eaten = 0.0
        results.append({
            'seed': s, 'pop': world.population(),
            'avg_eaten': avg_eaten, 'kills': world.kills,
        })
        print(f'  seed={s} pop={world.population()} '
              f'avg_eaten={avg_eaten:.2f} kills={world.kills}')

    if results:
        pops = [r['pop'] for r in results]
        eats = [r['avg_eaten'] for r in results]
        kls = [r['kills'] for r in results]
        print(f'\n[EVAL] pop={np.mean(pops):.1f}±{np.std(pops):.1f}  '
              f'avg_eaten={np.mean(eats):.2f}±{np.std(eats):.2f}  '
              f'kills={np.mean(kls):.1f}±{np.std(kls):.1f}')


def compare(args):
    """진화 vs PPO를 같은 World에서 비교."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = policy_path()
    if not path.exists():
        print(f'먼저 --steps로 학습 필요')
        return
    policy = load_policy(path, device)

    test_seeds = list(range(10_000, 10_000 + args.n_seeds))
    print(f'\nApples-to-apples: 진화 baseline vs PPO  '
          f'(시드 {args.n_seeds}개 × {args.ticks}틱)\n')

    ppo_results = []
    for s in test_seeds:
        w = make_world(s, policy)
        for _ in range(args.ticks):
            if w.population() == 0: break
            w.step()
        eaten = sum(c.lifetime_eaten for c in w.creatures) / max(1, w.population())
        ppo_results.append({'pop': w.population(), 'avg_eaten': eaten,
                            'kills': w.kills})

    evo_results = []
    for s in test_seeds:
        w = World(s)
        for _ in range(args.ticks):
            if w.population() == 0: break
            w.step()
        eaten = sum(c.lifetime_eaten for c in w.creatures) / max(1, w.population())
        evo_results.append({'pop': w.population(), 'avg_eaten': eaten,
                            'kills': w.kills})

    def stats(results, key):
        vs = [r[key] for r in results]
        return np.mean(vs), np.std(vs)

    print(f'{"metric":15} {"evolution":>15} {"PPO":>15}  '
          f'{"d":>6} {"t":>7} {"p":>8} {"95% CI":>20}')
    print('-' * 90)
    summary = {}
    try:
        from scipy import stats as scistats
        have_scipy = True
    except ImportError:
        have_scipy = False
    for m in ['pop', 'avg_eaten', 'kills']:
        em, es = stats(evo_results, m)
        pm, ps = stats(ppo_results, m)
        pooled = np.sqrt((es ** 2 + ps ** 2) / 2) or 1e-9
        d = (pm - em) / pooled
        # 통계 검정 — Welch t-test + bootstrap 95% CI of mean diff
        evo_vs = np.array([r[m] for r in evo_results])
        ppo_vs = np.array([r[m] for r in ppo_results])
        if have_scipy and len(evo_vs) >= 2 and len(ppo_vs) >= 2:
            tres = scistats.ttest_ind(ppo_vs, evo_vs, equal_var=False)
            t_stat, p_val = float(tres.statistic), float(tres.pvalue)
        else:
            t_stat, p_val = 0.0, 1.0
        # bootstrap CI of (PPO - Evolution) mean
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(2000):
            be = rng.choice(evo_vs, size=len(evo_vs), replace=True)
            bp = rng.choice(ppo_vs, size=len(ppo_vs), replace=True)
            diffs.append(bp.mean() - be.mean())
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        summary[m] = (em, es, pm, ps, d, t_stat, p_val, ci_lo, ci_hi)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
              ('*' if p_val < 0.05 else ''))
        print(f'{m:15} {em:8.2f}±{es:5.2f} {pm:8.2f}±{ps:5.2f}  '
              f'{d:+6.2f} {t_stat:+7.2f} {p_val:8.4f}{sig:>3} '
              f'[{ci_lo:+7.2f},{ci_hi:+7.2f}]')

    if args.plot:
        plot_compare(evo_results, ppo_results, summary, args.n_seeds, args.ticks)

    if args.video:
        from neural.compare_video import record_compare_video
        record_compare_video(test_seeds[0], args.ticks, policy,
                              headless=True, fps=60)


def plot_compare(evo_results, ppo_results, summary, n_seeds, ticks):
    """막대 + 시드별 산점도 + Cohen's d 표시."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['pop', 'avg_eaten', 'kills']
    metric_labels = ['Population', 'Avg Food Eaten', 'Predator Kills']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Apples-to-apples: Evolution vs PPO  '
                  f'(seeds={n_seeds}, ticks={ticks})',
                  fontsize=13, fontweight='bold')

    rng = np.random.default_rng(0)
    for ax, m, label in zip(axes, metrics, metric_labels):
        em, es, pm, ps, d, *_ = summary[m]   # F 추가로 9개 튜플 — 앞 5개만 사용
        ax.bar([0, 1], [em, pm], yerr=[es, ps],
                color=['#7fa9d6', '#e08a5b'], capsize=8,
                edgecolor='black', linewidth=1, alpha=0.85)
        # 시드별 점
        evo_vals = [r[m] for r in evo_results]
        ppo_vals = [r[m] for r in ppo_results]
        ax.scatter(rng.normal(0, 0.06, len(evo_vals)), evo_vals,
                    color='#345b87', s=22, alpha=0.7, zorder=3)
        ax.scatter(rng.normal(1, 0.06, len(ppo_vals)), ppo_vals,
                    color='#a05529', s=22, alpha=0.7, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Evolution', 'PPO'])
        ax.set_title(f'{label}\nCohen\'s d = {d:+.2f}',
                      fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        # 효과 크기 라벨
        magnitude = ('large' if abs(d) > 0.8 else
                     'medium' if abs(d) > 0.5 else
                     'small' if abs(d) > 0.2 else 'negligible')
        ax.text(0.5, 0.95, f'effect: {magnitude}',
                 transform=ax.transAxes, ha='center', va='top',
                 fontsize=9, color='#555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                           edgecolor='#bbb'))

    plt.tight_layout()
    out = policy_path().parent / f'compare_{n_seeds}seeds_{ticks}ticks.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[PLOT] saved → {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden', type=int, default=HIDDEN)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--eval-seeds', type=int, default=10)
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--ticks', type=int, default=1500)
    parser.add_argument('--plot', action='store_true',
                        help='--compare 결과를 PNG 차트로 저장')
    parser.add_argument('--video', action='store_true',
                        help='--compare 후 첫 시드를 나란히 mp4로 녹화')
    parser.add_argument('--arch', choices=['mlp', 'gru'], default='mlp',
                        help='C: GRU 정책 (recurrent)')
    parser.add_argument('--resume', action='store_true',
                        help='B: 기존 체크포인트에서 이어서 학습')
    parser.add_argument('--live-plot', action='store_true',
                        help='D: 25 iter마다 학습 곡선 PNG 저장')
    parser.add_argument('--curriculum', action='store_true',
                        help='H: 점진적 난이도 (시작은 포식자/벽 없음)')
    parser.add_argument('--dynamic-env', action='store_true',
                        help='J: 동적 환경 (음식 계절성, 포식자 웨이브)')
    parser.add_argument('--multi-resource', action='store_true',
                        help='K: 추가 음식 종류 (단백질/탄수화물)')
    args = parser.parse_args()

    if args.compare:
        compare(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
