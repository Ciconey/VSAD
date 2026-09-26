import os
import sys
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", "egl")

sys.path.append("./tdmpc2/tdmpc2")
import warnings

warnings.filterwarnings("ignore")

import time

import hydra
import imageio
import numpy as np
import torch
from WA.attacker import train_universal_poison
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from WA.tdmpc2_attack import TDMPC2
from termcolor import colored

torch.backends.cudnn.benchmark = True


def _unwrap_reset(reset_out):
    """兼容 Gym / Gymnasium 的 reset 返回格式"""
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _unwrap_step(step_out):
    """兼容 Gym / Gymnasium 的 step 返回格式"""
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    os.makedirs(cfg.work_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(cfg.work_dir, f"eval_results_{timestamp}.txt")

    with open(txt_path, "w", encoding="utf-8") as log_f:
        def log(msg, color=None, attrs=None):
            if color is not None:
                print(colored(msg, color, attrs=attrs or []))
            else:
                print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        log(f"Task: {cfg.task}", "blue", ["bold"])
        log(f"Model size: {cfg.get('model_size', 'default')}", "blue", ["bold"])
        log(f"Checkpoint: {cfg.checkpoint}", "blue", ["bold"])
        log(f"Results will be saved to: {txt_path}")

        if not cfg.multitask and ("mt80" in cfg.checkpoint or "mt30" in cfg.checkpoint):
            log("Warning: single-task evaluation of multi-task models is not currently supported.", "red", ["bold"])
            log("To evaluate a multi-task model, use task=mt80 or task=mt30.", "red", ["bold"])

        env = make_env(cfg)

        agent = TDMPC2(cfg)
        assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
        agent.load(cfg.checkpoint)

        if cfg.multitask:
            log(f"Evaluating agent on {len(cfg.tasks)} tasks:", "yellow", ["bold"])
        else:
            log(f"Evaluating agent on {cfg.task}:", "yellow", ["bold"])

        if cfg.save_video:
            video_dir = os.path.join(cfg.work_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

        scores = []
        tasks = cfg.tasks if cfg.multitask else [cfg.task]

        for task_idx, task in enumerate(tasks):
            if not task.startswith("mw-"):
                continue
            if not cfg.multitask:
                task_idx = None

            log(f"\n[{task}] Training rollout-level poison...", "red", ["bold"])

            # 这里假设你已经把 attacker.py 改成返回 poison_fn
            # poison_fn = train_universal_poison(
            #     agent=agent,
            #     env=env,
            #     task_idx=task_idx,
            #     cfg=cfg,
            #     steps=1000,
            #     buffer_size=256,
            #     batch_size=32,
            #     eps=0.15,

            #     # near-success state collection
            #     reward_threshold=1.0,
            #     warmup_steps=5000,
            #     top_percentile=80, #80

            #     # ranking objective
            #     margin=1.0,
            #     alpha_reward=0,
            #     gamma_delta=1e-4,
            #     eta_gate=3e-2,
            # )
            attack_cfg = cfg.attack

            poison_fn = train_universal_poison(
                agent=agent,
                env=env,
                task_idx=task_idx,
                cfg=cfg,
                steps=attack_cfg["steps"],
                buffer_size=attack_cfg["buffer_size"],
                batch_size=attack_cfg["batch_size"],
                eps=attack_cfg["eps"],
                reward_threshold=attack_cfg.get("reward_threshold", None),
                warmup_steps=attack_cfg["warmup_steps"],
                top_percentile=attack_cfg["top_percentile"],
                margin=attack_cfg["margin"],
                alpha_reward=attack_cfg["alpha_reward"],
                gamma_delta=attack_cfg["gamma_delta"],
                eta_gate=attack_cfg["eta_gate"],
                lr=attack_cfg.get("lr", 1e-3),
            )

            ep_rewards, ep_successes = [], []
            ep_safety_metrics = {}
            ep_action_shifts = []

            for i in range(cfg.eval_episodes):
                obs = _unwrap_reset(env.reset(task_idx=task_idx) if task_idx is not None else env.reset())
                done = False
                ep_reward = 0.0
                t = 0
                current_ep_safety = {}
                current_ep_action_shift = []

                if cfg.save_video:
                    frames = [env.render()]

                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)

                    prev_mean_backup = agent._prev_mean.clone()

                    # clean action
                    clean_action = agent.act(
                        obs_tensor,
                        t0=(t == 0),
                        eval_mode=True,
                        task=task_idx,
                        poison_fn=None,
                        root_delta=None,
                    )
                    
                    agent._prev_mean.copy_(prev_mean_backup)

                    # poisoned action
                    poison_action = agent.act(
                        obs_tensor,
                        t0=(t == 0),
                        eval_mode=True,
                        task=task_idx,
                        poison_fn=poison_fn,
                        root_delta=None,   # 先不加 root_delta；若你单独训练了 root_delta 再传
                    )

                    # 记录动作偏移，确认 planner 真被改到了
                    action_shift = (poison_action - clean_action).abs().mean().item()
                    current_ep_action_shift.append(action_shift)

                    # 用 poisoned action 执行环境
                    env_action = poison_action.cpu()
					# if not torch.is_tensor(env_action):
					# 	env_action = torch.as_tensor(env_action, dtype=torch.float32)
					# env_action = env_action.cpu()
                    obs, reward, done, info = _unwrap_step(env.step(env_action))

					# env_action = poison_action
					# if not torch.is_tensor(env_action):
					# 	env_action = torch.as_tensor(env_action, dtype=torch.float32)
					# env_action = env_action.cpu()

					# obs, reward, done, info = _unwrap_step(env.step(env_action))


                    ep_reward += reward
                    t += 1

                    for k, v in info.items():
                        if k.startswith("safety/"):
                            current_ep_safety[k] = current_ep_safety.get(k, 0.0) + float(v)

                    if cfg.save_video:
                        frames.append(env.render())

                ep_rewards.append(ep_reward)
                ep_successes.append(float(info.get("success", 0.0)))
                ep_action_shifts.append(np.mean(current_ep_action_shift) if len(current_ep_action_shift) > 0 else 0.0)

                jitter_val = current_ep_safety.get("safety/jitter", 0.0)
                impact_val = current_ep_safety.get("safety/impact", 0.0)
                shift_val = ep_action_shifts[-1]

                log(
                    f"  [Episode {i}] Reward: {ep_reward:.1f} | "
                    f"Success: {info.get('success', 0.0)} | "
                    f"ActionShift: {shift_val:.4f} | "
                    f"Jitter: {jitter_val:.2f} | "
                    f"Impact: {impact_val:.2f}"
                )

                for k, v in current_ep_safety.items():
                    if k not in ep_safety_metrics:
                        ep_safety_metrics[k] = []
                    ep_safety_metrics[k].append(v)

                if cfg.save_video:
                    imageio.mimsave(
                        os.path.join(video_dir, f"{task}-{i}.mp4"),
                        frames,
                        fps=15,
                    )

            mean_reward = float(np.mean(ep_rewards))
            mean_success = float(np.mean(ep_successes))
            mean_action_shift = float(np.mean(ep_action_shifts))

            if cfg.multitask:
                scores.append(mean_success * 100 if task.startswith("mw-") else mean_reward / 10)

            log(
                f"  {task:<22}\tR: {mean_reward:.01f}\tS: {mean_success:.02f}",
                "yellow"
            )

            log("\n=== Safety / Attack Diagnostics ===")
            log(f"action_shift_mean: {mean_action_shift:.6f}")
            for k, v_list in ep_safety_metrics.items():
                log(f"{k}: {np.mean(v_list):.4f} +/- {np.std(v_list):.4f}")
            log("===================================\n")

        if cfg.multitask:
            log(f"Normalized score: {np.mean(scores):.02f}", "yellow", ["bold"])

        log(f"Evaluation finished. Results saved in: {txt_path}")


if __name__ == "__main__":
    evaluate()
