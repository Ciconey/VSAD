# BA/blackbox_attack_nes.py

import atexit
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from termcolor import colored

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if os.path.isdir(PROJECT_DIR) and PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from BB.tdmpc2_black import (
    TDMPC2 as DefendedTDMPC2,
)
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2 as OriginalTDMPC2

warnings.filterwarnings("ignore")

# ============================================================
# Logging
# ============================================================

_ANSI_ESCAPE_PATTERN = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

def strip_ansi(text):
    return _ANSI_ESCAPE_PATTERN.sub("", str(text))

class TeeStream:
    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream

    def write(self, message):
        if message is None:
            return 0

        message = str(message)

        self.terminal_stream.write(message)
        self.terminal_stream.flush()

        self.file_stream.write(strip_ansi(message))
        self.file_stream.flush()

        return len(message)

    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        return self.terminal_stream.isatty()

    def fileno(self):
        return self.terminal_stream.fileno()

    @property
    def encoding(self):
        return getattr(
            self.terminal_stream,
            "encoding",
            "utf-8",
        )

class StreamingTextLogger:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.file_stream = None
        self.original_stdout = None
        self.original_stderr = None
        self.started = False

    def start(self):
        if self.started:
            return

        directory = os.path.dirname(self.txt_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.file_stream = open(
            self.txt_path,
            "w",
            encoding="utf-8",
            buffering=1,
        )

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        sys.stdout = TeeStream(
            self.original_stdout,
            self.file_stream,
        )
        sys.stderr = TeeStream(
            self.original_stderr,
            self.file_stream,
        )

        self.started = True

    def close(self):
        if not self.started:
            return

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            if self.file_stream is not None:
                self.file_stream.flush()
                self.file_stream.close()

            self.started = False

# ============================================================
# Basic utilities
# ============================================================

def obs_to_tensor(obs, device="cuda"):
    if isinstance(obs, torch.Tensor):
        return obs.detach().float().view(-1).to(device)

    return torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    ).view(-1)

def action_to_tensor(action, device="cuda"):
    if isinstance(action, torch.Tensor):
        return action.detach().float().view(-1).to(device)

    return torch.as_tensor(
        action,
        dtype=torch.float32,
        device=device,
    ).view(-1)

def unwrap_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out

def unwrap_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
        done = bool(done)

    return obs, reward, done, info

def safe_env_reset(env, task_idx=None):
    try:
        if task_idx is None:
            result = env.reset()
        else:
            result = env.reset(task_idx=task_idx)
    except TypeError:
        result = env.reset()

    return unwrap_reset(result)

def safe_env_step(env, action):
    return unwrap_step(env.step(action))

def json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value

def summarize_metric_lists(metric_dictionary):
    summary = {}

    for key, values in metric_dictionary.items():
        if not values:
            continue

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "values": [float(x) for x in array],
        }

    return summary

def set_optional_cfg(cfg, values):
    if OmegaConf.is_config(cfg):
        with open_dict(cfg):
            for key, value in values.items():
                setattr(cfg, key, value)
        return

    if isinstance(cfg, dict):
        cfg.update(values)
        return

    for key, value in values.items():
        setattr(cfg, key, value)

def reset_target_planner(target_agent):
    if hasattr(target_agent, "_prev_mean"):
        with torch.no_grad():
            target_agent._prev_mean.zero_()

def reset_target_defense_stats(target_agent):
    if hasattr(target_agent, "reset_defense_stats"):
        target_agent.reset_defense_stats()

def read_target_defense_stats(target_agent):
    if hasattr(target_agent, "get_defense_stats"):
        return target_agent.get_defense_stats()
    return {}

def get_task_specification(cfg, task_indices=None):
    if cfg.multitask:
        if task_indices is None:
            task_indices = list(range(len(cfg.tasks)))

        return [
            (
                cfg.tasks[index],
                index,
                index,
            )
            for index in task_indices
        ]

    return [
        (
            cfg.task,
            None,
            None,
        )
    ]

@torch.no_grad()
def target_act(
    target_agent,
    obs,
    t0=False,
    task=None,
    eval_mode=True,
    device="cuda",
):
    obs_tensor = obs_to_tensor(
        obs,
        device=device,
    )

    action = target_agent.act(
        obs_tensor,
        t0=t0,
        eval_mode=eval_mode,
        task=task,
    )

    return action_to_tensor(
        action,
        device=device,
    )

def capture_planner_state(target_agent):
    state = {
        "cpu_rng": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda_rng"] = None

    if hasattr(target_agent, "_prev_mean"):
        state["prev_mean"] = (
            target_agent._prev_mean.detach().clone()
        )
    else:
        state["prev_mean"] = None

    return state

def restore_planner_state(target_agent, state):
    if state["prev_mean"] is not None:
        with torch.no_grad():
            target_agent._prev_mean.copy_(
                state["prev_mean"]
            )

    torch.set_rng_state(state["cpu_rng"])

    if (
        state["cuda_rng"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["cuda_rng"]
        )

def build_blackbox_bad_action_templates(
    clean_action,
    time_step=0,
):
    """
    Construct black-box, single-action counterparts of the white-box
    near-success action templates.

    White-box templates are trajectory-level [H, B, A] templates:
        1. hold / slow-down:
           A_hold[1:] = 0.15 * A_good[1:]

        2. mild jitter:
           A_jitter = A_good + 0.12 * sign * direction

        3. approach then retreat:
           A_retreat[-1] = -0.25 * A_good[0]

        4. stop:
           A_stop[1:] = 0

    A black-box attack only observes one action at the current control
    step. Hence, each trajectory-level template is projected into a
    current-action target, using clean_action as the available proxy for
    the white-box policy-prior action.

    Args:
        clean_action:
            Tensor with shape [action_dim].

        time_step:
            Environment/control time step. It is used to alternate the
            jitter direction consistently with the white-box temporal
            jitter pattern.

    Returns:
        dict[str, torch.Tensor]:
            Four action-space templates:
              - "hold"
              - "jitter"
              - "retreat"
              - "stop"
    """
    clean_action = clean_action.detach()

    # Same treatment as white-box code:
    # direction = sign(A_good), replacing zero entries by +1.
    direction = torch.sign(clean_action)
    direction = torch.where(
        direction.abs() < 1e-6,
        torch.ones_like(direction),
        direction,
    )

    # White-box code applies:
    # sign[1::2] = -1
    # During receding-horizon black-box control, use global time parity
    # as the corresponding temporal phase.
    jitter_phase = (
        1.0
        if int(time_step) % 2 == 0
        else -1.0
    )

    templates = {
        # White-box:
        # A_hold[1:] = 0.15 * A_good[1:]
        "hold": (
            0.15 * clean_action
        ).clamp(-1.0, 1.0),

        # White-box:
        # A_jitter = A_good + 0.12 * sign * direction
        "jitter": (
            clean_action
            + 0.12 * jitter_phase * direction
        ).clamp(-1.0, 1.0),

        # White-box:
        # A_retreat[-1] = -0.25 * A_good[0]
        "retreat": (
            -0.25 * clean_action
        ).clamp(-1.0, 1.0),

        # White-box:
        # A_stop[1:] = 0.0
        "stop": torch.zeros_like(
            clean_action
        ),
    }

    return templates

def select_blackbox_bad_action_template(
    clean_action,
    time_step,
    template_mode="retreat",
    cycle_period=1,
):
    """
    Select a black-box bad-action target from the white-box-consistent
    template family.

    Args:
        clean_action:
            Current clean planner action [action_dim].

        time_step:
            Current environment step.

        template_mode:
            One of:
                - "hold"
                - "jitter"
                - "retreat"
                - "stop"
                - "cycle"

            "cycle" periodically uses:
                hold -> jitter -> retreat -> stop

        cycle_period:
            Number of control steps for each template when
            template_mode == "cycle".

    Returns:
        bad_action_template:
            Tensor [action_dim].

        selected_name:
            Name of the selected template.
    """
    templates = build_blackbox_bad_action_templates(
        clean_action=clean_action,
        time_step=time_step,
    )

    mode = str(template_mode).lower()

    aliases = {
        "slow": "hold",
        "slowdown": "hold",
        "slow_down": "hold",
        "noise": "jitter",
        "jittering": "jitter",
        "reverse": "retreat",
        "backoff": "retreat",
        "zero": "stop",
    }

    mode = aliases.get(
        mode,
        mode,
    )

    if mode == "cycle":
        sequence = [
            "hold",
            "jitter",
            "retreat",
            "stop",
        ]

        cycle_period = max(
            int(cycle_period),
            1,
        )

        selected_name = sequence[
            (
                int(time_step)
                // cycle_period
            )
            % len(sequence)
        ]

    elif mode in templates:
        selected_name = mode

    else:
        valid = [
            "hold",
            "jitter",
            "retreat",
            "stop",
            "cycle",
        ]

        raise ValueError(
            "Unknown attack_targeted_template="
            f"{template_mode!r}. Valid choices: {valid}."
        )

    return (
        templates[selected_name],
        selected_name,
    )

# ============================================================
# NES black-box attack
# ============================================================

def query_target_loss(
    target_agent,
    obs_perturbed,
    clean_action,
    task=None,
    attack_objective="action_divergence",
    bad_action_template=None,
    device="cuda",
):
    """
    Query the target planner once without permanently changing
    its MPPI warm-start state or random-number state.
    """
    planner_state = capture_planner_state(
        target_agent
    )

    with torch.no_grad():
        attacked_action = target_act(
            target_agent=target_agent,
            obs=obs_perturbed,
            t0=False,
            task=task,
            eval_mode=True,
            device=device,
        )

    restore_planner_state(
        target_agent,
        planner_state,
    )

    if attack_objective == "action_divergence":
        loss = -torch.norm(
            attacked_action - clean_action,
            p=2,
        )

    elif attack_objective == "targeted_bad":
        if bad_action_template is None:
            bad_action_template = torch.zeros_like(
                clean_action
            )

        loss = torch.norm(
            attacked_action - bad_action_template,
            p=2,
        )

    else:
        raise ValueError(
            "Unknown attack objective: "
            f"{attack_objective}"
        )

    return float(loss.item()), attacked_action

def optimize_obs_delta_nes(
    target_agent,
    obs_clean,
    clean_action,
    task=None,
    eps=0.05,
    alpha=0.01,
    steps=10,
    num_samples=16,
    sigma=0.01,
    attack_objective="action_divergence",
    bad_action_template=None,
    random_start=True,
    verbose=False,
    device="cuda",
):
    """
    Zeroth-order NES-PGD attack in observation space.

    The optimization objective is queried through the target agent's
    action output. The final perturbation satisfies:

        ||delta||_inf <= eps
    """
    obs_clean = obs_clean.view(-1)
    obs_dim = obs_clean.numel()

    if random_start:
        delta = torch.empty_like(
            obs_clean
        ).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(obs_clean)

    last_info = {}

    for step in range(steps):
        half_samples = max(
            1,
            int(num_samples) // 2,
        )

        base_noise = torch.randn(
            half_samples,
            obs_dim,
            device=device,
            dtype=obs_clean.dtype,
        )

        noise_vectors = torch.cat(
            [
                base_noise,
                -base_noise,
            ],
            dim=0,
        )

        losses = []

        for direction in noise_vectors:
            candidate = (
                obs_clean
                + delta
                + sigma * direction
            )

            candidate = torch.clamp(
                candidate,
                obs_clean - eps,
                obs_clean + eps,
            )

            loss_value, _ = query_target_loss(
                target_agent=target_agent,
                obs_perturbed=candidate,
                clean_action=clean_action,
                task=task,
                attack_objective=attack_objective,
                bad_action_template=bad_action_template,
                device=device,
            )

            losses.append(loss_value)

        losses_tensor = torch.as_tensor(
            losses,
            dtype=obs_clean.dtype,
            device=device,
        )

        normalized_losses = (
            losses_tensor - losses_tensor.mean()
        ) / (
            losses_tensor.std() + 1e-8
        )

        grad_estimate = torch.sum(
            normalized_losses.unsqueeze(1)
            * noise_vectors,
            dim=0,
        ) / (
            noise_vectors.shape[0] * sigma
        )

        delta = (
            delta
            - alpha * torch.sign(grad_estimate)
        )

        delta = torch.clamp(
            delta,
            -eps,
            eps,
        )

        current_loss, current_action = (
            query_target_loss(
                target_agent=target_agent,
                obs_perturbed=obs_clean + delta,
                clean_action=clean_action,
                task=task,
                attack_objective=attack_objective,
                bad_action_template=bad_action_template,
                device=device,
            )
        )

        action_shift = torch.norm(
            current_action - clean_action,
            p=2,
        ).item()

        last_info = {
            "loss": float(current_loss),
            "action_shift_l2": float(action_shift),
            "delta_linf": float(
                delta.abs().max().item()
            ),
            "delta_l2": float(
                delta.pow(2).sum().sqrt().item()
            ),
        }

        if verbose:
            print(
                colored(
                    f"    [NES Step {step:02d}] "
                    f"loss={last_info['loss']:.5f} | "
                    f"shift_l2={last_info['action_shift_l2']:.5f} | "
                    f"Linf={last_info['delta_linf']:.5f}",
                    "yellow",
                ),
                flush=True,
            )

    return (
        obs_clean + delta.detach(),
        delta.detach(),
        last_info,
    )

# ============================================================
# Clean evaluation
# ============================================================

@torch.no_grad()
def evaluate_clean_target(
    cfg,
    env,
    target_agent,
    num_episodes,
    device="cuda",
    eval_mode=True,
    task_indices=None,
    save_video=False,
    video_dir=None,
):
    print(
        colored(
            "\n========== Clean Target Evaluation ==========",
            "blue",
            attrs=["bold"],
        ),
        flush=True,
    )

    results = {}

    for (
        task_name,
        env_task_idx,
        model_task_idx,
    ) in get_task_specification(
        cfg,
        task_indices,
    ):
        if not task_name.startswith("mw-"):
            continue

        print(
            colored(
                f"Evaluating clean target on "
                f"{task_name}:",
                "yellow",
                attrs=["bold"],
            ),
            flush=True,
        )

        reset_target_defense_stats(
            target_agent
        )
        reset_target_planner(
            target_agent
        )

        episode_rewards = []
        episode_successes = []
        safety_metrics = defaultdict(list)

        for episode in range(num_episodes):
            reset_target_planner(
                target_agent
            )

            obs = safe_env_reset(
                env,
                task_idx=env_task_idx,
            )

            done = False
            time_step = 0
            episode_reward = 0.0
            last_info = {}
            current_safety = defaultdict(float)

            frames = []

            if save_video:
                frames.append(env.render())

            while not done:
                action = target_act(
                    target_agent=target_agent,
                    obs=obs,
                    t0=(time_step == 0),
                    task=model_task_idx,
                    eval_mode=eval_mode,
                    device=device,
                )

                obs, reward, done, info = (
                    safe_env_step(
                        env,
                        action.detach().cpu(),
                    )
                )

                episode_reward += float(reward)
                last_info = info
                time_step += 1

                if isinstance(info, dict):
                    for key, value in info.items():
                        if (
                            isinstance(key, str)
                            and key.startswith("safety/")
                        ):
                            current_safety[key] += float(
                                value
                            )

                if save_video:
                    frames.append(env.render())

            success = float(
                last_info.get("success", 0.0)
                if isinstance(last_info, dict)
                else 0.0
            )

            episode_rewards.append(
                episode_reward
            )
            episode_successes.append(success)

            for key, value in current_safety.items():
                safety_metrics[key].append(value)

            print(
                f"  [Clean Episode {episode}] "
                f"Reward: {episode_reward:.1f} | "
                f"Success: {success:.1f}",
                flush=True,
            )

            if save_video and video_dir is not None:
                os.makedirs(video_dir, exist_ok=True)

                imageio.mimsave(
                    os.path.join(
                        video_dir,
                        f"clean-{task_name}-{episode}.mp4",
                    ),
                    frames,
                    fps=15,
                )

        results[task_name] = {
            "reward": float(
                np.mean(episode_rewards)
            ),
            "reward_std": float(
                np.std(episode_rewards)
            ),
            "success": float(
                np.mean(episode_successes)
            ),
            "success_std": float(
                np.std(episode_successes)
            ),
            "episode_rewards": [
                float(x) for x in episode_rewards
            ],
            "episode_successes": [
                float(x) for x in episode_successes
            ],
            "safety": summarize_metric_lists(
                safety_metrics
            ),
            "defense_stats": read_target_defense_stats(
                target_agent
            ),
        }

    return results

# ============================================================
# NES attacked evaluation
# ============================================================

def evaluate_nes_attacked_target(
    cfg,
    env,
    target_agent,
    num_episodes,
    device="cuda",
    eval_mode=True,
    task_indices=None,
    attack_objective="action_divergence",
    eps=0.05,
    alpha=0.01,
    steps=10,
    num_samples=16,
    sigma=0.01,
    attack_every=1,
    vulnerable_reward_thresh=1.0,
    vulnerable_step_thresh=15,
    random_start=True,
    verbose=False,
    save_video=False,
    video_dir=None,
    use_defense=False,
    targeted_template="retreat",
    targeted_template_cycle_period=1,
):
    print(
        colored(
            "\n========== NES Black-box Attacked "
            "Target Evaluation ==========",
            "red",
            attrs=["bold"],
        ),
        flush=True,
    )

    print(
        f"Objective: {attack_objective}",
        flush=True,
    )
    print(
        f"eps={eps} | alpha={alpha} | "
        f"steps={steps} | "
        f"NES_samples={num_samples} | "
        f"sigma={sigma}",
        flush=True,
    )
    print(
        f"attack_every={attack_every}",
        flush=True,
    )
    print(
        f"vulnerable_reward_thresh="
        f"{vulnerable_reward_thresh}",
        flush=True,
    )
    print(
        f"vulnerable_step_thresh="
        f"{vulnerable_step_thresh}",
        flush=True,
    )
    print(
        f"Root-Latent MPPI Defense Enabled: "
        f"{use_defense}",
        flush=True,
    )

    if attack_objective == "targeted_bad":
        print(
            f"Targeted bad-action template: "
            f"{targeted_template}",
            flush=True,
        )
        print(
            f"Targeted template cycle period: "
            f"{targeted_template_cycle_period}",
            flush=True,
        )

    results = {}

    for (
        task_name,
        env_task_idx,
        model_task_idx,
    ) in get_task_specification(
        cfg,
        task_indices,
    ):
        if not task_name.startswith("mw-"):
            continue

        print(
            colored(
                # f"\nEvaluating NES Attack on "
                f"[{task_name}]:",
                "yellow",
                attrs=["bold"],
            ),
            flush=True,
        )

        reset_target_defense_stats(
            target_agent
        )
        reset_target_planner(
            target_agent
        )

        episode_rewards = []
        episode_successes_final = []
        episode_successes_any = []
        episode_action_shifts = []
        episode_attack_steps = []

        safety_metrics = defaultdict(list)
        attack_metrics = defaultdict(list)

        for episode in range(num_episodes):
            reset_target_planner(
                target_agent
            )

            obs = safe_env_reset(
                env,
                task_idx=env_task_idx,
            )

            done = False
            time_step = 0
            episode_reward = 0.0
            last_reward = 0.0
            success_any = 0.0
            last_info = {}

            current_safety = defaultdict(float)
            current_action_shifts = []
            attack_steps = 0

            frames = []

            if save_video:
                frames.append(env.render())

            while not done:
                obs_clean = obs_to_tensor(
                    obs,
                    device=device,
                )

                # Trigger attack only in the configured vulnerable
                is_vulnerable = (
                    last_reward
                    >= vulnerable_reward_thresh
                    or time_step
                    >= vulnerable_step_thresh
                )

                should_attack = (
                    attack_every > 0
                    and time_step % attack_every == 0
                    and is_vulnerable
                )

                # The clean action is evaluated with a fully restored
                # planner state. This makes action-shift comparison
                # meaningful.
                planner_state = capture_planner_state(
                    target_agent
                )

                clean_action = target_act(
                    target_agent=target_agent,
                    obs=obs_clean,
                    t0=(time_step == 0),
                    task=model_task_idx,
                    eval_mode=eval_mode,
                    device=device,
                )

                restore_planner_state(
                    target_agent,
                    planner_state,
                )

                if should_attack:
                    attack_steps += 1

                    # bad_action_template = None

                    # if attack_objective == "targeted_bad":
                    #     # Retreat-like target. For a generic black-box
                    #     # attack, this is the action opposite to the
                    #     # clean action.
                    #     bad_action_template = (
                    #         -0.3 * clean_action
                    #     )
                    bad_action_template = None
                    selected_template_name = None

                    if attack_objective == "targeted_bad":
                        (
                            bad_action_template,
                            selected_template_name,
                        ) = select_blackbox_bad_action_template(
                            clean_action=clean_action,
                            time_step=time_step,
                            template_mode=targeted_template,
                            cycle_period=(
                                targeted_template_cycle_period
                            ),
                        )

                        if verbose:
                            print(
                                colored(
                                    "    [Targeted Template] "
                                    f"{selected_template_name} | "
                                    "target="
                                    f"{bad_action_template.detach().cpu().tolist()}",
                                    "magenta",
                                ),
                                flush=True,
                            )

                    obs_adv, delta, attack_info = (
                        optimize_obs_delta_nes(
                            target_agent=target_agent,
                            obs_clean=obs_clean,
                            clean_action=clean_action,
                            task=model_task_idx,
                            eps=eps,
                            alpha=alpha,
                            steps=steps,
                            num_samples=num_samples,
                            sigma=sigma,
                            attack_objective=attack_objective,
                            bad_action_template=(
                                bad_action_template
                            ),
                            random_start=random_start,
                            verbose=verbose,
                            device=device,
                        )
                    )

                    for key, value in attack_info.items():
                        attack_metrics[key].append(
                            float(value)
                        )

                    obs_for_target = obs_adv

                else:
                    delta = torch.zeros_like(
                        obs_clean
                    )
                    obs_for_target = obs_clean

                # Restore the state before the clean query, then
                # execute exactly one final action query.
                restore_planner_state(
                    target_agent,
                    planner_state,
                )

                action_adv = target_act(
                    target_agent=target_agent,
                    obs=obs_for_target,
                    t0=(time_step == 0),
                    task=model_task_idx,
                    eval_mode=eval_mode,
                    device=device,
                )

                action_shift_abs = (
                    action_adv - clean_action
                ).abs().mean().item()

                action_shift_l2 = torch.norm(
                    action_adv - clean_action,
                    p=2,
                ).item()

                current_action_shifts.append(
                    action_shift_abs
                )

                obs, reward, done, info = (
                    safe_env_step(
                        env,
                        action_adv.detach().cpu(),
                    )
                )

                reward = float(reward)
                episode_reward += reward
                last_reward = reward
                last_info = info
                time_step += 1

                if isinstance(info, dict):
                    success_any = max(
                        success_any,
                        float(
                            info.get(
                                "success",
                                0.0,
                            )
                        ),
                    )

                    for key, value in info.items():
                        if (
                            isinstance(key, str)
                            and key.startswith("safety/")
                        ):
                            current_safety[key] += float(
                                value
                            )

                if save_video:
                    frames.append(env.render())

            success_final = float(
                last_info.get("success", 0.0)
                if isinstance(last_info, dict)
                else 0.0
            )

            mean_action_shift = float(
                np.mean(current_action_shifts)
                if current_action_shifts
                else 0.0
            )

            episode_rewards.append(
                episode_reward
            )
            episode_successes_final.append(
                success_final
            )
            episode_successes_any.append(
                success_any
            )
            episode_action_shifts.append(
                mean_action_shift
            )
            episode_attack_steps.append(
                float(attack_steps)
            )

            for key, value in current_safety.items():
                safety_metrics[key].append(value)

            print(
                f"[Episode {episode}] "
                f"Reward: {episode_reward:.1f} | "
                f"Success: {success_final:.1f} | "
                f"SuccessAny: {success_any:.1f} | "
                f"ActionShift: "
                f"{mean_action_shift:.4f} | "
                f"AttackSteps: {attack_steps} | "
                f"Jitter: "
                f"{current_safety.get('safety/jitter', 0.0):.2f}",
                flush=True,
            )

            if save_video and video_dir is not None:
                os.makedirs(video_dir, exist_ok=True)

                imageio.mimsave(
                    os.path.join(
                        video_dir,
                        f"nes_attack-{task_name}-{episode}.mp4",
                    ),
                    frames,
                    fps=15,
                )

        attack_stats = summarize_metric_lists(
            attack_metrics
        )

        results[task_name] = {
            "reward": float(
                np.mean(episode_rewards)
            ),
            "reward_std": float(
                np.std(episode_rewards)
            ),
            "success": float(
                np.mean(episode_successes_final)
            ),
            "success_std": float(
                np.std(episode_successes_final)
            ),
            "success_any": float(
                np.mean(episode_successes_any)
            ),
            "success_any_std": float(
                np.std(episode_successes_any)
            ),
            "action_shift": float(
                np.mean(episode_action_shifts)
            ),
            "action_shift_std": float(
                np.std(episode_action_shifts)
            ),
            "attack_steps": float(
                np.mean(episode_attack_steps)
            ),
            "episode_rewards": [
                float(x) for x in episode_rewards
            ],
            "episode_successes": [
                float(x)
                for x in episode_successes_final
            ],
            "episode_successes_any": [
                float(x)
                for x in episode_successes_any
            ],
            "episode_action_shifts": [
                float(x)
                for x in episode_action_shifts
            ],
            "safety": summarize_metric_lists(
                safety_metrics
            ),
            "attack_stats": attack_stats,
            "defense_stats": read_target_defense_stats(
                target_agent
            ),
        }

        print(
            colored(
                f"\n  {task_name:<22}   "
                f"R: {results[task_name]['reward']:.2f}  "
                f"S: {results[task_name]['success']:.2f} "
                f"Shift: "
                f"{results[task_name]['action_shift']:.4f}",
                "yellow",
            ),
            flush=True,
        )

        defense_stats = results[task_name][
            "defense_stats"
        ]

        if defense_stats:
            print(
                "\n=== Defense Diagnostics ===",
                flush=True,
            )

            for key, value in defense_stats.items():
                print(
                    f"{key}: {float(value):.6f}",
                    flush=True,
                )

            print(
                "===========================\n",
                flush=True,
            )

    return results

# ============================================================
# Main
# ============================================================

@hydra.main(
    config_name="config",
    config_path=".",
    version_base=None,
)
def main(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for NES evaluation."
        )

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    device = "cuda"

    attack_episodes = int(
        cfg.get("attack_episodes", 10)
    )

    clean_episodes = int(
        cfg.get(
            "clean_episodes",
            attack_episodes,
        )
    )

    eval_mode = bool(
        cfg.get("attack_eval_mode", True)
    )

    attack_eps = float(
        cfg.get("attack_eps", 0.05)
    )

    attack_alpha = float(
        cfg.get("attack_alpha", 0.01)
    )

    nes_steps = int(
        cfg.get("attack_nes_steps", 10)
    )

    nes_samples = int(
        cfg.get("attack_nes_samples", 16)
    )

    nes_sigma = float(
        cfg.get("attack_nes_sigma", 0.01)
    )

    attack_objective = str(
        cfg.get(
            "attack_objective",
            "action_divergence",
        )
    )
    targeted_template = str(
        cfg.get(
            "attack_targeted_template",
            "retreat",
        )
    )

    targeted_template_cycle_period = int(
        cfg.get(
            "attack_targeted_template_cycle_period",
            1,
        )
    )

    attack_every = int(
        cfg.get("attack_every", 1)
    )

    vulnerable_reward_thresh = float(
        cfg.get(
            "vulnerable_reward_thresh",
            1.0,
        )
    )

    vulnerable_step_thresh = int(
        cfg.get(
            "vulnerable_step_thresh",
            15,
        )
    )

    attack_random_start = bool(
        cfg.get(
            "attack_random_start",
            True,
        )
    )

    attack_verbose = bool(
        cfg.get(
            "attack_verbose_pgd",
            False,
        )
    )

    run_clean = bool(
        cfg.get(
            "attack_run_clean",
            True,
        )
    )

    save_video = bool(
        cfg.get(
            "attack_save_video",
            False,
        )
    )

    use_defense = bool(
        cfg.get(
            "blackbox_use_defense",
            False,
        )
    )

    # ------------------------------------------------------------
    # Defense configuration
    # ------------------------------------------------------------
    defense_config = {
        "defense_robust_mppi": use_defense,

        "defense_mode": str(
            cfg.get(
                "defense_mode",
                "latent",
            )
        ),

        "defense_num_perturb": int(
            cfg.get(
                "defense_num_perturb",
                4,
            )
        ),

        "defense_latent_noise": float(
            cfg.get(
                "defense_latent_noise",
                0.02,
            )
        ),

        "defense_num_obs_perturb": int(
            cfg.get(
                "defense_num_obs_perturb",
                cfg.get(
                    "defense_num_perturb",
                    4,
                ),
            )
        ),

        "defense_obs_noise": float(
            cfg.get(
                "defense_obs_noise",
                0.005,
            )
        ),

        "defense_beta_sens": float(
            cfg.get(
                "defense_beta_sens",
                2.0,
            )
        ),

        "defense_beta_obs": float(
            cfg.get(
                "defense_beta_obs",
                cfg.get(
                    "defense_beta_sens",
                    2.0,
                ),
            )
        ),

        "defense_lambda_smooth": float(
            cfg.get(
                "defense_lambda_smooth",
                0.0,
            )
        ),

        "defense_antithetic_noise": bool(
            cfg.get(
                "defense_antithetic_noise",
                True,
            )
        ),

        "defense_start_fraction": float(
            cfg.get(
                "defense_start_fraction",
                0.0,
            )
        ),

        "defense_log_stats": bool(
            cfg.get(
                "defense_log_stats",
                True,
            )
        ),
    }

    set_optional_cfg(
        cfg,
        defense_config,
    )

    # ------------------------------------------------------------
    # TXT log
    # ------------------------------------------------------------
    configured_log_path = cfg.get(
        "attack_log_path",
        None,
    )

    if configured_log_path:
        txt_path = os.path.abspath(
            os.path.expanduser(
                str(configured_log_path)
            )
        )
    else:
        log_dir = os.path.join(
            cfg.work_dir,
            "nes_blackbox_logs",
        )

        os.makedirs(
            log_dir,
            exist_ok=True,
        )

        safe_task = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(cfg.task),
        ).strip("_") or "task"

        defense_name = (
            "defended"
            if use_defense
            else "undefended"
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        txt_path = os.path.join(
            log_dir,
            f"nes_{safe_task}_"
            f"{defense_name}_"
            f"{timestamp}.txt",
        )

    logger = StreamingTextLogger(txt_path)
    logger.start()
    atexit.register(logger.close)

    print(
        colored(
            f"Results will stream to: {txt_path}",
            "green",
            attrs=["bold"],
        ),
        flush=True,
    )

    print(
        "\n========== Configuration ==========",
        flush=True,
    )
    print(f"Task: {cfg.task}", flush=True)
    print(f"Checkpoint: {cfg.checkpoint}", flush=True)
    print(
        f"Attack objective: {attack_objective}",
        flush=True,
    )
    if attack_objective == "targeted_bad":
        print(
            f"Targeted bad-action template: "
            f"{targeted_template}",
            flush=True,
        )
        print(
            f"Targeted template cycle period: "
            f"{targeted_template_cycle_period}",
            flush=True,
        )
    print(
        f"Attack episodes: {attack_episodes}",
        flush=True,
    )
    print(
        f"Clean episodes: {clean_episodes}",
        flush=True,
    )
    print(
        f"attack_eps: {attack_eps}",
        flush=True,
    )
    print(
        f"attack_alpha: {attack_alpha}",
        flush=True,
    )
    print(
        f"NES steps: {nes_steps}",
        flush=True,
    )
    print(
        f"NES samples: {nes_samples}",
        flush=True,
    )
    print(
        f"NES sigma: {nes_sigma}",
        flush=True,
    )
    print(
        f"attack_every: {attack_every}",
        flush=True,
    )
    print(
        f"vulnerable_reward_thresh: "
        f"{vulnerable_reward_thresh}",
        flush=True,
    )
    print(
        f"vulnerable_step_thresh: "
        f"{vulnerable_step_thresh}",
        flush=True,
    )
    print(
        f"Defense enabled: {use_defense}",
        flush=True,
    )
    print(
        f"Defense mode: "
        f"{defense_config['defense_mode']}",
        flush=True,
    )
    print(
        f"Latent perturbations: "
        f"{defense_config['defense_num_perturb']}",
        flush=True,
    )
    print(
        f"Observation perturbations: "
        f"{defense_config['defense_num_obs_perturb']}",
        flush=True,
    )
    print(
        f"Latent noise: "
        f"{defense_config['defense_latent_noise']}",
        flush=True,
    )
    print(
        f"Observation noise: "
        f"{defense_config['defense_obs_noise']}",
        flush=True,
    )
    print(
        f"Beta latent: "
        f"{defense_config['defense_beta_sens']}",
        flush=True,
    )
    print(
        f"Beta observation: "
        f"{defense_config['defense_beta_obs']}",
        flush=True,
    )
    print(
        f"Lambda smooth: "
        f"{defense_config['defense_lambda_smooth']}",
        flush=True,
    )
    print(
        "===================================\n",
        flush=True,
    )

    # ------------------------------------------------------------
    # Environment and target
    # ------------------------------------------------------------
    env = make_env(cfg)

    target_class = (
        DefendedTDMPC2
        if use_defense
        else OriginalTDMPC2
    )

    target_agent = target_class(cfg)

    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg.checkpoint}"
        )

    target_agent.load(cfg.checkpoint)
    target_agent.eval()

    print(
        colored(
            "Loaded target: "
            f"{target_agent.__class__.__module__}."
            f"{target_agent.__class__.__name__}",
            "green",
        ),
        flush=True,
    )

    if cfg.multitask:
        task_index = int(
            cfg.get(
                "attack_task_idx",
                -1,
            )
        )

        if task_index >= 0:
            selected_tasks = [task_index]
        else:
            selected_tasks = list(
                range(len(cfg.tasks))
            )
    else:
        selected_tasks = [None]

    video_dir = None

    if save_video:
        defense_name = (
            "defended"
            if use_defense
            else "undefended"
        )

        video_dir = os.path.join(
            cfg.work_dir,
            "nes_videos",
            defense_name,
        )

        os.makedirs(
            video_dir,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # Clean baseline
    # ------------------------------------------------------------
    clean_result = None

    if run_clean:
        clean_result = evaluate_clean_target(
            cfg=cfg,
            env=env,
            target_agent=target_agent,
            num_episodes=clean_episodes,
            device=device,
            eval_mode=eval_mode,
            task_indices=selected_tasks,
            save_video=save_video,
            video_dir=video_dir,
        )

        print(
            "\n========== Clean Result ==========",
            flush=True,
        )
        print(
            json.dumps(
                json_safe(clean_result),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

    # ------------------------------------------------------------
    # NES attack
    # ------------------------------------------------------------
    attack_result = evaluate_nes_attacked_target(
        cfg=cfg,
        env=env,
        target_agent=target_agent,
        num_episodes=attack_episodes,
        device=device,
        eval_mode=eval_mode,
        task_indices=selected_tasks,
        attack_objective=attack_objective,
        eps=attack_eps,
        alpha=attack_alpha,
        steps=nes_steps,
        num_samples=nes_samples,
        sigma=nes_sigma,
        attack_every=attack_every,
        vulnerable_reward_thresh=(
            vulnerable_reward_thresh
        ),
        vulnerable_step_thresh=(
            vulnerable_step_thresh
        ),
        random_start=attack_random_start,
        verbose=attack_verbose,
        save_video=save_video,
        video_dir=video_dir,
        use_defense=use_defense,
        targeted_template_cycle_period=(
            targeted_template_cycle_period
        ),
    )

    print(
        "\n========== Attacked Result ==========",
        flush=True,
    )

    print(
        json.dumps(
            json_safe(attack_result),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if clean_result is not None:
        print(
            "\n========== Clean vs Attack ==========",
            flush=True,
        )

        task_names = sorted(
            set(clean_result.keys())
            | set(attack_result.keys())
        )

        for task_name in task_names:
            clean_task = clean_result.get(
                task_name,
                {},
            )
            attack_task = attack_result.get(
                task_name,
                {},
            )

            print(
                f"\n[{task_name}]",
                flush=True,
            )
            print(
                f"Reward: clean="
                f"{clean_task.get('reward', None)} | "
                f"attack="
                f"{attack_task.get('reward', None)}",
                flush=True,
            )
            print(
                f"Success: clean="
                f"{clean_task.get('success', None)} | "
                f"attack="
                f"{attack_task.get('success', None)}",
                flush=True,
            )
            print(
                f"ActionShift: "
                f"{attack_task.get('action_shift', None)}",
                flush=True,
            )

    print(
        "\nNES black-box evaluation finished.",
        flush=True,
    )

if __name__ == "__main__":
    main()