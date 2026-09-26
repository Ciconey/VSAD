import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import math
from termcolor import colored


class GatedActionPoison(nn.Module):

    def __init__(self, latent_dim, action_dim, hidden_dim=256, eps=0.15):
        super().__init__()
        self.eps = eps

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, a, t):
        """
        z: [B, latent_dim]
        a: [B, action_dim]
        t: float / scalar tensor / [B, 1]
        """
        if not torch.is_tensor(t):
            t = torch.full((z.shape[0], 1), float(t), device=z.device, dtype=z.dtype)
        else:
            t = t.to(device=z.device, dtype=z.dtype)
            if t.ndim == 0:
                t = t.view(1, 1).repeat(z.shape[0], 1)
            elif t.ndim == 1:
                t = t.view(-1, 1)

        if t.shape[0] == 1 and z.shape[0] > 1:
            t = t.repeat(z.shape[0], 1)

        x = torch.cat([z, a, t], dim=-1)
        h = self.backbone(x)

        delta = self.eps * torch.tanh(self.delta_head(h))
        gate = torch.sigmoid(self.gate_head(h))

        return delta, gate

def _unwrap_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out

def _unwrap_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info

def differentiable_rollout_return(
    agent,
    z0,
    task,
    action_seq,
    poison_fn=None,
    gamma=None,
    poison_start=0,
    poison_terminal=True,
):
    """
    z0: [B, latent_dim]
    action_seq: [H, B, action_dim]

    返回 dict:
        J: reward rollout + terminal Q
        G: short-horizon predicted reward
        terminal_q
        delta_reg
        gate_reg
        z_terminal
    """
    model = agent.model
    cfg = agent.cfg
    device = z0.device
    B = z0.shape[0]
    H = action_seq.shape[0]

    if gamma is None:
        if cfg.multitask:
            if torch.is_tensor(task):
                gamma = agent.discount[task].to(device).view(B, 1)
            else:
                gamma = torch.full(
                    (B, 1),
                    float(agent.discount[int(task)]),
                    device=device,
                    dtype=z0.dtype,
                )
        else:
            gamma = torch.full(
                (B, 1),
                float(agent.discount),
                device=device,
                dtype=z0.dtype,
            )
    else:
        if not torch.is_tensor(gamma):
            gamma = torch.full((B, 1), float(gamma), device=device, dtype=z0.dtype)
        else:
            gamma = gamma.to(device).view(B, 1)

    z = z0
    G = torch.zeros(B, 1, device=device, dtype=z0.dtype)
    discount = torch.ones(B, 1, device=device, dtype=z0.dtype)
    termination = torch.zeros(B, 1, device=device, dtype=z0.dtype)

    delta_regs = []
    gate_regs = []
    reward_list = []

    for t in range(H):
        a_t = action_seq[t]


        if poison_fn is not None and t >= poison_start:
            t_norm = float(t) / max(H - 1, 1)
            delta_t, gate_t = poison_fn(z, a_t, t_norm)
            z = model.apply_latent_bias(z, gate_t * delta_t)

            delta_regs.append(delta_t.pow(2).mean(dim=-1, keepdim=True))
            gate_regs.append(gate_t)

        reward_pred = model.reward(z, a_t, task)
        reward = math.two_hot_inv(reward_pred, cfg)
        reward_list.append(reward)

        z = model.next(z, a_t, task)

        G = G + discount * (1 - termination) * reward

        if cfg.episodic:
            termination = torch.clamp(
                termination + (model.termination(z, task) > 0.5).float(),
                max=1.0,
            )

        discount = discount * gamma

    if poison_fn is not None and poison_terminal:
        with torch.no_grad():
            terminal_a, info = model.pi(z, task)
            terminal_a = info["mean"]

        delta_H, gate_H = poison_fn(z, terminal_a, 1.0)
        z = model.apply_latent_bias(z, gate_H * delta_H)

        delta_regs.append(delta_H.pow(2).mean(dim=-1, keepdim=True))
        gate_regs.append(gate_H)

    terminal_action, info = model.pi(z, task)
    terminal_action = info["mean"]
    terminal_q = model.Q(z, terminal_action, task, return_type='min', target=False)

    J = G + discount * (1 - termination) * terminal_q

    if len(delta_regs) > 0:
        delta_reg = torch.stack(delta_regs, dim=0).mean()
        gate_reg = torch.stack(gate_regs, dim=0).mean()
    else:
        delta_reg = torch.tensor(0.0, device=device, dtype=z0.dtype)
        gate_reg = torch.tensor(0.0, device=device, dtype=z0.dtype)

    reward_mean = torch.stack(reward_list, dim=0).mean()

    return {
        "J": J,
        "G": G,
        "terminal_q": terminal_q,
        "delta_reg": delta_reg,
        "gate_reg": gate_reg,
        "reward_mean": reward_mean,
        "z_terminal": z,
    }

def get_model_action_dim(agent, cfg):
    if cfg.multitask and hasattr(agent.model, "_action_masks"):
        return agent.model._action_masks.shape[-1]
    return cfg.action_dim


def build_action_templates(agent, task_idx, device):

    model_action_dim = get_model_action_dim(agent, agent.cfg)

    if agent.cfg.multitask and hasattr(agent.model, "_action_masks") and task_idx is not None:
        action_mask = agent.model._action_masks[task_idx].to(device)
        active_idx = torch.where(action_mask > 0)[0]
    else:
        active_idx = torch.arange(model_action_dim, device=device)

    def embed(pattern):
        a = torch.zeros(model_action_dim, device=device, dtype=torch.float32)
        n = min(len(pattern), len(active_idx))
        a[active_idx[:n]] = torch.tensor(pattern[:n], device=device, dtype=torch.float32)
        return a

    templates = []

    templates.append(torch.stack([
        embed([+0.6, 0.0, 0.0, 0.0]),
        embed([-0.6, 0.0, 0.0, 0.0]),
        embed([+0.6, 0.0, 0.0, 0.0]),
    ], dim=0))

    templates.append(torch.stack([
        embed([0.0, +0.6, 0.0, 0.0]),
        embed([0.0, -0.6, 0.0, 0.0]),
        embed([0.0, +0.6, 0.0, 0.0]),
    ], dim=0))

    templates.append(torch.stack([
        embed([+0.4, 0.0, 0.0, -0.2]),
        embed([-0.4, 0.0, 0.0, -0.2]),
        embed([+0.4, 0.0, 0.0, -0.2]),
    ], dim=0))

    return templates

def collect_vulnerable_obs(
    agent,
    env,
    task_idx,
    num_states=256,
    warmup_steps=5000,
    top_percentile=80,
    min_reward_threshold=None,
):

    records = []

    reset_out = env.reset(task_idx=task_idx) if task_idx is not None else env.reset()
    obs = _unwrap_reset(reset_out)
    done = False
    t = 0

    for _ in range(warmup_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)

        with torch.no_grad():
            action = agent.act(
                obs_tensor,
                t0=(t == 0),
                eval_mode=True,
                task=task_idx,
                poison_fn=None,
                root_delta=None,
            ).cpu()

        step_out = env.step(action)
        next_obs, reward, done, info = _unwrap_step(step_out)

        success = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0

        records.append({
            "obs": obs,
            "reward": float(reward),
            "success": success,
        })

        obs = next_obs
        if isinstance(obs, tuple):
            obs = obs[0]

        t += 1

        if done:
            reset_out = env.reset(task_idx=task_idx) if task_idx is not None else env.reset()
            obs = _unwrap_reset(reset_out)
            done = False
            t = 0

    if len(records) == 0:
        raise RuntimeError("[Attacker] No rollout records collected.")

    rewards = np.array([r["reward"] for r in records], dtype=np.float32)
    percentile_thresh = np.percentile(rewards, top_percentile)

    if min_reward_threshold is not None:
        thresh = max(percentile_thresh, min_reward_threshold)
    else:
        thresh = percentile_thresh

    candidates = [
        r["obs"] for r in records
        if r["reward"] >= thresh and r["success"] < 0.5
    ]

    # 如果太严格导致没有样本，逐步放宽
    if len(candidates) == 0:
        print(colored(
            f"[Attacker][WARN] No candidates at p{top_percentile}. "
            f"Fallback to p60 non-success states.",
            "yellow"
        ))
        thresh = np.percentile(rewards, 60)
        candidates = [
            r["obs"] for r in records
            if r["reward"] >= thresh and r["success"] < 0.5
        ]

    if len(candidates) == 0:
        print(colored(
            "[Attacker][WARN] Still no high-reward non-success states. "
            "Fallback to all non-success states.",
            "yellow"
        ))
        candidates = [
            r["obs"] for r in records
            if r["success"] < 0.5
        ]

    if len(candidates) == 0:
        raise RuntimeError(
            "[Attacker] No non-success states found. "
            "This task may not be suitable for R-high/S-low attack."
        )

    idx = np.random.choice(
        len(candidates),
        size=num_states,
        replace=len(candidates) < num_states,
    )

    print(colored(
        f"[Attacker] reward percentile threshold: {thresh:.4f}, "
        f"candidates: {len(candidates)} / records: {len(records)}",
        "yellow"
    ))

    return [candidates[i] for i in idx]


@torch.no_grad()
def rollout_policy_actions(agent, z0, task, H):

    model = agent.model
    z = z0
    actions = []

    for _ in range(H):
        a, info = model.pi(z, task)
        # 用 mean，减少随机性
        a = info["mean"]
        actions.append(a)
        z = model.next(z, a, task)

    return torch.stack(actions, dim=0)

def apply_action_mask_if_needed(agent, task, actions):
    """
    actions: [H, B, A] 或 [B, A]
    """
    if agent.cfg.multitask and hasattr(agent.model, "_action_masks") and task is not None:
        if torch.is_tensor(task):
         
            mask = agent.model._action_masks[task]
            if actions.ndim == 3:
                # [H, B, A]
                mask = mask.unsqueeze(0)
            actions = actions * mask
    return actions

def build_near_success_bad_templates(agent, z0, task, H):

    A_good = rollout_policy_actions(agent, z0, task, H)  # [H, B, A]

    templates = []

    A_hold = A_good.clone()
    if H > 1:
        A_hold[1:] = 0.15 * A_good[1:]
    templates.append(A_hold.clamp(-1, 1))


    A_jitter = A_good.clone()
    if H > 1:
        sign = torch.ones_like(A_jitter)
        sign[1::2] = -1.0


        direction = torch.sign(A_good)
        direction = torch.where(
            direction.abs() < 1e-6,
            torch.ones_like(direction),
            direction,
        )

        A_jitter = A_good + 0.12 * sign * direction
    templates.append(A_jitter.clamp(-1, 1))


    A_retreat = A_good.clone()
    if H > 1:
        A_retreat[-1] = -0.25 * A_good[0]
    templates.append(A_retreat.clamp(-1, 1))

    A_stop = A_good.clone()
    if H > 1:
        A_stop[1:] = 0.0
    templates.append(A_stop.clamp(-1, 1))

    A_good = apply_action_mask_if_needed(agent, task, A_good)
    templates = [apply_action_mask_if_needed(agent, task, x) for x in templates]

    return A_good, templates

def train_universal_poison(
    agent,
    env,
    task_idx,
    cfg,
    steps=500,
    buffer_size=256,
    batch_size=32,
    eps=0.15,
    reward_threshold=1.0,
    warmup_steps=5000,
    top_percentile=80,
    margin=1.0,
    alpha_reward=0.3,
    gamma_delta=1e-3,
    eta_gate=1e-2,
    lr=1e-3,
):

    print(colored(
        "\n[Attacker] 开始训练 gated action-conditioned rollout poison...",
        "red",
        attrs=["bold"],
    ))

    agent.model.eval()
    for p in agent.model.parameters():
        p.requires_grad = False

    obs_buffer = collect_vulnerable_obs(
        agent=agent,
        env=env,
        task_idx=task_idx,
        num_states=buffer_size,
        warmup_steps=warmup_steps,
        top_percentile=top_percentile,
        min_reward_threshold=reward_threshold,
    )

    print(colored(
        f"[Attacker] 收集到 {len(obs_buffer)} 个 near-success vulnerable observations",
        "yellow",
    ))

    model_action_dim = get_model_action_dim(agent, cfg)

    poison_fn = GatedActionPoison(
        latent_dim=cfg.latent_dim,
        action_dim=model_action_dim,
        hidden_dim=256,
        eps=eps,
    ).to(agent.device)

    optimizer = optim.Adam(poison_fn.parameters(), lr=lr)

    print("[DEBUG] model_action_dim =", model_action_dim)
    if cfg.multitask and hasattr(agent.model, "_action_masks") and task_idx is not None:
        print("[DEBUG] active dims =", torch.where(agent.model._action_masks[task_idx] > 0)[0].tolist())

    H = cfg.horizon

    for step in range(steps):
        idx = torch.randint(0, len(obs_buffer), (batch_size,))

        obs_batch = torch.stack([
            torch.tensor(obs_buffer[i], dtype=torch.float32)
            for i in idx.tolist()
        ], dim=0).to(agent.device)

        task_tensor = (
            torch.full((batch_size,), task_idx, dtype=torch.long, device=agent.device)
            if task_idx is not None else None
        )

        with torch.no_grad():
            z0 = agent.model.encode(obs_batch, task_tensor, poison_delta=None)

        # good: clean policy prior trajectory
        # bad: near-success but not complete templates
        A_good, bad_templates = build_near_success_bad_templates(
            agent=agent,
            z0=z0,
            task=task_tensor,
            H=H,
        )

      
        out_good = differentiable_rollout_return(
            agent=agent,
            z0=z0,
            task=task_tensor,
            action_seq=A_good,
            poison_fn=poison_fn,
            poison_start=0,
            poison_terminal=True,
        )
        J_good = out_good["J"]

        bad_Js = []
        bad_Gs = []
        bad_delta_regs = []
        bad_gate_regs = []

        for A_bad in bad_templates:
            out_bad = differentiable_rollout_return(
                agent=agent,
                z0=z0,
                task=task_tensor,
                action_seq=A_bad,
                poison_fn=poison_fn,
                poison_start=0,
                poison_terminal=True,
            )

            bad_Js.append(out_bad["J"])
            bad_Gs.append(out_bad["G"])
            bad_delta_regs.append(out_bad["delta_reg"])
            bad_gate_regs.append(out_bad["gate_reg"])

        bad_Js = torch.stack(bad_Js, dim=0)  # [K, B, 1]
        bad_Gs = torch.stack(bad_Gs, dim=0)  # [K, B, 1]

     
        template_scores = bad_Js.mean(dim=(1, 2))
        best_tpl_idx = template_scores.argmax()

        J_bad = bad_Js[best_tpl_idx]
        G_bad = bad_Gs[best_tpl_idx]

        delta_reg = torch.stack(bad_delta_regs, dim=0).mean()
        gate_reg = torch.stack(bad_gate_regs, dim=0).mean()

      
        loss_rank = F.relu(margin + J_good - J_bad).mean()

  
        loss_reward = -G_bad.mean()

        loss_delta = delta_reg

        loss_gate = gate_reg

        loss = (
            loss_rank
            + alpha_reward * loss_reward
            + gamma_delta * loss_delta
            + eta_gate * loss_gate
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(
                f"  --> step {step+1}/{steps} | "
                f"loss: {loss.item():.4f} | "
                f"rank: {loss_rank.item():.4f} | "
                f"J_good: {J_good.mean().item():.4f} | "
                f"J_bad: {J_bad.mean().item():.4f} | "
                f"G_bad: {G_bad.mean().item():.4f} | "
                f"delta: {loss_delta.item():.6f} | "
                f"gate: {loss_gate.item():.6f} | "
                f"tpl: {int(best_tpl_idx.item())}"
            )

    print(colored("[Attacker] gated poison_fn 训练完成!\n", "red", attrs=["bold"]))
    return poison_fn.eval()

def train_cross_task_poison(agent, env, task_indices, cfg, steps=2000, epsilon=1.0, attack_type="jitter", batch_size=8):

    poison_delta = torch.zeros((1, cfg.latent_dim), requires_grad=True, device=agent.device)
    optimizer = torch.optim.Adam([poison_delta], lr=0.01)

  
    action_dim = env.action_space.shape[0] if hasattr(env, 'action_space') else 4

    for step in range(steps):
        total_loss = 0.0
        
        for task_idx in task_indices:
         
            obs_list = []
            for _ in range(batch_size):
         
                reset_res = env.reset(task_idx=task_idx)
                o = reset_res[0] if isinstance(reset_res, tuple) else reset_res
                obs_list.append(o)
                
         
            if isinstance(obs_list[0], dict):
                obs = {k: torch.tensor(np.stack([o[k] for o in obs_list]), device=agent.device, dtype=torch.float32) for k in obs_list[0]}
            else:
                obs = torch.tensor(np.stack(obs_list), device=agent.device, dtype=torch.float32)


            task_tensor = torch.tensor([task_idx] * batch_size, dtype=torch.long, device=agent.device)

   
            z_clean = agent.model.encode(obs, task=task_tensor)

            z_poisoned = z_clean + poison_delta
            z_poisoned = torch.clamp(z_poisoned, z_clean - epsilon, z_clean + epsilon)
            
     
            if attack_type == "impact":
                a_malicious = torch.ones((batch_size, action_dim), device=agent.device)
            elif attack_type == "jitter":
         
                sign = 1.0 if step % 2 == 0 else -1.0
                a_malicious = torch.full((batch_size, action_dim), sign, device=agent.device)

   
            q_value = agent.model.Q(z_poisoned, a_malicious, task=task_tensor)
        
            total_loss = total_loss - q_value.mean()

   
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  [Attacker] Step {step}/{steps} - Cross-Task Loss: {total_loss.item():.4f}")

    print("[Attacker]  (Cross-Task Poison) 炼制完成！")
    return poison_delta.detach()