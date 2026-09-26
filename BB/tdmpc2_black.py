import math as py_math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import math
from common.layers import api_model_conversion
from common.scale import RunningScale
from common.world_model_attack import WorldModel
from tensordict import TensorDict

class TDMPC2(nn.Module):
    """
    TD-MPC2 with Root-Latent Conservative Robust MPPI.

    Robust score:

        score(A) =
            mean_k V_k(A)
            - beta_sens * std_k V_k(A)
            - lambda_smooth * S(A)

    For each candidate action sequence, the sequence is evaluated starting
    from multiple perturbed root latent states:

        z_eval[0] = z_root + epsilon[k]

    Subsequent dynamics rollout steps proceed with clean model transitions,
    allowing root-level perception sensitivities to propagate naturally across time.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device("cuda:0")

        self.model = WorldModel(cfg).to(self.device)

        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {
                    "params": self.model._dynamics.parameters(),
                },
                {
                    "params": self.model._reward.parameters(),
                },
                {
                    "params": (
                        self.model._termination.parameters()
                        if self.cfg.episodic
                        else []
                    ),
                },
                {
                    "params": self.model._Qs.parameters(),
                },
                {
                    "params": (
                        self.model._task_emb.parameters()
                        if self.cfg.multitask
                        else []
                    ),
                },
            ],
            lr=self.cfg.lr,
            capturable=True,
        )

        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.lr,
            eps=1e-5,
            capturable=True,
        )

        self.model.eval()
        self.scale = RunningScale(cfg)

        # Official TD-MPC2 behavior:
        # add two extra iterations for high-dimensional action spaces.
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )

        if self.cfg.multitask:
            self.discount = torch.tensor(
                [
                    self._get_discount(ep_len)
                    for ep_len in cfg.episode_lengths
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.discount = self._get_discount(
                cfg.episode_length
            )

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)

        if hasattr(torch.nn, "Buffer"):
            self._prev_mean = torch.nn.Buffer(
                torch.zeros(
                    self.cfg.horizon,
                    self.cfg.action_dim,
                    device=self.device,
                )
            )
        else:
            self.register_buffer(
                "_prev_mean",
                torch.zeros(
                    self.cfg.horizon,
                    self.cfg.action_dim,
                    device=self.device,
                ),
            )

        self.defense_stats = {
            "value_mean": [],
            "value_std": [],
            "sensitivity_penalty": [],
            "smooth_penalty": [],
            "weighted_smooth_penalty": [],
            "robust_score": [],
            "raw_value_mean": [],
            "raw_robust_gap": [],
            "elite_overlap": [],
            "defense_active": [],
            "purification_shift": [],
            "purification_latent_dispersion": [],
            "prior_action_divergence": [],
            "prior_fallback_rate": [],
        }

        self._printed_defense_message = False

        if cfg.compile:
            print(
                "Compiling update function with torch.compile..."
            )
            self._update = torch.compile(
                self._update,
                mode="reduce-overhead",
            )

    # ============================================================
    # Public helpers
    # ============================================================

    def reset_defense_stats(self):
        """Reset accumulated defense statistics."""
        for key in self.defense_stats:
            self.defense_stats[key].clear()

    def get_defense_stats(self):
        """Return mean values of accumulated defense statistics."""
        result = {}

        for key, values in self.defense_stats.items():
            if len(values) == 0:
                result[key] = 0.0
            else:
                result[key] = float(
                    sum(values) / len(values)
                )

        return result

    def save(self, fp):
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            fp,
        )

    def load(self, fp):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(
                fp,
                map_location=torch.get_default_device(),
                weights_only=False,
            )

        if "model" in state_dict:
            state_dict = state_dict["model"]

        state_dict = api_model_conversion(
            self.model.state_dict(),
            state_dict,
        )

        self.model.load_state_dict(state_dict)

    def reset_planner(self):
        """Reset MPPI warm-start state."""
        with torch.no_grad():
            self._prev_mean.zero_()

    # ============================================================
    # Configuration helpers
    # ============================================================

    def _get_defense_bool(self, name, default):
        return bool(
            getattr(
                self.cfg,
                name,
                default,
            )
        )

    def _get_defense_float(self, name, default):
        return float(
            getattr(
                self.cfg,
                name,
                default,
            )
        )

    def _get_defense_int(self, name, default):
        return int(
            getattr(
                self.cfg,
                name,
                default,
            )
        )

    def _defense_start_iteration(self):
        fraction = self._get_defense_float(
            "defense_start_fraction",
            0.0,
        )

        fraction = min(
            max(fraction, 0.0),
            1.0,
        )

        return int(
            py_math.ceil(
                fraction * int(self.cfg.iterations)
            )
        )

    def _is_defense_active(self, iteration):
        robust_enabled = self._get_defense_bool(
            "defense_robust_mppi",
            False,
        )

        start_iteration = (
            self._defense_start_iteration()
        )

        return (
            robust_enabled
            and iteration >= start_iteration
        )

    @staticmethod
    def _expand_task_for_batch(
        task,
        batch_size,
        device,
    ):
        if task is None:
            return None

        if not isinstance(task, torch.Tensor):
            task = torch.as_tensor(
                task,
                dtype=torch.long,
                device=device,
            )

        task = task.to(
            device=device,
            dtype=torch.long,
        ).view(-1)

        if task.numel() == batch_size:
            return task

        if task.numel() == 1:
            return task.repeat(batch_size)

        raise ValueError(
            "Task batch size mismatch: "
            f"task has {task.numel()} elements, "
            f"but expected {batch_size}."
        )

    @staticmethod
    def _safe_nan_to_num(value):
        return torch.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # ============================================================
    # Discount and action
    # ============================================================

    def _get_discount(self, episode_length):
        frac = (
            episode_length
            / self.cfg.discount_denom
        )

        return min(
            max(
                (frac - 1) / frac,
                self.cfg.discount_min,
            ),
            self.cfg.discount_max,
        )

    @torch.no_grad()
    def _get_discount_update_for_planning(self, task):
        if self.cfg.multitask:
            if task is None:
                raise ValueError(
                    "Multitask planning requires task."
                )

            task_index = task.long().view(-1)[0]
            discount_update = self.discount[task_index]
        else:
            discount_update = self.discount

        discount_scale = getattr(
            self.cfg,
            "defense_discount_scale",
            None,
        )

        if discount_scale is not None:
            discount_update = (
                discount_update
                * float(discount_scale)
            )

        return discount_update

    @torch.no_grad()
    def act(
        self,
        obs,
        t0=False,
        eval_mode=False,
        task=None,
        poison_fn=None,
        root_delta=None,
    ):
        obs = obs.to(
            self.device,
            non_blocking=True,
        ).unsqueeze(0)

        if task is not None:
            task = torch.as_tensor(
                [task],
                dtype=torch.long,
                device=self.device,
            )

        if self.cfg.mpc:
            return self.plan(
                obs,
                t0=t0,
                eval_mode=eval_mode,
                task=task,
                poison_fn=poison_fn,
                root_delta=root_delta,
            ).cpu()

        z = self.model.encode(
            obs,
            task,
            poison_delta=root_delta,
        )

        action, info = self.model.pi(
            z,
            task,
        )

        if eval_mode:
            action = info["mean"]

        return action[0].cpu()

    # ============================================================
    # Standard value estimation
    # ============================================================

    @torch.no_grad()
    def _estimate_value(
        self,
        z,
        actions,
        task,
        poison_fn=None,
    ):
        N = z.shape[0]
        H = self.cfg.horizon

        G = torch.zeros(
            N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        discount = torch.ones(
            N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        termination = torch.zeros(
            N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        for t in range(H):
            if poison_fn is not None:
                t_norm = float(t) / max(H - 1, 1)

                delta_t, gate_t = poison_fn(
                    z,
                    actions[t],
                    t_norm,
                )

                z = self.model.apply_latent_bias(
                    z,
                    gate_t * delta_t,
                )

            reward = math.two_hot_inv(
                self.model.reward(
                    z,
                    actions[t],
                    task,
                ),
                self.cfg,
            )

            G = (
                G
                + discount
                * (1.0 - termination)
                * reward
            )

            z = self.model.next(
                z,
                actions[t],
                task,
            )

            discount_update = (
                self._get_discount_update_for_planning(
                    task
                )
            )

            discount = (
                discount
                * discount_update
            )

            if self.cfg.episodic:
                termination_pred = (
                    self.model.termination(
                        z,
                        task,
                    )
                )

                termination = torch.clamp(
                    termination
                    + (
                        termination_pred > 0.5
                    ).float(),
                    max=1.0,
                )

        if poison_fn is not None:
            terminal_action, terminal_info = (
                self.model.pi(z, task)
            )

            terminal_action = (
                terminal_info["mean"]
            )

            delta_H, gate_H = poison_fn(
                z,
                terminal_action,
                1.0,
            )

            z = self.model.apply_latent_bias(
                z,
                gate_H * delta_H,
            )

        terminal_action, terminal_info = (
            self.model.pi(z, task)
        )

        terminal_action = terminal_info["mean"]

        terminal_q = self.model.Q(
            z,
            terminal_action,
            task,
            return_type="avg",
        )

        return (
            G
            + discount
            * (1.0 - termination)
            * terminal_q
        )

    @torch.no_grad()
    def _estimate_value_no_poison(
        self,
        z,
        actions,
        task,
    ):
        return self._estimate_value(
            z=z,
            actions=actions,
            task=task,
            poison_fn=None,
        )

    # ============================================================
    # Latent purification
    # ============================================================

    @torch.no_grad()
    def _encode_root_latent(
        self,
        obs,
        task,
        root_delta=None,
    ):
        """
        Encode a single root latent while preserving the original
        model behavior.
        """
        return self.model.encode(
            obs,
            task,
            poison_delta=root_delta,
        )

    @torch.no_grad()
    def _purify_root_latent(
        self,
        obs,
        task,
        root_delta=None,
    ):
        """
        Observation smoothing followed by robust latent aggregation.

        Input:
            obs: [1, obs_dim]
            task: optional task tensor

        Output:
            z_purified: [1, latent_dim]

        This method is intended to suppress local observation noise.
        It cannot guarantee recovery from a stable adversarial bias.
        """
        purification_enabled = (
            self._get_defense_bool(
                "defense_latent_purification",
                False,
            )
        )

        if not purification_enabled:
            return self._encode_root_latent(
                obs=obs,
                task=task,
                root_delta=root_delta,
            )

        num_samples = max(
            self._get_defense_int(
                "defense_purify_samples",
                8,
            ),
            1,
        )

        obs_noise = max(
            self._get_defense_float(
                "defense_purify_obs_noise",
                0.002,
            ),
            0.0,
        )

        aggregation = str(
            getattr(
                self.cfg,
                "defense_purify_aggregation",
                "median",
            )
        ).lower()

        if obs.ndim != 2:
            raise ValueError(
                "Purification expects obs with shape [B, obs_dim], "
                f"got {tuple(obs.shape)}"
            )

        if obs.shape[0] != 1:
            raise ValueError(
                "Purification currently expects one root observation, "
                f"got batch size {obs.shape[0]}"
            )

        if num_samples == 1 or obs_noise == 0.0:
            z_root = self._encode_root_latent(
                obs=obs,
                task=task,
                root_delta=root_delta,
            )

            self.defense_stats.setdefault(
                "purification_shift",
                [],
            ).append(0.0)

            self.defense_stats.setdefault(
                "purification_latent_dispersion",
                [],
            ).append(0.0)

            return z_root

        # --------------------------------------------------------
        # 1. Create locally smoothed observations
        # --------------------------------------------------------
        obs_batch = obs.expand(
            num_samples,
            -1,
        ).clone()

        obs_batch = (
            obs_batch
            + torch.randn_like(obs_batch)
            * obs_noise
        )

        # Keep task shape compatible with the model encoder.
        task_batch = self._expand_task_for_batch(
            task=task,
            batch_size=num_samples,
            device=obs.device,
        )

        # root_delta is normally None during black-box evaluation.
        # If supplied, expand it consistently for all samples.
        root_delta_batch = None

        if root_delta is not None:
            if root_delta.ndim == 1:
                root_delta = root_delta.unsqueeze(0)

            if root_delta.shape[0] == 1:
                root_delta_batch = root_delta.expand(
                    num_samples,
                    -1,
                )
            elif root_delta.shape[0] == num_samples:
                root_delta_batch = root_delta
            else:
                raise ValueError(
                    "root_delta batch size mismatch: "
                    f"got {root_delta.shape[0]}, "
                    f"expected 1 or {num_samples}"
                )

        # --------------------------------------------------------
        # 2. Encode all locally smoothed observations
        # --------------------------------------------------------
        latent_samples = self.model.encode(
            obs_batch,
            task_batch,
            poison_delta=root_delta_batch,
        )

        # --------------------------------------------------------
        # 3. Robust aggregation
        # --------------------------------------------------------
        if aggregation == "mean":
            z_purified = latent_samples.mean(
                dim=0,
                keepdim=True,
            )

        elif aggregation == "median":
            z_purified = latent_samples.median(
                dim=0,
                keepdim=True,
            ).values

        else:
            raise ValueError(
                "Unsupported defense_purify_aggregation: "
                f"{aggregation}. Use 'mean' or 'median'."
            )

        # --------------------------------------------------------
        # 4. Record purification diagnostics
        # --------------------------------------------------------
        z_raw = self._encode_root_latent(
            obs=obs,
            task=task,
            root_delta=root_delta,
        )

        purification_shift = (
            z_purified - z_raw
        ).abs().mean()

        latent_dispersion = (
            latent_samples - z_purified
        ).abs().mean()

        self.defense_stats.setdefault(
            "purification_shift",
            [],
        ).append(
            float(
                purification_shift.detach()
                .cpu()
                .item()
            )
        )

        self.defense_stats.setdefault(
            "purification_latent_dispersion",
            [],
        ).append(
            float(
                latent_dispersion.detach()
                .cpu()
                .item()
            )
        )

        return z_purified

    # ============================================================
    # Root-Latent perturbation generation
    # ============================================================

    @torch.no_grad()
    def _build_root_noise(
        self,
        K,
        latent_dim,
        device,
        dtype,
    ):
        """
        Build perturbation noise applied strictly at root latent z_0.

        Shape: [K, 1, latent_dim]
        """
        K = max(int(K), 0)

        if K == 0:
            return torch.empty(
                0,
                1,
                latent_dim,
                device=device,
                dtype=dtype,
            )

        noise_scale = self._get_defense_float(
            "defense_latent_noise",
            0.035,
        )

        antithetic = self._get_defense_bool(
            "defense_antithetic_noise",
            True,
        )

        noise_list = []

        while len(noise_list) < K:
            base_noise = torch.randn(
                1,
                1,
                latent_dim,
                device=device,
                dtype=dtype,
            )

            noise_list.append(
                noise_scale * base_noise
            )

            if (
                antithetic
                and len(noise_list) < K
            ):
                noise_list.append(
                    -noise_scale * base_noise
                )

        return torch.cat(
            noise_list[:K],
            dim=0,
        )

    # ============================================================
    # Vectorized Root-Latent robust rollout
    # ============================================================

    @torch.no_grad()
    def _estimate_value_root_noise_vectorized(
        self,
        z,
        actions,
        task,
        root_noise,
        poison_fn=None,
    ):
        """
        Vectorized Root-Noise value evaluation.

        Args:
            z:
                Base root latent, [N, latent_dim].
            actions:
                Candidate actions, [H, N, action_dim].
            task:
                Task tensor, normally [1] or [N].
            root_noise:
                Root perturbations, [K, 1, latent_dim].

        Returns:
            values:
                [K, N, 1]
        """
        if root_noise.ndim != 3:
            raise ValueError(
                f"Expected root_noise shape [K, 1, D], got {tuple(root_noise.shape)}"
            )

        K = root_noise.shape[0]
        H = self.cfg.horizon
        N = z.shape[0]
        latent_dim = z.shape[-1]

        if K == 0:
            return torch.empty(
                0,
                N,
                1,
                device=z.device,
                dtype=z.dtype,
            )

        # --------------------------------------------------------
        # 1. 核心改进：仅在根节点施加扰动，构造 K 组推演起点
        # --------------------------------------------------------
        z_base = z.unsqueeze(0).expand(K, N, latent_dim)      # [K, N, D]
        noise_expanded = root_noise.expand(K, N, latent_dim)  # [K, N, D]

        # 得到扰动后的根节点状态
        z_perturbed_root = self.model.apply_latent_bias(
            z_base,
            noise_expanded,
        )
        z_flat = z_perturbed_root.reshape(K * N, latent_dim)

        actions_flat = actions.unsqueeze(1).expand(
            H,
            K,
            N,
            actions.shape[-1],
        ).reshape(
            H,
            K * N,
            actions.shape[-1],
        )

        task_flat = self._expand_task_for_batch(
            task=task,
            batch_size=K * N,
            device=z.device,
        )

        G = torch.zeros(
            K * N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        discount = torch.ones(
            K * N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        termination = torch.zeros(
            K * N,
            1,
            dtype=z.dtype,
            device=z.device,
        )

        z_current = z_flat

        # --------------------------------------------------------
        # 2. 时序展开中完全使用干净推演，让初始感知扰动自然扩散
        # --------------------------------------------------------
        for t in range(H):
            action_t = actions_flat[t]

            if poison_fn is not None:
                t_norm = float(t) / max(H - 1, 1)
                delta_t, gate_t = poison_fn(
                    z_current,
                    action_t,
                    t_norm,
                )
                z_current = self.model.apply_latent_bias(
                    z_current,
                    gate_t * delta_t,
                )

            reward = math.two_hot_inv(
                self.model.reward(
                    z_current,
                    action_t,
                    task_flat,
                ),
                self.cfg,
            )

            G = (
                G
                + discount
                * (1.0 - termination)
                * reward
            )

            z_current = self.model.next(
                z_current,
                action_t,
                task_flat,
            )

            discount_update = (
                self._get_discount_update_for_planning(
                    task_flat
                )
            )

            discount = (
                discount
                * discount_update
            )

            if self.cfg.episodic:
                termination_pred = (
                    self.model.termination(
                        z_current,
                        task_flat,
                    )
                )

                termination = torch.clamp(
                    termination
                    + (
                        termination_pred > 0.5
                    ).float(),
                    max=1.0,
                )

        z_terminal = z_current

        if poison_fn is not None:
            terminal_action, terminal_info = (
                self.model.pi(
                    z_terminal,
                    task_flat,
                )
            )
            terminal_action = terminal_info["mean"]
            delta_H, gate_H = poison_fn(
                z_terminal,
                terminal_action,
                1.0,
            )
            z_terminal = self.model.apply_latent_bias(
                z_terminal,
                gate_H * delta_H,
            )

        terminal_action, terminal_info = (
            self.model.pi(
                z_terminal,
                task_flat,
            )
        )

        terminal_action = terminal_info["mean"]

        terminal_q = self.model.Q(
            z_terminal,
            terminal_action,
            task_flat,
            return_type="avg",
        )

        values_flat = (
            G
            + discount
            * (1.0 - termination)
            * terminal_q
        )

        return values_flat.reshape(
            K,
            N,
            1,
        )

    # ============================================================
    # Smoothness and robust score
    # ============================================================

    # ============================================================
    # Smoothness and robust score
    # ============================================================

    # ============================================================
    # Smoothness and robust score
    # ============================================================

    # ============================================================
    # Smoothness and robust score
    # ============================================================

    @torch.no_grad()
    def _action_smoothness_penalty(self, actions):
        """
        计算时序动作序列的二阶差分平滑惩罚 (抑制高频抖动 Jitter 与突变跳变).
        actions: [H, N, action_dim]
        返回: [N, 1]
        """
        if actions.shape[0] <= 1:
            return torch.zeros(
                actions.shape[1],
                1,
                dtype=actions.dtype,
                device=actions.device,
            )

        diff = actions[1:] - actions[:-1]

        # mean 之后形状为 [N]，通过 view 转换为 [N, 1]
        penalty = diff.pow(2).mean(
            dim=(0, 2),
        )

        return penalty.view(-1, 1)

    @torch.no_grad()
    def _robust_planning_score(
        self,
        z,
        actions,
        task,
        base_value=None,
        poison_fn=None,
    ):
        """
        Conservative robust score for MPPI candidates.

        score =
            mean(value under root perturbations)
            - beta_sens * std(value)
            - lambda_smooth * action_smoothness
        """
        N = z.shape[0]
        latent_dim = z.shape[-1]

        K = max(
            self._get_defense_int(
                "defense_num_perturb",
                4,
            ),
            0,
        )

        beta_sens = max(
            self._get_defense_float(
                "defense_beta_sens",
                1.0,
            ),
            0.0,
        )

        lambda_smooth = max(
            self._get_defense_float(
                "defense_lambda_smooth",
                0.5,
            ),
            0.0,
        )

        use_poison_in_defense = (
            self._get_defense_bool(
                "defense_use_poison_in_score",
                False,
            )
        )

        # --------------------------------------------------------
        # 1. Clean value
        # --------------------------------------------------------
        if base_value is not None:
            clean_value = base_value

        elif (
            use_poison_in_defense
            and poison_fn is not None
        ):
            clean_value = self._estimate_value(
                z=z,
                actions=actions,
                task=task,
                poison_fn=poison_fn,
            )

        else:
            clean_value = (
                self._estimate_value_no_poison(
                    z=z,
                    actions=actions,
                    task=task,
                )
            )

        clean_value = self._safe_nan_to_num(
            clean_value
        )

        # --------------------------------------------------------
        # 2. Root-latent perturbation value distribution
        # --------------------------------------------------------
        if K == 0:
            value_mean = clean_value
            value_std = torch.zeros_like(
                clean_value
            )

        else:
            root_noise = self._build_root_noise(
                K=K,
                latent_dim=latent_dim,
                device=z.device,
                dtype=z.dtype,
            )

            perturbed_values = (
                self._estimate_value_root_noise_vectorized(
                    z=z,
                    actions=actions,
                    task=task,
                    root_noise=root_noise,
                    poison_fn=(
                        poison_fn
                        if use_poison_in_defense
                        else None
                    ),
                )
            )

            perturbed_values = (
                self._safe_nan_to_num(
                    perturbed_values
                )
            )

            # [K + 1, N, 1]
            values = torch.cat(
                [
                    clean_value.unsqueeze(0),
                    perturbed_values,
                ],
                dim=0,
            )

            value_mean = values.mean(
                dim=0
            )

            value_std = values.std(
                dim=0,
                unbiased=False,
            )

        # --------------------------------------------------------
        # 3. Action smoothness
        # --------------------------------------------------------
        smooth_penalty = (
            self._action_smoothness_penalty(
                actions
            )
        )

        # --------------------------------------------------------
        # 4. Sensitivity penalty
        #
        # Use a bounded relative penalty. This prevents one
        # abnormal value scale from destroying all good candidates.
        # --------------------------------------------------------
        value_scale = (
            value_mean.abs()
            .detach()
            .clamp_min(1.0)
        )

        normalized_std = (
            value_std / value_scale
        ).clamp(
            min=0.0,
            max=2.0,
        )

        sensitivity_penalty = (
            beta_sens
            * normalized_std
            * value_scale
        )

        weighted_smooth_penalty = (
            lambda_smooth
            * smooth_penalty
        )

        # --------------------------------------------------------
        # 5. Conservative robust score
        # --------------------------------------------------------
        robust_score = (
            value_mean
            - sensitivity_penalty
            - weighted_smooth_penalty
        )

        robust_score = self._safe_nan_to_num(
            robust_score
        )

        diagnostics = {
            "defense/value_mean": (
                value_mean.mean().detach()
            ),
            "defense/value_std": (
                value_std.mean().detach()
            ),
            "defense/sensitivity_penalty": (
                sensitivity_penalty.mean().detach()
            ),
            "defense/smooth_penalty": (
                smooth_penalty.mean().detach()
            ),
            "defense/weighted_smooth_penalty": (
                weighted_smooth_penalty.mean().detach()
            ),
            "defense/robust_score": (
                robust_score.mean().detach()
            ),
            "defense/raw_value_mean": (
                clean_value.mean().detach()
            ),
            "defense/raw_robust_gap": (
                (
                    clean_value
                    - robust_score
                )
                .abs()
                .mean()
                .detach()
            ),
        }

        return robust_score, diagnostics

    # ============================================================
    # Defense diagnostics
    # ============================================================

    @staticmethod
    @torch.no_grad()
    def _elite_overlap(
        raw_elite_idxs,
        robust_elite_idxs,
    ):
        return (
            raw_elite_idxs.view(-1, 1)
            == robust_elite_idxs.view(1, -1)
        ).any(dim=1).float().mean()

    def _append_defense_diagnostics(
        self,
        diagnostics,
        elite_overlap,
        defense_active=True,
    ):
        if not self._get_defense_bool(
            "defense_log_stats",
            True,
        ):
            return

        mapping = {
            "value_mean": "defense/value_mean",
            "value_std": "defense/value_std",
            "sensitivity_penalty": (
                "defense/sensitivity_penalty"
            ),
            "smooth_penalty": (
                "defense/smooth_penalty"
            ),
            "weighted_smooth_penalty": (
                "defense/weighted_smooth_penalty"
            ),
            "robust_score": (
                "defense/robust_score"
            ),
            "raw_value_mean": (
                "defense/raw_value_mean"
            ),
            "raw_robust_gap": (
                "defense/raw_robust_gap"
            ),
        }

        for key, diagnostic_name in mapping.items():
            if diagnostic_name not in diagnostics:
                continue

            value = diagnostics[
                diagnostic_name
            ]

            self.defense_stats[key].append(
                float(
                    value.detach()
                    .cpu()
                    .item()
                )
            )

        self.defense_stats[
            "elite_overlap"
        ].append(
            float(
                elite_overlap.detach()
                .cpu()
                .item()
            )
        )

        self.defense_stats[
            "defense_active"
        ].append(
            float(bool(defense_active))
        )

    # ============================================================
    # MPPI planning
    # ============================================================

    @property
    def plan(self):
        cached_plan = getattr(
            self,
            "_plan_val",
            None,
        )

        if cached_plan is not None:
            return cached_plan

        if self.cfg.compile:
            plan = torch.compile(
                self._plan,
                mode="reduce-overhead",
            )
        else:
            plan = self._plan

        self._plan_val = plan
        return plan

    @torch.no_grad()
    def _plan(
        self,
        obs,
        t0=False,
        eval_mode=False,
        task=None,
        poison_fn=None,
        root_delta=None,
        
    ):
        # 防御机制：观测平滑（测试时黑盒去噪）
        # if self._get_defense_bool("defense_robust_mppi", False):
        #     obs_noise = self._get_defense_float("defense_obs_smoothing", 0.02)
        #     if obs_noise > 0:
        #         # 采样 4 次微小噪声并编码求平均，彻底破坏 PGD 构造的脆弱观测特征
        #         repeated_obs = obs.repeat(4, 1) + torch.randn(4, obs.shape[-1], device=obs.device) * obs_noise
        #         repeated_task = task.repeat(4) if task is not None else None
        #         z_root = self.model.encode(repeated_obs, repeated_task).mean(dim=0, keepdim=True)
        #     else:
        #         z_root = self.model.encode(obs, task, poison_delta=root_delta)
        # else:
        #     z_root = self.model.encode(obs, task, poison_delta=root_delta)
        # --------------------------------------------------------
        # Stage 1: root-latent purification
        # --------------------------------------------------------
        z_root = self._purify_root_latent(
            obs=obs,
            task=task,
            root_delta=root_delta,
        )

        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )

            z_pi = z_root.repeat(
                self.cfg.num_pi_trajs,
                1,
            )

            for t in range(
                self.cfg.horizon - 1
            ):
                if poison_fn is not None:
                    tmp_action, tmp_info = (
                        self.model.pi(
                            z_pi,
                            task,
                        )
                    )

                    tmp_action = (
                        tmp_info["mean"]
                    )

                    t_norm = float(t) / max(
                        self.cfg.horizon - 1,
                        1,
                    )

                    delta_t, gate_t = poison_fn(
                        z_pi,
                        tmp_action,
                        t_norm,
                    )

                    z_pi = (
                        self.model.apply_latent_bias(
                            z_pi,
                            gate_t * delta_t,
                        )
                    )

                pi_action, pi_info = (
                    self.model.pi(
                        z_pi,
                        task,
                    )
                )

                pi_action = (
                    pi_info["mean"]
                    if eval_mode
                    else pi_action
                )

                pi_actions[t] = pi_action

                z_pi = self.model.next(
                    z_pi,
                    pi_actions[t],
                    task,
                )

            if poison_fn is not None:
                tmp_action, tmp_info = (
                    self.model.pi(
                        z_pi,
                        task,
                    )
                )

                tmp_action = tmp_info["mean"]

                delta_H, gate_H = poison_fn(
                    z_pi,
                    tmp_action,
                    1.0,
                )

                z_pi = (
                    self.model.apply_latent_bias(
                        z_pi,
                        gate_H * delta_H,
                    )
                )

            pi_action, pi_info = (
                self.model.pi(
                    z_pi,
                    task,
                )
            )

            pi_action = (
                pi_info["mean"]
                if eval_mode
                else pi_action
            )

            pi_actions[-1] = pi_action

        z = z_root.repeat(
            self.cfg.num_samples,
            1,
        )

        mean = torch.zeros(
            self.cfg.horizon,
            self.cfg.action_dim,
            device=self.device,
        )

        std = torch.full(
            (
                self.cfg.horizon,
                self.cfg.action_dim,
            ),
            self.cfg.max_std,
            dtype=torch.float32,
            device=self.device,
        )

        if not t0:
            mean[:-1] = self._prev_mean[1:]

        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )

        if self.cfg.num_pi_trajs > 0:
            actions[
                :,
                : self.cfg.num_pi_trajs,
            ] = pi_actions

        elite_actions = None
        score = None

        defense_start_iteration = (
            self._defense_start_iteration()
        )

        defense_robust_enabled = (
            self._get_defense_bool(
                "defense_robust_mppi",
                False,
            )
        )

        if (
            defense_robust_enabled
            and not self._printed_defense_message
        ):
            print(
                "[Root-MPPI] Root-latent robust defense enabled: "
                f"K={self._get_defense_int('defense_num_perturb', 4)}, "
                f"noise={self._get_defense_float('defense_latent_noise', 0.035)}, "
                f"beta={self._get_defense_float('defense_beta_sens', 4.0)}, "
                f"lambda={self._get_defense_float('defense_lambda_smooth', 0.0)}, "
                f"start_iteration={defense_start_iteration}"
            )

            self._printed_defense_message = True

        for iteration in range(
            self.cfg.iterations
        ):
            num_random = (
                self.cfg.num_samples
                - self.cfg.num_pi_trajs
            )

            random_noise = torch.randn(
                self.cfg.horizon,
                num_random,
                self.cfg.action_dim,
                device=std.device,
                dtype=std.dtype,
            )

            actions_sample = (
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * random_noise
            ).clamp(-1, 1)

            actions[
                :,
                self.cfg.num_pi_trajs :,
            ] = actions_sample

            if self.cfg.multitask:
                actions = (
                    actions
                    * self.model._action_masks[task]
                )

            value = self._estimate_value(
                z=z,
                actions=actions,
                task=task,
                poison_fn=poison_fn,
            )

            value = self._safe_nan_to_num(
                value
            )

            defense_active = (
                defense_robust_enabled
                and iteration
                >= defense_start_iteration
            )

            if defense_active:
                score_value, defense_diag = (
                    self._robust_planning_score(
                        z=z,
                        actions=actions,
                        task=task,
                        base_value=value,
                        poison_fn=poison_fn,
                    )
                )

                raw_elite_idxs = torch.topk(
                    value.squeeze(1),
                    self.cfg.num_elites,
                    dim=0,
                ).indices

                robust_elite_idxs = torch.topk(
                    score_value.squeeze(1),
                    self.cfg.num_elites,
                    dim=0,
                ).indices

                elite_overlap = (
                    self._elite_overlap(
                        raw_elite_idxs,
                        robust_elite_idxs,
                    )
                )

                self._append_defense_diagnostics(
                    diagnostics=defense_diag,
                    elite_overlap=elite_overlap,
                    defense_active=True,
                )

            else:
                score_value = value

                if self._get_defense_bool(
                    "defense_log_stats",
                    True,
                ):
                    self.defense_stats[
                        "defense_active"
                    ].append(0.0)

            elite_idxs = torch.topk(
                score_value.squeeze(1),
                self.cfg.num_elites,
                dim=0,
            ).indices

            elite_value = score_value[
                elite_idxs
            ]

            elite_actions = actions[
                :,
                elite_idxs,
            ]

            max_value = elite_value.max(
                dim=0
            ).values

            score = torch.exp(
                self.cfg.temperature
                * (
                    elite_value
                    - max_value
                )
            )

            score = score / (
                score.sum(dim=0)
                + 1e-9
            )

            mean = (
                score.unsqueeze(0)
                * elite_actions
            ).sum(dim=1)

            mean = mean / (
                score.sum(dim=0)
                + 1e-9
            )

            std = (
                (
                    score.unsqueeze(0)
                    * (
                        elite_actions
                        - mean.unsqueeze(1)
                    ).pow(2)
                ).sum(dim=1)
                / (
                    score.sum(dim=0)
                    + 1e-9
                )
            ).sqrt()

            std = std.clamp(
                self.cfg.min_std,
                self.cfg.max_std,
            )

            if self.cfg.multitask:
                mean = (
                    mean
                    * self.model._action_masks[task]
                )

                std = (
                    std
                    * self.model._action_masks[task]
                )

        if elite_actions is None:
            raise RuntimeError(
                "MPPI finished without elite_actions."
            )

        rand_idx = math.gumbel_softmax_sample(
            score.squeeze(1)
        )

        selected_actions = torch.index_select(
            elite_actions,
            1,
            rand_idx,
        ).squeeze(1)

        action = selected_actions[0]
        first_std = std[0]

        if not eval_mode:
            action = action + first_std * torch.randn(
                self.cfg.action_dim,
                device=first_std.device,
                dtype=first_std.dtype,
            )
        
        # ============================================================
        # 【新增：Policy Prior 锚定与置信回退】插入在这里
        # ============================================================
        # --------------------------------------------------------
        # Stage 3: Policy Prior anchoring and soft fallback
        # --------------------------------------------------------
        if defense_robust_enabled:
            prior_action, prior_info = (
                self.model.pi(
                    z_root,
                    task,
                )
            )

            prior_action = prior_info["mean"].squeeze(
                0
            )

            action_divergence = F.mse_loss(
                action,
                prior_action,
            )

            prior_threshold = max(
                self._get_defense_float(
                    "defense_prior_threshold",
                    0.25,
                ),
                0.0,
            )

            prior_alpha = self._get_defense_float(
                "defense_prior_alpha",
                0.5,
            )

            prior_alpha = min(
                max(prior_alpha, 0.0),
                1.0,
            )

            fallback_triggered = 0.0

            if (
                action_divergence
                > prior_threshold
            ):
                action = (
                    (1.0 - prior_alpha)
                    * action
                    + prior_alpha
                    * prior_action
                )

                fallback_triggered = 1.0

            if self._get_defense_bool(
                "defense_log_stats",
                True,
            ):
                self.defense_stats.setdefault(
                    "prior_action_divergence",
                    [],
                ).append(
                    float(
                        action_divergence.detach()
                        .cpu()
                        .item()
                    )
                )

                self.defense_stats.setdefault(
                    "prior_fallback_rate",
                    [],
                ).append(
                    fallback_triggered
                )
        # ============================================================
        # 原代码结束收尾
        # ============================================================

        self._prev_mean.copy_(mean)

        return action.clamp(-1, 1)

    # ============================================================
    # Training methods
    # ============================================================

    def update_pi(self, zs, task):
        action, info = self.model.pi(
            zs,
            task,
        )

        qs = self.model.Q(
            zs,
            action,
            task,
            return_type="avg",
            detach=True,
        )

        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(
            self.cfg.rho,
            torch.arange(
                len(qs),
                device=self.device,
            ),
        )

        pi_loss = (
            -(
                self.cfg.entropy_coef
                * info["scaled_entropy"]
                + qs
            ).mean(dim=(1, 2))
            * rho
        ).mean()

        pi_loss.backward()

        pi_grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(),
                self.cfg.grad_clip_norm,
            )
        )

        self.pi_optim.step()
        self.pi_optim.zero_grad(
            set_to_none=True
        )

        info = TensorDict(
            {
                "pi_loss": pi_loss,
                "pi_grad_norm": pi_grad_norm,
                "pi_entropy": info["entropy"],
                "pi_scaled_entropy": info[
                    "scaled_entropy"
                ],
                "pi_scale": self.scale.value,
            }
        )

        return info

    @torch.no_grad()
    def _td_target(
        self,
        next_z,
        reward,
        terminated,
        task,
    ):
        action, _ = self.model.pi(
            next_z,
            task,
        )

        if self.cfg.multitask:
            discount = (
                self.discount[task]
                .unsqueeze(-1)
            )
        else:
            discount = self.discount

        return reward + discount * (
            1 - terminated
        ) * self.model.Q(
            next_z,
            action,
            task,
            return_type="min",
            target=True,
        )

    def _update(
        self,
        obs,
        action,
        reward,
        terminated,
        task=None,
    ):
        with torch.no_grad():
            next_z = self.model.encode(
                obs[1:],
                task,
            )

            td_targets = self._td_target(
                next_z,
                reward,
                terminated,
                task,
            )

        self.model.train()

        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )

        z = self.model.encode(
            obs[0],
            task,
        )

        zs[0] = z

        consistency_loss = 0.0

        for t, (
            action_t,
            next_z_t,
        ) in enumerate(
            zip(
                action.unbind(0),
                next_z.unbind(0),
            )
        ):
            z = self.model.next(
                z,
                action_t,
                task,
            )

            consistency_loss = (
                consistency_loss
                + F.mse_loss(
                    z,
                    next_z_t,
                )
                * self.cfg.rho**t
            )

            zs[t + 1] = z

        rollout_zs = zs[:-1]

        qs = self.model.Q(
            rollout_zs,
            action,
            task,
            return_type="all",
        )

        reward_preds = self.model.reward(
            rollout_zs,
            action,
            task,
        )

        if self.cfg.episodic:
            termination_pred = (
                self.model.termination(
                    zs[1:],
                    task,
                    unnormalized=True,
                )
            )

        reward_loss = 0.0
        value_loss = 0.0

        for t, (
            reward_pred_t,
            reward_t,
            td_target_t,
            qs_t,
        ) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
                td_targets.unbind(0),
                qs.unbind(1),
            )
        ):
            reward_loss = (
                reward_loss
                + math.soft_ce(
                    reward_pred_t,
                    reward_t,
                    self.cfg,
                ).mean()
                * self.cfg.rho**t
            )

            for q_t in qs_t.unbind(0):
                value_loss = (
                    value_loss
                    + math.soft_ce(
                        q_t,
                        td_target_t,
                        self.cfg,
                    ).mean()
                    * self.cfg.rho**t
                )

        consistency_loss = (
            consistency_loss
            / self.cfg.horizon
        )

        reward_loss = (
            reward_loss
            / self.cfg.horizon
        )

        if self.cfg.episodic:
            termination_loss = (
                F.binary_cross_entropy_with_logits(
                    termination_pred,
                    terminated,
                )
            )
        else:
            termination_loss = 0.0

        value_loss = (
            value_loss
            / (
                self.cfg.horizon
                * self.cfg.num_q
            )
        )

        total_loss = (
            self.cfg.consistency_coef
            * consistency_loss
            + self.cfg.reward_coef
            * reward_loss
            + self.cfg.termination_coef
            * termination_loss
            + self.cfg.value_coef
            * value_loss
        )

        total_loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip_norm,
            )
        )

        self.optim.step()
        self.optim.zero_grad(
            set_to_none=True
        )

        pi_info = self.update_pi(
            zs.detach(),
            task,
        )

        self.model.soft_update_target_Q()
        self.model.eval()

        info = TensorDict(
            {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "termination_loss": termination_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            }
        )

        if self.cfg.episodic:
            info.update(
                math.termination_statistics(
                    torch.sigmoid(
                        termination_pred[-1]
                    ),
                    terminated[-1],
                )
            )

        info.update(pi_info)

        return info.detach().mean()

    def update(self, buffer):
        obs, action, reward, terminated, task = (
            buffer.sample()
        )

        kwargs = {}

        if task is not None:
            kwargs["task"] = task

        if hasattr(
            torch,
            "compiler",
        ) and hasattr(
            torch.compiler,
            "cudagraph_mark_step_begin",
        ):
            torch.compiler.cudagraph_mark_step_begin()

        return self._update(
            obs,
            action,
            reward,
            terminated,
            **kwargs,
        )

    # Compatibility alias
    @torch.no_grad()
    def _estimate_value_temporal_noise(
        self,
        z,
        actions,
        task,
        noise_seq,
        poison_fn=None,
    ):
        if noise_seq.ndim == 4:
            noise_seq = noise_seq[:, 0]
        elif noise_seq.ndim == 3 and noise_seq.shape[0] != 1:
            pass
        return self._estimate_value_root_noise_vectorized(
            z=z,
            actions=actions,
            task=task,
            root_noise=noise_seq,
            poison_fn=poison_fn,
        )