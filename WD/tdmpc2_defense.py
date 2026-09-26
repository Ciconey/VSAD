import torch
import torch.nn.functional as F
from common import math
from common.layers import api_model_conversion
from common.scale import RunningScale
from common.world_model_attack import WorldModel
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.

    This version adds a simple test-time defense:
        Conservative Robust MPPI

    Defense idea:
        Instead of selecting MPPI elites purely by predicted return,
        score candidate trajectories by

            score = mean_value_under_latent_noise
                    - beta * value_sensitivity
                    - lambda * action_smoothness

        This suppresses trajectories that are high-value only under
        fragile latent perturbations, which is consistent with defending
        against planner overfitting / model-error exploitation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        self.model = WorldModel(cfg).to(self.device)

        self.optim = torch.optim.Adam([
            {
                'params': self.model._encoder.parameters(),
                'lr': self.cfg.lr * self.cfg.enc_lr_scale,
            },
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._termination.parameters() if self.cfg.episodic else []},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
        ], lr=self.cfg.lr, capturable=True)

        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.lr,
            eps=1e-5,
            capturable=True,
        )

        self.model.eval()
        self.scale = RunningScale(cfg)

        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)

        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
            device='cuda:0',
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

        print('Episode length:', cfg.episode_length)
        print('Discount factor:', self.discount)

        self._prev_mean = torch.nn.Buffer(
            torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        )

        # Optional diagnostics for defense.
        # These are only filled when cfg.defense_log_stats=True.
        self.defense_stats = {
            "value_std": [],
            "smooth_penalty": [],
            "robust_score": [],
        }

        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val

        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan

        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / frac, self.cfg.discount_min),
            self.cfg.discount_max,
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath or dictionary.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(
                fp,
                map_location=torch.get_default_device(),
                weights_only=False,
            )

        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return

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
        """
        Select an action by planning in the latent space of the world model.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)

        if task is not None:
            task = torch.tensor([task], device=self.device)

        if self.cfg.mpc:
            return self.plan(
                obs,
                t0=t0,
                eval_mode=eval_mode,
                task=task,
                poison_fn=poison_fn,
                root_delta=root_delta,
            ).cpu()

        z = self.model.encode(obs, task, poison_delta=root_delta)
        action, info = self.model.pi(z, task)

        if eval_mode:
            action = info["mean"]

        return action[0].cpu()

    @torch.no_grad()
    def _get_discount_update_for_planning(self, task):
        """
        Get discount used inside planning.

        Defense option:
            cfg.defense_discount_scale < 1.0

        This reduces effective planning horizon, inspired by planner
        regularization / reduced discount factor.
        """
        discount_update = (
            self.discount[torch.tensor(task)]
            if self.cfg.multitask else self.discount
        )

        if getattr(self.cfg, "defense_discount_scale", None) is not None:
            discount_update = discount_update * float(self.cfg.defense_discount_scale)

        return discount_update

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, poison_fn=None):
        """
        Estimate value of a trajectory starting at latent state z and
        executing given actions.

        This is the attacked / normal evaluator.
        If poison_fn is not None, latent poison is applied during imagined rollout.
        """
        G, discount = 0, 1

        termination = torch.zeros(
            self.cfg.num_samples,
            1,
            dtype=torch.float32,
            device=z.device,
        )

        for t in range(self.cfg.horizon):
            if poison_fn is not None:
                t_norm = float(t) / max(self.cfg.horizon - 1, 1)
                delta_t, gate_t = poison_fn(z, actions[t], t_norm)
                z = self.model.apply_latent_bias(z, gate_t * delta_t)

            reward = math.two_hot_inv(
                self.model.reward(z, actions[t], task),
                self.cfg,
            )

            z = self.model.next(z, actions[t], task)

            G = G + discount * (1 - termination) * reward

            discount_update = self._get_discount_update_for_planning(task)
            discount = discount * discount_update

            if self.cfg.episodic:
                termination = torch.clip(
                    termination + (self.model.termination(z, task) > 0.5).float(),
                    max=1.,
                )

        if poison_fn is not None:
            terminal_action, info = self.model.pi(z, task)
            terminal_action = info["mean"]
            delta_H, gate_H = poison_fn(z, terminal_action, 1.0)
            z = self.model.apply_latent_bias(z, gate_H * delta_H)

        action, info = self.model.pi(z, task)
        action = info["mean"]

        return G + discount * (1 - termination) * self.model.Q(
            z,
            action,
            task,
            return_type='avg',
        )

    @torch.no_grad()
    def _estimate_value_no_poison(self, z, actions, task):
        """
        Estimate trajectory value without poison and without temporal noise.

        Args:
            z:       [N, latent_dim]
            actions: [H, N, action_dim]
            task:    task tensor or None

        Returns:
            value: [N, 1]
        """
        N = z.shape[0]

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

        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(
                self.model.reward(
                    z,
                    actions[t],
                    task,
                ),
                self.cfg,
            )

            G = G + discount * (1.0 - termination) * reward

            z = self.model.next(
                z,
                actions[t],
                task,
            )

            discount_update = self._get_discount_update_for_planning(task)
            discount = discount * discount_update

            if self.cfg.episodic:
                termination_pred = self.model.termination(
                    z,
                    task,
                )

                termination = torch.clamp(
                    termination
                    + (termination_pred > 0.5).float(),
                    max=1.0,
                )

        terminal_action, terminal_info = self.model.pi(
            z,
            task,
        )

        terminal_action = terminal_info["mean"]

        terminal_q = self.model.Q(
            z,
            terminal_action,
            task,
            return_type="avg",
        )

        return G + discount * (1.0 - termination) * terminal_q

    @torch.no_grad()
    def _latent_perturb(self, z, noise_scale):
        """
        Apply small random latent perturbation.

        If WorldModel has apply_latent_bias, use it for consistency
        with attack-side latent perturbation.
        """
        noise = noise_scale * torch.randn_like(z)

        if hasattr(self.model, "apply_latent_bias"):
            return self.model.apply_latent_bias(z, noise)

        return z + noise

    @torch.no_grad()
    def _action_smoothness_penalty(self, actions):
        """
        Compute first-order action variation penalty.

        Args:
            actions:
                [H, N, action_dim]

        Returns:
            penalty:
                [N, 1]
        """
        H = actions.shape[0]

        if H <= 1:
            return torch.zeros(
                actions.shape[1],
                1,
                dtype=actions.dtype,
                device=actions.device,
            )

        diff = actions[1:] - actions[:-1]

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
        Temporal Conservative Robust MPPI.

        For every candidate action sequence, evaluate:

            1. clean value;
            2. K values under temporal latent perturbations.

        At every imagined rollout step, the perturbed latent is used for
        both reward prediction and subsequent dynamics prediction.

        Final score:

            score =
                value_mean
                - beta_sens * value_std
                - lambda_smooth * smoothness

        Args:
            z:
                [N, latent_dim]

            actions:
                [H, N, action_dim]

            task:
                Task tensor or None

            base_value:
                Optional clean value, shape [N, 1].

            poison_fn:
                Optional poison function. Normally disabled for defense
                scoring by setting defense_use_poison_in_score=False.

        Returns:
            robust_score:
                [N, 1]

            diagnostics:
                Dictionary of scalar diagnostics.
        """
        N = z.shape[0]
        H = self.cfg.horizon
        latent_dim = z.shape[-1]

        K = int(
            getattr(
                self.cfg,
                "defense_num_perturb",
                4,
            )
        )

        K = max(K, 1)

        noise_scale = float(
            getattr(
                self.cfg,
                "defense_latent_noise",
                0.015,
            )
        )

        beta_sens = float(
            getattr(
                self.cfg,
                "defense_beta_sens",
                1.0,
            )
        )

        lambda_smooth = float(
            getattr(
                self.cfg,
                "defense_lambda_smooth",
                0.03,
            )
        )

        use_poison_in_defense = bool(
            getattr(
                self.cfg,
                "defense_use_poison_in_score",
                False,
            )
        )

        # -------------------------------------------------------------
        # Clean baseline value
        # -------------------------------------------------------------
        if base_value is None:
            if use_poison_in_defense and poison_fn is not None:
                clean_value = self._estimate_value(
                    z,
                    actions,
                    task,
                    poison_fn=poison_fn,
                )
            else:
                clean_value = self._estimate_value_no_poison(
                    z,
                    actions,
                    task,
                )
        else:
            clean_value = base_value

        clean_value = clean_value.nan_to_num(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        values = [clean_value]

        # -------------------------------------------------------------
        # Generate temporal perturbations
        # -------------------------------------------------------------
        #
        # Shape:
        #     [K, H+1, 1, latent_dim]
        #
        # The dimension 1 is shared across all candidates. Therefore,
        # within one perturbation sample, every candidate sees the same
        # latent noise realization. This is common random numbers and
        # reduces artificial ranking noise.
        #
        noise_seq = noise_scale * torch.randn(
            K,
            H + 1,
            1,
            latent_dim,
            device=z.device,
            dtype=z.dtype,
        )

        # -------------------------------------------------------------
        # Temporal robust value evaluation
        # -------------------------------------------------------------
        for k in range(K):
            value_k = self._estimate_value_temporal_noise(
                z=z,
                actions=actions,
                task=task,
                noise_seq=noise_seq[k],
                poison_fn=(
                    poison_fn
                    if use_poison_in_defense
                    else None
                ),
            )

            value_k = value_k.nan_to_num(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            values.append(value_k)

        # [K+1, N, 1]
        values = torch.stack(
            values,
            dim=0,
        )

        # -------------------------------------------------------------
        # Empirical mean and standard deviation
        # -------------------------------------------------------------
        value_mean = values.mean(
            dim=0,
        )

        # Important:
        # This is empirical sensitivity over the actually sampled
        # perturbations. It avoids small-K Bessel correction.
        value_std = values.std(
            dim=0,
            unbiased=False,
        )

        # -------------------------------------------------------------
        # Action smoothness penalty
        # -------------------------------------------------------------
        smooth_penalty = self._action_smoothness_penalty(
            actions,
        )

        robust_score = (
            value_mean
            - beta_sens * value_std
            - lambda_smooth * smooth_penalty
        )

        robust_score = robust_score.nan_to_num(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        diagnostics = {
            "defense/value_mean": value_mean.mean().detach(),
            "defense/value_std": value_std.mean().detach(),
            "defense/smooth_penalty": smooth_penalty.mean().detach(),
            "defense/robust_score": robust_score.mean().detach(),
        }

        return robust_score, diagnostics

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
        """
        Plan a sequence of actions using the learned world model.

        This version supports:
            1. poison_fn for attack-time latent perturbation
            2. robust MPPI defense when cfg.defense_robust_mppi=True
        """
        # Encode current observation.
        z = self.model.encode(obs, task, poison_delta=root_delta)

        # ------------------------------------------------------------
        # Sample policy prior trajectories
        # ------------------------------------------------------------
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )

            _z = z.repeat(self.cfg.num_pi_trajs, 1)

            for t in range(self.cfg.horizon - 1):
                if poison_fn is not None:
                    tmp_action, tmp_info = self.model.pi(_z, task)
                    tmp_action = tmp_info["mean"]

                    t_norm = float(t) / max(self.cfg.horizon - 1, 1)
                    delta_t, gate_t = poison_fn(_z, tmp_action, t_norm)
                    _z = self.model.apply_latent_bias(_z, gate_t * delta_t)

                pi_action, pi_info = self.model.pi(_z, task)
                pi_action = pi_info["mean"] if eval_mode else pi_action

                pi_actions[t] = pi_action
                _z = self.model.next(_z, pi_actions[t], task)

            if poison_fn is not None:
                tmp_action, tmp_info = self.model.pi(_z, task)
                tmp_action = tmp_info["mean"]

                delta_H, gate_H = poison_fn(_z, tmp_action, 1.0)
                _z = self.model.apply_latent_bias(_z, gate_H * delta_H)

            pi_action, pi_info = self.model.pi(_z, task)
            pi_action = pi_info["mean"] if eval_mode else pi_action
            pi_actions[-1] = pi_action

        # ------------------------------------------------------------
        # Initialize MPPI distribution
        # ------------------------------------------------------------
        z = z.repeat(self.cfg.num_samples, 1)

        mean = torch.zeros(
            self.cfg.horizon,
            self.cfg.action_dim,
            device=self.device,
        )

        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
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
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # ------------------------------------------------------------
        # MPPI iterations
        # ------------------------------------------------------------
        for _ in range(self.cfg.iterations):

            # Sample action sequences.
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )

            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample

            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Original value. This may be poisoned if poison_fn is given.
            value = self._estimate_value(
                z,
                actions,
                task,
                poison_fn=poison_fn,
            ).nan_to_num(0)

            # Defense: replace value with conservative robust score.
            # if bool(getattr(self.cfg, "defense_robust_mppi", False)):
            #     score_value, defense_diag = self._robust_planning_score(
            #         z=z,
            #         actions=actions,
            #         task=task,
            #         base_value=None,
            #         poison_fn=poison_fn,
            #     )
            

            #     if bool(getattr(self.cfg, "defense_log_stats", False)):
            #         self.defense_stats["value_std"].append(
            #             float(defense_diag["defense/value_std"].detach().cpu())
            #         )
            #         self.defense_stats["smooth_penalty"].append(
            #             float(defense_diag["defense/smooth_penalty"].detach().cpu())
            #         )
            #         self.defense_stats["robust_score"].append(
            #             float(defense_diag["defense/robust_score"].detach().cpu())
            #         )
            # else:
            #     score_value = value

            defense_enabled = bool(
                getattr(
                    self.cfg,
                    "defense_robust_mppi",
                    False,
                )
            )

            defense_mode = getattr(
                self.cfg,
                "defense_mode",
                "temporal",
            )

            if defense_enabled and defense_mode == "temporal":
                score_value, defense_diag = self._robust_planning_score(
                    z=z,
                    actions=actions,
                    task=task,
                    base_value=None,
                    poison_fn=poison_fn,
                )

                if bool(
                    getattr(
                        self.cfg,
                        "defense_log_stats",
                        False,
                    )
                ):
                    self.defense_stats["value_std"].append(
                        float(
                            defense_diag[
                                "defense/value_std"
                            ].detach().cpu()
                        )
                    )

                    self.defense_stats["smooth_penalty"].append(
                        float(
                            defense_diag[
                                "defense/smooth_penalty"
                            ].detach().cpu()
                        )
                    )

                    self.defense_stats["robust_score"].append(
                        float(
                            defense_diag[
                                "defense/robust_score"
                            ].detach().cpu()
                        )
                    )
            else:
                score_value = value

            # Select elites according to defended score.
            elite_idxs = torch.topk(
                score_value.squeeze(1),
                self.cfg.num_elites,
                dim=0,
            ).indices

            elite_value = score_value[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            # Update MPPI distribution.
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)

            mean = (
                score.unsqueeze(0) * elite_actions
            ).sum(dim=1) / (score.sum(0) + 1e-9)

            std = (
                (
                    score.unsqueeze(0)
                    * (elite_actions - mean.unsqueeze(1)) ** 2
                ).sum(dim=1) / (score.sum(0) + 1e-9)
            ).sqrt()

            std = std.clamp(self.cfg.min_std, self.cfg.max_std)

            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # ------------------------------------------------------------
        # Select action
        # ------------------------------------------------------------
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)

        a, std = actions[0], std[0]

        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)

        self._prev_mean.copy_(mean)

        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)

        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(
            self.cfg.rho,
            torch.arange(len(qs), device=self.device),
        )

        pi_loss = (
            -(
                self.cfg.entropy_coef * info["scaled_entropy"] + qs
            ).mean(dim=(1, 2)) * rho
        ).mean()

        pi_loss.backward()

        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
        )

        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })

        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task):
        """
        Compute TD target.
        """
        action, _ = self.model.pi(next_z, task)

        discount = (
            self.discount[task].unsqueeze(-1)
            if self.cfg.multitask else self.discount
        )

        return reward + discount * (1 - terminated) * self.model.Q(
            next_z,
            action,
            task,
            return_type='min',
            target=True,
        )

    def _update(self, obs, action, reward, terminated, task=None):
        """
        Main gradient update.
        """
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )

        z = self.model.encode(obs[0], task)
        zs[0] = z

        consistency_loss = 0

        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = (
                consistency_loss
                + F.mse_loss(z, _next_z) * self.cfg.rho ** t
            )
            zs[t + 1] = z

        _zs = zs[:-1]

        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)

        if self.cfg.episodic:
            termination_pred = self.model.termination(
                zs[1:],
                task,
                unnormalized=True,
            )

        reward_loss, value_loss = 0, 0

        for t, (
            rew_pred_unbind,
            rew_unbind,
            td_targets_unbind,
            qs_unbind,
        ) in enumerate(zip(
            reward_preds.unbind(0),
            reward.unbind(0),
            td_targets.unbind(0),
            qs.unbind(1),
        )):
            reward_loss = (
                reward_loss
                + math.soft_ce(
                    rew_pred_unbind,
                    rew_unbind,
                    self.cfg,
                ).mean() * self.cfg.rho ** t
            )

            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = (
                    value_loss
                    + math.soft_ce(
                        qs_unbind_unbind,
                        td_targets_unbind,
                        self.cfg,
                    ).mean() * self.cfg.rho ** t
                )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon

        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_pred,
                terminated,
            )
        else:
            termination_loss = 0.

        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
        )

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip_norm,
        )

        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        pi_info = self.update_pi(zs.detach(), task)

        self.model.soft_update_target_Q()

        self.model.eval()

        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })

        if self.cfg.episodic:
            info.update(
                math.termination_statistics(
                    torch.sigmoid(termination_pred[-1]),
                    terminated[-1],
                )
            )

        info.update(pi_info)

        return info.detach().mean()

    def update(self, buffer):
        """
        Main update function.
        """
        obs, action, reward, terminated, task = buffer.sample()

        kwargs = {}
        if task is not None:
            kwargs["task"] = task

        torch.compiler.cudagraph_mark_step_begin()

        return self._update(
            obs,
            action,
            reward,
            terminated,
            **kwargs,
        )

    @torch.no_grad()
    def _estimate_value_temporal_noise(
        self,
        z,
        actions,
        task,
        noise_seq,
        poison_fn=None,
    ):
        """
        Estimate trajectory value under temporal latent perturbations.

        At every imagined rollout step:

            z_eval = SimNorm(z + noise_t)
            reward = R(z_eval, a_t)
            z_next = d(z_eval, a_t)

        Thus, the perturbation affects both the current reward prediction
        and all subsequent imagined dynamics.

        Args:
            z:
                Initial latent state, shape [N, latent_dim].

            actions:
                Candidate action sequences, shape [H, N, action_dim].

            task:
                Task tensor or None.

            noise_seq:
                Temporal latent perturbations.

                Recommended shape:
                    [H + 1, 1, latent_dim]

                The second dimension is 1 intentionally. The same perturbation
                sample is shared by all candidate trajectories in one robustness
                evaluation, which reduces ranking noise between candidates.

            poison_fn:
                Optional white-box poison function. This is normally disabled
                for defense scoring. If enabled, poison and temporal noise are
                applied sequentially.

        Returns:
            value:
                Robustness-evaluation value, shape [N, 1].
        """
        N = z.shape[0]
        H = self.cfg.horizon

        if noise_seq.ndim == 2:
            # [H + 1, latent_dim] -> [H + 1, 1, latent_dim]
            noise_seq = noise_seq.unsqueeze(1)

        assert noise_seq.shape[0] >= H + 1, (
            f"noise_seq must have at least H+1 entries, "
            f"got {noise_seq.shape[0]}, expected {H + 1}"
        )

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
            a_t = actions[t]

            # ---------------------------------------------------------
            # 1. Temporal latent perturbation
            # ---------------------------------------------------------
            z_eval = self.model.apply_latent_bias(
                z,
                noise_seq[t],
            )

            # ---------------------------------------------------------
            # 2. Optional attack perturbation
            # ---------------------------------------------------------
            if poison_fn is not None:
                t_norm = float(t) / max(H - 1, 1)

                delta_t, gate_t = poison_fn(
                    z_eval,
                    a_t,
                    t_norm,
                )

                z_eval = self.model.apply_latent_bias(
                    z_eval,
                    gate_t * delta_t,
                )

            # ---------------------------------------------------------
            # 3. Reward prediction from perturbed latent
            # ---------------------------------------------------------
            reward = math.two_hot_inv(
                self.model.reward(
                    z_eval,
                    a_t,
                    task,
                ),
                self.cfg,
            )

            G = G + discount * (1.0 - termination) * reward

            # ---------------------------------------------------------
            # 4. Perturbed latent is used as dynamics input
            # ---------------------------------------------------------
            z = self.model.next(
                z_eval,
                a_t,
                task,
            )

            discount_update = self._get_discount_update_for_planning(task)
            discount = discount * discount_update

            if self.cfg.episodic:
                termination_pred = self.model.termination(
                    z,
                    task,
                )

                termination = torch.clamp(
                    termination
                    + (termination_pred > 0.5).float(),
                    max=1.0,
                )

        # -------------------------------------------------------------
        # Terminal-state temporal perturbation
        # -------------------------------------------------------------
        z_terminal = self.model.apply_latent_bias(
            z,
            noise_seq[H],
        )

        if poison_fn is not None:
            terminal_action, terminal_info = self.model.pi(
                z_terminal,
                task,
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

        terminal_action, terminal_info = self.model.pi(
            z_terminal,
            task,
        )

        terminal_action = terminal_info["mean"]

        terminal_q = self.model.Q(
            z_terminal,
            terminal_action,
            task,
            return_type="avg",
        )

        return G + discount * (1.0 - termination) * terminal_q