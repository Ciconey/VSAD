import gym
import numpy as np
from envs.wrappers.timeout import Timeout
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False

		# 【新增1】初始化历史状态记录器，用于计算“变化率”
		self.prev_action = np.zeros(self.env.action_space.shape)
		self.prev_obs = None

	def reset(self, **kwargs):
		obs = super().reset(**kwargs).astype(np.float32)
		self.env.step(np.zeros(self.env.action_space.shape))
		# 【新增2】回合重置时，清空历史记录
		self.prev_action = np.zeros(self.env.action_space.shape)
		self.prev_obs = obs.copy()
		return obs

	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, _, info = self.env.step(action.copy())
			reward += r
		obs = obs.astype(np.float32)

		# 指标 A: 通用控制异常 —— 动作抖动 (Action Jitter)
		# 计算当前动作与上一步动作的 L2 距离 (欧氏距离)
		jitter = np.linalg.norm(action - self.prev_action)

		curr_pos = obs[:3]
		if self.prev_obs is not None:
			prev_pos = self.prev_obs[:3]
			impact = np.linalg.norm(curr_pos - prev_pos)
		else:
			impact = 0.0

		# 解析 Meta-World 的 Observation (观测向量)
		# Meta-World V2 的 obs 通常是 39 维向量：
		# obs[0:3] 是机械臂末端 (Gripper) 的 [X, Y, Z] 坐标
		# obs[4:7] 是桌面上主要目标物体 (Object) 的 [X, Y, Z] 坐标
		gripper_pos = obs[0:3]
		obj_pos = obs[4:7]

		info['safety/jitter'] = jitter
		info['safety/impact'] = impact

		# 【新增4】更新历史状态，供下一次 step 计算使用
		self.prev_action = action.copy()
		self.prev_obs = obs.copy()
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(
			offscreen=True, resolution=(384, 384), camera_name=self.camera_name
		).copy()

	def get_env_state(self):
		if hasattr(self.env, "get_env_state"):
			return self.env.get_env_state()
		if hasattr(self.env, "sim") and hasattr(self.env.sim, "get_state"):
			return self.env.sim.get_state()
		raise RuntimeError("Underlying Meta-World environment has no state API")

	def set_env_state(self, state):
		if hasattr(self.env, "set_env_state"):
			self.env.set_env_state(state)
			return
		if hasattr(self.env, "sim") and hasattr(self.env.sim, "set_state"):
			self.env.sim.set_state(state)
			self.env.sim.forward()
			return
		raise RuntimeError("Underlying Meta-World environment has no state API")


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
	env = MetaWorldWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	# env.max_episode_steps = env._max_episode_steps
	return env
