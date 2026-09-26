import os
import time
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings

warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from termcolor import colored

from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):

	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
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

		if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
			print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
			print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

		# Make environment
		env = make_env(cfg)

		# Load agent
		agent = TDMPC2(cfg)
		assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
		agent.load(cfg.checkpoint)
		
		# Evaluate
		if cfg.multitask:
			print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		else:
			print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
		if cfg.save_video:
			video_dir = os.path.join(cfg.work_dir, 'videos')
			os.makedirs(video_dir, exist_ok=True)
		scores = []
		tasks = cfg.tasks if cfg.multitask else [cfg.task]

		

		for task_idx, task in enumerate(tasks):
			# if task != 'mw-assembly':
			if not task.startswith('mw-'):
				continue
			if not cfg.multitask:
				task_idx = None
			ep_rewards, ep_successes = [], []

			ep_safety_metrics = {} # 
			for i in range(cfg.eval_episodes):
				obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0

				current_ep_safety = {} 
				if cfg.save_video:
					frames = [env.render()]
				while not done:
					action = agent.act(obs, t0=t==0, task=task_idx)
					obs, reward, done, info = env.step(action)
					ep_reward += reward
					t += 1

					
					for k, v in info.items():
						if k.startswith('safety/'): 
							current_ep_safety[k] = current_ep_safety.get(k, 0.0) + v

					if cfg.save_video:
						frames.append(env.render())
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])

				jitter_val = current_ep_safety.get('safety/jitter', 0.0)
				impact_val = current_ep_safety.get('safety/impact', 0.0)


				log(
                    f"  [Episode {i}] Reward: {ep_reward:.1f} | "
                    f"Success: {info.get('success', 0.0)} | "
                    f"Jitter: {jitter_val:.2f} | "
                    f"Impact: {impact_val:.2f}"
                )
					

				for k, v in current_ep_safety.items():
					if k not in ep_safety_metrics:
						ep_safety_metrics[k] = []
					ep_safety_metrics[k].append(v)

				if cfg.save_video:
					imageio.mimsave(
						os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
			ep_rewards = np.mean(ep_rewards)
			ep_successes = np.mean(ep_successes)
			if cfg.multitask:
				scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)

			log(
                f"  {task:<22}\tR: {ep_rewards:.01f}\tS: {ep_successes:.02f}",
                "yellow"
            )

			log("\n=== Safety Metrics (Reward Hacking Analysis) ===")


			for k, v_list in ep_safety_metrics.items():

				log(f"{k}: {np.mean(v_list):.4f} +/- {np.std(v_list):.4f}")
			
	
			log("===================================\n")

		if cfg.multitask:
	
			log(f"Normalized score: {np.mean(scores):.02f}", "yellow", ["bold"])


if __name__ == '__main__':
	evaluate()
