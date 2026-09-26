export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

export MUJOCO_PY_FORCE_CPU=1
export MUJOCO_GL=osmesa

export MUJOCO_PY_MUJOCO_PATH=./.mujoco/mujoco210
 

export LD_LIBRARY_PATH=./.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
 
unset DISPLAY


python ./tdmpc2/tdmpc2/WA/evaluate_attack7.py\
    task=mt80 \
    model_size=48 \
    checkpoint=./tdmpc2/weight/mt80-48M.pt \
    work_dir=./tdmpc2/log \
    eval_episodes=50 \
    attack.steps=3000 \
    attack.buffer_size=512 \
    attack.batch_size=64 \
    attack.eps=0.30 \
    attack.reward_threshold=null \
    attack.warmup_steps=10000 \
    attack.top_percentile=70 \
    attack.margin=0.25 \
    attack.alpha_reward=0.15 \
    attack.gamma_delta=1e-4 \
    attack.eta_gate=1e-3 \
    attack.lr=1e-3 \
    save_video=false \
    exp_name=WA_result