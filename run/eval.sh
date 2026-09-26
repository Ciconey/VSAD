export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

export MUJOCO_PY_FORCE_CPU=1
export MUJOCO_GL=osmesa

export MUJOCO_PY_MUJOCO_PATH=./.mujoco/mujoco210
 

export LD_LIBRARY_PATH=./.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
 
unset DISPLAY



python ./tdmpc2/tdmpc2/evaluate.py \
 task=mt80 \
 model_size=48 \
 checkpoint=./tdmpc2/weight/nicklashansen/tdmpc2/multitask/mt80-48M.pt \
 work_dir=./tdmpc2/log \
 eval_episodes=50 \
 save_video=false \
 exp_name=orgin_50