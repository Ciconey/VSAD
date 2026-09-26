export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

export MUJOCO_PY_FORCE_CPU=1
export MUJOCO_GL=osmesa
# MuJoCo 2.1.0 路径
export MUJOCO_PY_MUJOCO_PATH=./.mujoco/mujoco210
 
# MuJoCo 动态库路径
export LD_LIBRARY_PATH=./.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
 
unset DISPLAY

# CONDA_PATH=$(conda info --base)
# source "$CONDA_PATH/etc/profile.d/conda.sh"

# conda activate  tdmpc2_310

python VSAD/WD/evaluate_defense.py\
    task=mt80 \
    model_size=48 \
    checkpoint=./tdmpc2/weight/mt80-48M.pt \
    work_dir=./tdmpc2/log \
    eval_episodes=50 \
    save_video=false \
    +defense_robust_mppi=true \
    +defense_mode=temporal \
    +defense_start_fraction=0.5 \
    +defense_num_perturb=4 \
    +defense_latent_noise=0.030 \
    +defense_beta_sens=2.0 \
    +defense_lambda_smooth=0.05 \
    +defense_use_poison_in_score=false \
    +defense_log_stats=true \
    compile=false \
    exp_name=WD_result


  
	
