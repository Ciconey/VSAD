export CUDA_VISIBLE_DEVICES=6
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

# ================= 变量配置区（以后只需修改这几项） =================
EXP_NAME="BA_48_0916_nes_50"
BASE_DIR="./tdmpc2"
LOG_DIR="${BASE_DIR}/log/${EXP_NAME}"

# 自动创建对应实验的日志文件夹（避免 tee 报错找不到目录）
mkdir -p "${LOG_DIR}"

# ================= 运行脚本 =================
python -u  VSAD/BB/blackbox_attack.py \
    task=mt80 \
    model_size=48 \
    checkpoint="${BASE_DIR}/weight/mt80-48M.pt" \
    work_dir="${BASE_DIR}/log" \
    exp_name="${EXP_NAME}" \
    save_video=false \
    +blackbox_use_defense=false \
    +defense_mode=observation \
    +defense_num_obs_perturb=4 \
    +defense_obs_noise=0.005 \
    +defense_beta_obs=2.0 \
    +defense_lambda_smooth=0.05 \
    +attack_objective=targeted_bad \
    +attack_eps=0.1 \
    +attack_alpha=0.015 \
    +attack_nes_steps=5 \
    +attack_nes_samples=8 \
    +attack_nes_sigma=0.01 \
    +attack_every=1 \
    +vulnerable_reward_thresh=2.0 \
    +vulnerable_step_thresh=999 \
    +attack_episodes=50 \
    +clean_episodes=5 \
    +attack_run_clean=false \
    +attack_save_video=false \
    compile=false \
    2>&1 | tee "${LOG_DIR}/attack_$(date +%Y%m%d_%H%M%S).txt"




