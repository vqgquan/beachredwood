#!/bin/bash
# Chạy lại Part III (seed 0) trên máy local để đối chiếu với results/ trên GitHub.
# Kết quả gốc từ GitHub được sao lưu sang part3/results_github/ trước khi chạy.
set -e
cd "$(dirname "$0")"
LOG=part3/rerun_local.log
echo "=== $(date) start ===" | tee "$LOG"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r part2/requirements.txt
python -c "import stable_baselines3, torch; print('sb3', stable_baselines3.__version__, 'torch', torch.__version__, 'threads', torch.get_num_threads())" | tee -a "$LOG"

if [ ! -d part3/results_github ]; then
  cp -R part3/results part3/results_github
fi
rm -f part3/results/ablation_results.csv

export SDL_VIDEODRIVER=dummy
python -u part3/run_ablation.py --timesteps 700000 --n-envs 8 --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date) done ===" | tee -a "$LOG"
