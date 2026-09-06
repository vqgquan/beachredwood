#!/bin/bash
# Chạy lại seed_check hoàn toàn mới: giữ file seed_check.csv cũ làm backup,
# train lại seeds 1 2, và tự gộp seed 0 từ ablation_results.csv (lần chạy local mới).
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Chưa có .venv -- chạy ./run_part3_local.sh trước." >&2
  exit 1
fi
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy

OLD=part3/results/seed_check.csv
if [ -f "$OLD" ]; then
  BAK="part3/results/seed_check_backup_$(date +%Y%m%d_%H%M%S).csv"
  mv "$OLD" "$BAK"
  echo "Đã backup $OLD -> $BAK"
fi

LOG=part3/seedcheck_fresh.log
echo "=== $(date) seed_check fresh: seeds 1 2 ===" | tee "$LOG"
python -u part3/seed_check.py --seeds 1 2 "$@" 2>&1 | tee -a "$LOG"
echo "=== $(date) done ===" | tee -a "$LOG"
