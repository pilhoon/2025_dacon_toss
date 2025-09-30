#!/bin/bash
# Monitor XGBoost training progress

LOG_FILE="/home/km/work/2025_dacon_toss/plan4/007_output_v2.log"

while true; do
    clear
    echo "=================================="
    echo "XGBoost Training Monitor"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="

    # Check GPU usage
    echo -e "\n=== GPU Status ==="
    nvidia-smi --query-gpu=gpu_name,utilization.gpu,memory.used,memory.total --format=csv,noheader

    # Check process
    echo -e "\n=== Process Status ==="
    ps aux | grep "007_xgboost_optuna" | grep -v grep | head -1 | awk '{print "PID: "$2", CPU: "$3"%, MEM: "$4"% ("$6/1024/1024" GB)"}'

    # Check log file
    echo -e "\n=== Log File ==="
    if [ -f "$LOG_FILE" ]; then
        SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
        echo "Size: $SIZE"
        echo -e "\n--- Last 20 lines ---"
        tail -20 "$LOG_FILE"
    else
        echo "Log file not found yet..."
    fi

    sleep 30
done