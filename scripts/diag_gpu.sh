#!/bin/bash
# diag_gpu.sh - Diagnose GPU usage for ollama models
# Run on H100/GH200 to check which models use CUDA

set -e

echo "=== GPU Diagnostic Script ==="
echo "Date: $(date)"
echo ""

# Check GPU
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
echo ""

# Check ollama
echo "=== Ollama Version ==="
ollama --version 2>/dev/null || echo "ollama not installed"
echo ""

# Start ollama if not running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting ollama..."
    ollama serve > /tmp/ollama_diag.log 2>&1 &
    sleep 5
fi

# Models to test - focus on the ones that failed
MODELS=(
    "phi4"
    "qwen2.5:14b"
    "mistral-nemo"
    "gemma3:12b"
    "deepseek-r1:7b"
)

echo "=== Testing Models ==="
echo ""

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Testing: $model"
    echo "----------------------------------------"

    # Try to pull
    echo "Pulling..."
    if ! ollama pull "$model" 2>&1 | tail -3; then
        echo "SKIP: Failed to pull"
        continue
    fi

    # Clear GPU memory
    sleep 2

    # Get baseline GPU usage
    GPU_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    # Run a simple inference with timeout
    echo "Running inference..."
    START=$(date +%s.%N)

    # Monitor GPU in background
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1 > /tmp/gpu_monitor_$$.csv 2>/dev/null &
    MONITOR_PID=$!

    # Run inference
    RESPONSE=$(timeout 60 ollama run "$model" "What is 2+2? Reply with just the number." 2>&1 | head -5)

    END=$(date +%s.%N)

    # Stop monitor
    kill $MONITOR_PID 2>/dev/null || true
    sleep 1

    # Get GPU usage after
    GPU_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    # Analyze GPU monitor output
    if [ -f /tmp/gpu_monitor_$$.csv ]; then
        MAX_GPU=$(cat /tmp/gpu_monitor_$$.csv | cut -d',' -f1 | grep -v "%" | sort -n | tail -1)
        MAX_MEM=$(cat /tmp/gpu_monitor_$$.csv | cut -d',' -f2 | sort -n | tail -1)
        rm /tmp/gpu_monitor_$$.csv
    else
        MAX_GPU="N/A"
        MAX_MEM="N/A"
    fi

    DURATION=$(echo "$END - $START" | bc)

    echo "Response: ${RESPONSE:0:100}..."
    echo "Duration: ${DURATION}s"
    echo "GPU memory before: ${GPU_BEFORE} MiB"
    echo "GPU memory after: ${GPU_AFTER} MiB"
    echo "Max GPU util during inference: ${MAX_GPU}%"
    echo "Max GPU memory during inference: ${MAX_MEM} MiB"

    # Determine if using GPU
    if [ "$GPU_AFTER" -gt "$((GPU_BEFORE + 100))" ] 2>/dev/null; then
        echo "STATUS: USING GPU (memory increased)"
    elif [ "${MAX_GPU:-0}" -gt 10 ] 2>/dev/null; then
        echo "STATUS: USING GPU (utilization detected)"
    else
        echo "STATUS: POSSIBLY CPU-ONLY (no GPU activity detected)"
    fi

    echo ""

    # Unload model to free memory
    ollama stop "$model" 2>/dev/null || true
    sleep 2
done

echo "=== CPU Check During Idle ==="
top -bn1 | head -20

echo ""
echo "=== Done ==="
