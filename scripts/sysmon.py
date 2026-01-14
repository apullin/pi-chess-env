#!/usr/bin/env python3
"""
Lightweight system monitor that outputs NDJSON.

Captures per-CPU utilization and GPU stats at 1-second intervals.
Designed to catch single-CPU bottlenecks and GPU utilization drops.

Usage:
    # Run in background, log to file
    python scripts/sysmon.py > monitor.jsonl &
    MON_PID=$!

    # Run your workload
    python scripts/benchmark_llm.py qwen2.5:0.5b

    # Stop monitoring
    kill $MON_PID

    # Analyze logs
    python scripts/sysmon.py --analyze monitor.jsonl

Dependencies:
    pip install psutil
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone


def get_cpu_stats():
    """Get per-CPU utilization using psutil."""
    try:
        import psutil
        # Per-CPU percentages (non-blocking after first call)
        per_cpu = psutil.cpu_percent(percpu=True)
        return {
            "cpu_percent": per_cpu,
            "cpu_avg": sum(per_cpu) / len(per_cpu) if per_cpu else 0,
            "cpu_max": max(per_cpu) if per_cpu else 0,
            "cpu_count": len(per_cpu),
        }
    except ImportError:
        return {"error": "psutil not installed"}


def get_gpu_stats():
    """Get GPU utilization using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return {"error": "nvidia-smi failed", "stderr": result.stderr}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "gpu_util": float(parts[1]),
                    "mem_util": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "mem_total_mb": float(parts[4]),
                    "temp_c": float(parts[5]),
                })

        return {
            "gpus": gpus,
            "gpu_util_avg": sum(g["gpu_util"] for g in gpus) / len(gpus) if gpus else 0,
        }
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout"}
    except Exception as e:
        return {"error": str(e)}


def monitor(interval: float = 1.0):
    """Main monitoring loop. Outputs NDJSON to stdout."""
    # Prime the CPU stats (first call returns 0)
    try:
        import psutil
        psutil.cpu_percent(percpu=True)
    except ImportError:
        pass

    time.sleep(0.1)  # Brief pause for accurate first reading

    while True:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "epoch": time.time(),
        }
        record.update(get_cpu_stats())
        record.update(get_gpu_stats())

        print(json.dumps(record), flush=True)
        time.sleep(interval)


def analyze(logfile: str):
    """Analyze monitoring logs and print summary."""
    records = []
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("No valid records found")
        return

    print(f"\n{'='*60}")
    print("MONITORING SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {records[-1]['epoch'] - records[0]['epoch']:.1f}s ({len(records)} samples)")

    # CPU analysis
    cpu_maxes = [r.get("cpu_max", 0) for r in records]
    cpu_avgs = [r.get("cpu_avg", 0) for r in records]

    print(f"\nCPU:")
    print(f"  Avg utilization: {sum(cpu_avgs)/len(cpu_avgs):.1f}%")
    print(f"  Peak single-core: {max(cpu_maxes):.1f}%")
    print(f"  Samples with core >95%: {sum(1 for m in cpu_maxes if m > 95)} ({100*sum(1 for m in cpu_maxes if m > 95)/len(cpu_maxes):.0f}%)")

    # GPU analysis
    gpu_utils = [r.get("gpu_util_avg", 0) for r in records if "gpu_util_avg" in r]
    if gpu_utils:
        print(f"\nGPU:")
        print(f"  Avg utilization: {sum(gpu_utils)/len(gpu_utils):.1f}%")
        print(f"  Min utilization: {min(gpu_utils):.1f}%")
        print(f"  Max utilization: {max(gpu_utils):.1f}%")
        print(f"  Samples with GPU <10%: {sum(1 for u in gpu_utils if u < 10)} ({100*sum(1 for u in gpu_utils if u < 10)/len(gpu_utils):.0f}%)")
        print(f"  Samples with GPU >90%: {sum(1 for u in gpu_utils if u > 90)} ({100*sum(1 for u in gpu_utils if u > 90)/len(gpu_utils):.0f}%)")

    # Find bottleneck windows (CPU maxed, GPU idle)
    bottleneck_windows = []
    for r in records:
        if r.get("cpu_max", 0) > 95 and r.get("gpu_util_avg", 100) < 20:
            bottleneck_windows.append(r["timestamp"])

    if bottleneck_windows:
        print(f"\nCPU BOTTLENECK DETECTED:")
        print(f"  {len(bottleneck_windows)} samples with CPU >95% and GPU <20%")
        print(f"  First occurrence: {bottleneck_windows[0]}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight system monitor (NDJSON output)")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--analyze", "-a", type=str, metavar="FILE", help="Analyze a log file instead of monitoring")
    args = parser.parse_args()

    if args.analyze:
        analyze(args.analyze)
    else:
        try:
            monitor(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
