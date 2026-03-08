#!/bin/bash
# Monitor running Nature critique experiments

echo "========================================"
echo "DKO Nature Critique Experiment Monitor"
echo "========================================"
echo "Time: $(date)"
echo ""

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -10

echo ""
echo "=== Running Processes ==="
ps aux | grep -E "run_10seed|run_hybrid|mechanistic" | grep python | grep -v grep | while read line; do
  pid=$(echo $line | awk '{print $2}')
  cpu=$(echo $line | awk '{print $3}')
  script=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
  echo "  PID $pid (CPU: $cpu%): $script"
done

echo ""
echo "=== Checkpoint Counts ==="
echo "Neural (GPU 5 - esol/lipo/qm9): $(find results/10seed_validation/checkpoints -name 'best_model.pt' 2>/dev/null | wc -l) / 90"
echo "Neural (GPU 3 - freesolv): $(find results/10seed_validation_freesolv/checkpoints -name 'best_model.pt' 2>/dev/null | wc -l) / 30"

echo ""
echo "=== Result Files ==="
if [ -f results/hybrid_10seed/results_partial.json ]; then
  n=$(python3 -c "import json; print(len(json.load(open('results/hybrid_10seed/results_partial.json'))))" 2>/dev/null || echo "0")
  echo "Hybrid 10-seed: $n / 280 experiments"
else
  echo "Hybrid 10-seed: No results yet"
fi

if [ -f results/mechanistic_analysis/mechanistic_analysis.json ]; then
  size=$(wc -c < results/mechanistic_analysis/mechanistic_analysis.json)
  echo "Mechanistic analysis: $size bytes"
else
  echo "Mechanistic analysis: Not complete"
fi

echo ""
echo "=== Log Files (last 5 lines) ==="
for log in logs/10seed_validation_gpu5.log logs/10seed_validation_gpu3.log logs/hybrid_10seed.log logs/mechanistic_analysis.log; do
  if [ -s "$log" ]; then
    echo "--- $log ---"
    tail -5 "$log" 2>/dev/null | grep -v "DEPRECATION"
  fi
done
