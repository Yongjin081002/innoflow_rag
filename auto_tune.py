import os
import sys
import shutil
import subprocess

seeds = [0, 1, 2, 3, 5, 7, 10, 13, 17, 21, 42, 77, 99, 123, 256, 512, 777, 1004, 2024, 3090]

best_seed = None
best_avg = 0
best_top1 = 0

for seed in seeds:
    result = subprocess.run(
        [sys.executable, "train_seed.py", str(seed)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )

    output = result.stdout + result.stderr
    top1 = -1
    avg = -1.0
    for line in output.split("\n"):
        if line.startswith("RESULT:"):
            parts = line.replace("RESULT:", "").split(",")
            top1 = int(parts[0])
            avg = float(parts[1])

    if top1 < 0:
        print(f"seed={seed:>4d}  FAILED")
        if os.path.exists(f"./tmp_model_{seed}"):
            shutil.rmtree(f"./tmp_model_{seed}")
        continue

    is_best = top1 > best_top1 or (top1 == best_top1 and avg > best_avg)
    marker = " ★ BEST" if is_best else ""
    print(f"seed={seed:>4d}  Top1={top1}/10  avg={avg:.4f}{marker}")

    if is_best:
        best_seed = seed
        best_top1 = top1
        best_avg = avg
        if os.path.exists("./bge-m3-finetuned"):
            shutil.rmtree("./bge-m3-finetuned")
        shutil.copytree(f"./tmp_model_{seed}", "./bge-m3-finetuned")

    if os.path.exists(f"./tmp_model_{seed}"):
        shutil.rmtree(f"./tmp_model_{seed}")

print(f"\n{'='*60}")
print(f"최적 시드: {best_seed}")
print(f"Top1: {best_top1}/10, 평균 score: {best_avg:.4f}")
print(f"모델 저장: ./bge-m3-finetuned")
