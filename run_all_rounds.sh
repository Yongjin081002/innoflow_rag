#!/bin/bash
set -e
cd /home/minsung0830/innoflow_rag
export CUDA_VISIBLE_DEVICES=1

BEST_PATH="./bge-m3-finetuned"

run_round() {
    local round=$1 base=$2 seed=$3 lr=$4 bs=$5 epochs=$6 warmup=$7 label=$8
    local out="./tmp_tune_r${round}"

    echo ""
    echo ">>> Round ${round} 시작: ${label}"
    python3 run_single_round.py \
        --base "$base" --seed "$seed" --lr "$lr" --bs "$bs" \
        --epochs "$epochs" --warmup "$warmup" --out "$out" \
        --round "$round" --label "$label" 2>&1

    # 결과 확인
    if [ -f "${out}_result.json" ]; then
        top1=$(python3 -c "import json; d=json.load(open('${out}_result.json')); print(d['top1'])")
        avg=$(python3 -c "import json; d=json.load(open('${out}_result.json')); print(d['avg'])")

        if [ "$top1" = "10" ] && python3 -c "exit(0 if $avg >= 0.9 else 1)"; then
            echo ""
            echo "*** 목표 달성! Top1=${top1}/10, 평균=${avg} ***"
            if [ -d "${BEST_PATH}_backup" ]; then rm -rf "${BEST_PATH}_backup"; fi
            cp -r "$BEST_PATH" "${BEST_PATH}_backup"
            rm -rf "$BEST_PATH"
            cp -r "$out" "$BEST_PATH"
            echo "모델 저장 완료: ${BEST_PATH}"
            exit 0
        fi
    fi
}

# Round 1: 기존모델 + 보강 + 하드네거티브
run_round 1 "$BEST_PATH" 42 2e-5 4 10 30 "기존모델+보강 (ep10,lr2e-5)"

# Round 2: 기존모델 + epoch 증가
run_round 2 "$BEST_PATH" 42 1e-5 4 15 40 "기존모델+보강 (ep15,lr1e-5)"

# Round 3: 원본부터 재학습
run_round 3 "BAAI/bge-m3" 42 3e-5 4 15 30 "원본재학습 (ep15,lr3e-5,bs4)"

# Round 4: 원본부터 epoch 20
run_round 4 "BAAI/bge-m3" 42 2e-5 4 20 40 "원본재학습 (ep20,lr2e-5,bs4)"

# Round 5: seed 변경
run_round 5 "BAAI/bge-m3" 77 3e-5 4 20 30 "원본재학습 (ep20,seed77)"

# Round 6: batch 2
run_round 6 "BAAI/bge-m3" 42 3e-5 2 20 30 "원본재학습 (ep20,bs2,lr3e-5)"

# Round 7: 더 높은 epoch
run_round 7 "BAAI/bge-m3" 42 3e-5 4 25 40 "원본재학습 (ep25,lr3e-5,bs4)"

# Round 8: seed 123
run_round 8 "BAAI/bge-m3" 123 3e-5 4 20 30 "원본재학습 (ep20,seed123)"

echo ""
echo "모든 라운드 완료. 목표 미달성."
