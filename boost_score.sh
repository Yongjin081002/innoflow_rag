#!/bin/bash
# Round 2 결과(Top1=10/10, avg=0.84)를 기반으로 score를 0.9 이상 부스팅
set -e
cd /home/minsung0830/innoflow_rag
export CUDA_VISIBLE_DEVICES=1

# Round 2 결과 모델을 기반으로 사용
BASE="./tmp_tune_r2"
if [ ! -d "$BASE" ]; then
    echo "tmp_tune_r2 모델이 없습니다. 먼저 기본 라운드를 실행하세요."
    exit 1
fi

run_round() {
    local round=$1 base=$2 seed=$3 lr=$4 bs=$5 epochs=$6 warmup=$7 label=$8
    local out="./tmp_boost_r${round}"

    echo ""
    echo ">>> Boost Round ${round} 시작: ${label}"
    python3 run_single_round.py \
        --base "$base" --seed "$seed" --lr "$lr" --bs "$bs" \
        --epochs "$epochs" --warmup "$warmup" --out "$out" \
        --round "$round" --label "$label" 2>&1

    if [ -f "${out}_result.json" ]; then
        top1=$(python3 -c "import json; d=json.load(open('${out}_result.json')); print(d['top1'])")
        avg=$(python3 -c "import json; d=json.load(open('${out}_result.json')); print(d['avg'])")

        if [ "$top1" = "10" ] && python3 -c "exit(0 if $avg >= 0.9 else 1)"; then
            echo ""
            echo "*** 목표 달성! Top1=${top1}/10, 평균=${avg} ***"
            BEST="./bge-m3-finetuned"
            if [ -d "${BEST}_backup" ]; then rm -rf "${BEST}_backup"; fi
            cp -r "$BEST" "${BEST}_backup"
            rm -rf "$BEST"
            cp -r "$out" "$BEST"
            echo "모델 저장 완료: ${BEST}"
            exit 0
        fi
    fi
}

# 기존 모델(Round2=Top10/10)을 더 높은 epoch으로 추가 학습
# Score를 0.9 이상으로 올려야 함 - 더 공격적인 학습

run_round 1 "$BASE" 42 3e-5 2 20 20 "R2기반 추가학습 (ep20,lr3e-5,bs2)"
run_round 2 "$BASE" 42 5e-5 2 15 10 "R2기반 추가학습 (ep15,lr5e-5,bs2)"
run_round 3 "$BASE" 42 3e-5 2 30 20 "R2기반 추가학습 (ep30,lr3e-5,bs2)"
run_round 4 "$BASE" 42 5e-5 4 20 15 "R2기반 추가학습 (ep20,lr5e-5,bs4)"
run_round 5 "$BASE" 42 1e-4 2 10 10 "R2기반 추가학습 (ep10,lr1e-4,bs2)"
run_round 6 "$BASE" 123 3e-5 2 25 20 "R2기반 추가학습 (ep25,seed123)"
run_round 7 "$BASE" 42 5e-5 2 25 15 "R2기반 추가학습 (ep25,lr5e-5,bs2)"
run_round 8 "$BASE" 77 5e-5 2 20 15 "R2기반 추가학습 (ep20,lr5e-5,seed77)"

echo ""
echo "모든 부스트 라운드 완료. 목표 미달성."
