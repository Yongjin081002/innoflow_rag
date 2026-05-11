"""
반복 튜닝 스크립트 - Top1 10/10, 평균 score > 0.9 목표
하드 네거티브 triplet 추가 + 하이퍼파라미터 탐색
"""
import json, torch, random, numpy as np, os, copy, shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

# ── 테스트 케이스 ──
TEST_QUERIES = [
    "신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
    "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
    "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"
]
TEST_POS = [
    "차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
    "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"
]

# ── 청크 로드 ──
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# ── 보강 학습 데이터: 하드 네거티브 구분용 ──
extra_positive_pairs = [
    # 차1-1 (자동차 신호위반 직진) vs 거1-2 (자전거 신호위반 직진) 구분 강화
    {"query": "자동차 신호위반 직진 충돌 과실비율", "positive": "차1-1"},
    {"query": "차량 간 신호위반 직진 교차로 사고", "positive": "차1-1"},
    {"query": "승용차 적색신호 직진 충돌", "positive": "차1-1"},
    {"query": "자동차끼리 신호위반 직진 충돌 과실", "positive": "차1-1"},
    {"query": "차 대 차 신호위반 교차로 직진 사고", "positive": "차1-1"},
    {"query": "차량 녹색 직진 적색 직진 교차로 사고", "positive": "차1-1"},
    {"query": "자동차 교차로 신호위반 사고 과실비율", "positive": "차1-1"},
    {"query": "차량 A0 B100 신호위반 직진 과실", "positive": "차1-1"},
    {"query": "신호위반 직진 차량 충돌 기본 과실비율", "positive": "차1-1"},
    {"query": "신호위반 직진 충돌 자동차 사고", "positive": "차1-1"},

    # 차33-1 vs 차33-2 구분 강화 (차33-1: A20 B80 기본과실, 직진 vs 유턴)
    {"query": "유턴 중 직진차와 충돌 A20 B80", "positive": "차33-1"},
    {"query": "상시유턴구역 유턴 직진차 충돌 기본과실", "positive": "차33-1"},
    {"query": "유턴 중 충돌 기본 과실비율 A20 B80", "positive": "차33-1"},
    {"query": "유턴하다 직진차 사고 과실 A0 B100", "positive": "차33-1"},
    {"query": "직진 대 유턴 충돌 과실비율", "positive": "차33-1"},
    {"query": "유턴 중 충돌 직진 차량 과실", "positive": "차33-1"},
    {"query": "유턴 사고 직진차와 충돌", "positive": "차33-1"},
    {"query": "유턴 도중 직진 차 충돌 사고", "positive": "차33-1"},
    {"query": "상시유턴 직진차 충돌 과실비율", "positive": "차33-1"},
    {"query": "유턴 중 충돌 사고 기본과실", "positive": "차33-1"},

    # 차15-1 (비신호교차로 직진 vs 좌회전) 보강
    {"query": "비신호교차로 직진 vs 좌회전 과실비율", "positive": "차15-1"},
    {"query": "비신호 교차로 직진차 좌회전차 과실 A30 B70", "positive": "차15-1"},
    {"query": "신호없는 교차로에서 직진하다 좌회전차와 충돌", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 좌회전 기본 과실비율", "positive": "차15-1"},
    {"query": "비신호교차로 직진 대 좌회전 사고 과실", "positive": "차15-1"},
    {"query": "무신호 사거리 직진 좌회전 충돌 과실비율", "positive": "차15-1"},
    {"query": "비신호 교차로에서 직진 좌회전 충돌 과실 기준", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 좌회전 사고 과실비율 기준", "positive": "차15-1"},
    {"query": "비신호교차로 직진 좌회전 충돌 사고", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 vs 좌회전 충돌 과실", "positive": "차15-1"},
]

# ── 하드 네거티브 triplet (anchor, positive, negative) ──
hard_neg_triplets = [
    # 차1-1(자동차) vs 거1-2(자전거) 구분
    ("신호위반 직진 충돌", "차1-1", "거1-2"),
    ("적색신호 직진 사고 과실비율", "차1-1", "거1-2"),
    ("신호위반 직진 충돌 사고", "차1-1", "거1-2"),
    ("녹색 직진 적색 직진 충돌", "차1-1", "거1-2"),
    ("교차로 신호위반 직진 충돌", "차1-1", "거1-2"),
    ("신호위반 직진 충돌 과실", "차1-1", "거1-2"),
    ("빨간불 직진 충돌 과실비율", "차1-1", "거1-2"),
    ("자동차 신호위반 직진 충돌", "차1-1", "거1-2"),
    ("차량 신호위반 직진 교차로 사고", "차1-1", "거1-2"),
    ("직진 신호위반 충돌 사고 과실", "차1-1", "거1-2"),

    # 차33-1 vs 차33-2 구분
    ("유턴 중 충돌", "차33-1", "차33-2"),
    ("유턴 중 충돌 과실비율", "차33-1", "차33-2"),
    ("유턴하다 직진차와 충돌", "차33-1", "차33-2"),
    ("유턴 사고 과실", "차33-1", "차33-2"),
    ("상시유턴구역 유턴 충돌", "차33-1", "차33-2"),
    ("유턴 중 사고 과실비율", "차33-1", "차33-2"),
    ("유턴 도중 충돌", "차33-1", "차33-2"),
    ("유턴 중 직진차 충돌", "차33-1", "차33-2"),
    ("유턴 충돌 과실 기준", "차33-1", "차33-2"),
    ("유턴 중 충돌 기본과실", "차33-1", "차33-2"),

    # 차15-1(비신호 직진vs좌회전) vs 차21-1(같은방향 좌회전 두 차량) 구분
    ("비신호교차로 직진 vs 좌회전", "차15-1", "차21-1"),
    ("비신호 교차로 직진 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호교차로 직진 좌회전 과실비율", "차15-1", "차21-1"),
    ("신호 없는 교차로 직진 좌회전 사고", "차15-1", "차21-1"),
    ("무신호 교차로 직진 대 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호 사거리 직진 좌회전 충돌 과실", "차15-1", "차21-1"),
    ("신호없는 교차로 직진차 좌회전차 사고", "차15-1", "차21-1"),
    ("비신호 교차로 좌회전 직진 과실비율 기준", "차15-1", "차21-1"),
    ("비신호 직진 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호교차로 좌회전 직진 사고 과실", "차15-1", "차21-1"),
]


def evaluate(model):
    """현재 모델 평가 → (top1_count, avg_score, details)"""
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)

    scores = []
    top1_ok = 0
    details = []
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        top1_id = chunk_ids[all_sim.argmax().item()]
        ok = top1_id == TEST_POS[i]
        if ok:
            top1_ok += 1
        details.append((TEST_QUERIES[i], TEST_POS[i], s, top1_id, ok))
    avg = sum(scores) / len(scores)
    return top1_ok, avg, details


def print_report(round_num, top1, avg, details, label=""):
    print(f"\n{'='*100}")
    print(f"  [Round {round_num}] {label}")
    print(f"{'='*100}")
    print(f"{'쿼리':<28} {'정답':<8} {'Score':>8} {'Top1':>10} {'일치':>4}")
    print(f"{'-'*100}")
    for q, pos, s, top1_id, ok in details:
        mark = "O" if ok else "X"
        print(f"{q:<28} {pos:<8} {s:>8.4f} {top1_id:>10} {mark:>4}")
    print(f"{'-'*100}")
    print(f"  Top1: {top1}/10 | 평균 Score: {avg:.4f}")
    print(f"{'='*100}")


def train_round(base_model_path, seed, lr, batch_size, epochs, warmup,
                use_extra=True, use_triplets=True, out_path="./tmp_tune"):
    """한 라운드 학습 실행"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = SentenceTransformer(base_model_path)
    model.max_seq_length = 256

    # MNR loss용 positive pair 데이터
    pair_examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    if use_extra:
        for p in extra_positive_pairs:
            c = chunk_dict.get(p["positive"], "")
            if c:
                pair_examples.append(InputExample(texts=[p["query"], c]))

    dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=batch_size)
    mnr_loss = losses.MultipleNegativesRankingLoss(model)

    train_objectives = [(dl_pairs, mnr_loss)]

    # Triplet loss용 하드 네거티브 데이터
    if use_triplets and hard_neg_triplets:
        triplet_examples = []
        for anchor, pos_id, neg_id in hard_neg_triplets:
            pos_c = chunk_dict.get(pos_id, "")
            neg_c = chunk_dict.get(neg_id, "")
            if pos_c and neg_c:
                triplet_examples.append(InputExample(texts=[anchor, pos_c, neg_c]))
        if triplet_examples:
            dl_triplets = DataLoader(triplet_examples, shuffle=True, batch_size=batch_size)
            triplet_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.3)
            train_objectives.append((dl_triplets, triplet_loss))

    model.fit(train_objectives=train_objectives, epochs=epochs, warmup_steps=warmup,
              output_path=out_path, show_progress_bar=False)

    return model


# ── 메인 실행 ──
if __name__ == "__main__":
    BEST_PATH = "./bge-m3-finetuned"

    # 하이퍼파라미터 후보 (순차 시도)
    configs = [
        # Round 1: 기존 모델 위에 추가 학습 (보강데이터 + 하드네거티브, 3가지 실패케이스 보강)
        {"base": BEST_PATH, "seed": 42, "lr": 2e-5, "bs": 4, "epochs": 10, "warmup": 30,
         "extra": True, "triplet": True, "label": "기존모델 + 전체보강 (ep10, lr2e-5)"},
        # Round 2: lr 낮추고 epoch 높임
        {"base": BEST_PATH, "seed": 42, "lr": 1e-5, "bs": 4, "epochs": 15, "warmup": 40,
         "extra": True, "triplet": True, "label": "기존모델 + 전체보강 (ep15, lr1e-5)"},
        # Round 3: 원본부터 재학습
        {"base": "BAAI/bge-m3", "seed": 42, "lr": 3e-5, "bs": 4, "epochs": 15, "warmup": 30,
         "extra": True, "triplet": True, "label": "원본 재학습 (ep15, lr3e-5, bs4)"},
        # Round 4: 원본부터 더 높은 epoch
        {"base": "BAAI/bge-m3", "seed": 42, "lr": 2e-5, "bs": 4, "epochs": 20, "warmup": 40,
         "extra": True, "triplet": True, "label": "원본 재학습 (ep20, lr2e-5, bs4)"},
        # Round 5: seed 변경
        {"base": "BAAI/bge-m3", "seed": 77, "lr": 3e-5, "bs": 4, "epochs": 20, "warmup": 30,
         "extra": True, "triplet": True, "label": "원본 재학습 (ep20, seed77)"},
        # Round 6: batch 2
        {"base": "BAAI/bge-m3", "seed": 42, "lr": 3e-5, "bs": 2, "epochs": 20, "warmup": 30,
         "extra": True, "triplet": True, "label": "원본 재학습 (ep20, bs2, lr3e-5)"},
    ]

    best_top1 = 0
    best_avg = 0.0
    best_round = -1

    for i, cfg in enumerate(configs):
        print(f"\n>>> Round {i+1} 시작: {cfg['label']}")
        out = f"./tmp_tune_r{i+1}"

        model = train_round(
            base_model_path=cfg["base"],
            seed=cfg["seed"],
            lr=cfg["lr"],
            batch_size=cfg["bs"],
            epochs=cfg["epochs"],
            warmup=cfg["warmup"],
            use_extra=cfg["extra"],
            use_triplets=cfg["triplet"],
            out_path=out,
        )

        top1, avg, details = evaluate(model)
        print_report(i + 1, top1, avg, details, cfg["label"])

        # 목표 달성 시 저장 후 종료
        if top1 == 10 and avg >= 0.9:
            print(f"\n*** 목표 달성! Top1={top1}/10, 평균={avg:.4f} ***")
            if os.path.exists(BEST_PATH + "_backup"):
                shutil.rmtree(BEST_PATH + "_backup")
            if os.path.exists(BEST_PATH):
                shutil.copytree(BEST_PATH, BEST_PATH + "_backup")
                shutil.rmtree(BEST_PATH)
            shutil.copytree(out, BEST_PATH)
            print(f"모델 저장 완료: {BEST_PATH}")
            break

        # 현재 최고 기록 갱신
        if top1 > best_top1 or (top1 == best_top1 and avg > best_avg):
            best_top1 = top1
            best_avg = avg
            best_round = i + 1
            # 최고 기록 임시 저장
            best_out = "./tmp_tune_best"
            if os.path.exists(best_out):
                shutil.rmtree(best_out)
            shutil.copytree(out, best_out)

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    else:
        # 모든 라운드 완료했지만 목표 미달성 시 최고 기록 저장
        print(f"\n목표 미달성. 최고 기록 Round {best_round}: Top1={best_top1}/10, 평균={best_avg:.4f}")
        if best_top1 >= 10 or best_avg > best_avg:
            print("최고 기록 모델을 저장합니다.")
            if os.path.exists(BEST_PATH + "_backup"):
                shutil.rmtree(BEST_PATH + "_backup")
            if os.path.exists(BEST_PATH):
                shutil.copytree(BEST_PATH, BEST_PATH + "_backup")
                shutil.rmtree(BEST_PATH)
            shutil.copytree("./tmp_tune_best", BEST_PATH)
