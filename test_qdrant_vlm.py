"""
boost_best_vlm_v4 + Qdrant(embed_text) 기반 VLM 쿼리 테스트
- v3 baseline (content 전체 임베딩) vs v4 Qdrant (embed_text 임베딩) 비교
"""
import json, os, sys
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from qdrant_client import QdrantClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 데이터 로드 ──
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C
VLM_TEST_ITEMS = GROUP_A + GROUP_B + GROUP_C

TEST_QUERIES = [
    "신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
    "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
    "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"
]
TEST_POS = [
    "차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
    "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"
]

QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_fault_data")
COLLECTION = "fault_rules"


def eval_keyword(model, chunk_emb):
    query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)
    scores, top1_ok = [], 0
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        if chunk_ids[all_sim.argsort(descending=True)[0]] == TEST_POS[i]:
            top1_ok += 1
    return top1_ok, sum(scores) / len(scores)


def eval_vlm_content(model, chunk_emb):
    """기존 방식: content 전체 임베딩 기반 검색"""
    top1_ok, top3_ok = 0, 0
    scores = []
    group_results = {"A": [], "B": [], "C": []}
    details = []
    for item in VLM_TEST_ITEMS:
        q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        sims = cos_sim(q_emb, chunk_emb)[0]
        sorted_idx = sims.argsort(descending=True)
        top_ids = [chunk_ids[j] for j in sorted_idx[:5]]
        expected = item["expected"]
        exp_idx = chunk_ids.index(expected)
        s = sims[exp_idx].item()
        scores.append(s)
        rank = (top_ids.index(expected) + 1) if expected in top_ids else 999
        if rank == 1: top1_ok += 1
        if rank <= 3: top3_ok += 1
        group = item["name"][0]
        group_results[group].append({"rank": rank, "score": s})
        details.append({"name": item["name"], "expected": expected, "top1": top_ids[0], "rank": rank, "score": s})
    n = len(VLM_TEST_ITEMS)
    group_stats = {}
    for g, items in group_results.items():
        gt1 = sum(1 for x in items if x["rank"] == 1)
        gavg = sum(x["score"] for x in items) / len(items)
        group_stats[g] = {"top1": gt1, "total": 10, "avg": gavg}
    return {"top1": top1_ok, "top3": top3_ok, "total": n, "avg": sum(scores)/n, "group_stats": group_stats, "details": details}


def eval_vlm_qdrant(model, client):
    """새 방식: Qdrant embed_text 기반 검색"""
    top1_ok, top3_ok = 0, 0
    scores_list = []
    group_results = {"A": [], "B": [], "C": []}
    details = []

    # Qdrant에서 모든 포인트 가져와서 id 매핑
    all_pts, _ = client.scroll(collection_name=COLLECTION, limit=200, with_vectors=True)
    qdrant_ids = [pt.payload["id"] for pt in all_pts]
    qdrant_vecs = [pt.vector for pt in all_pts]

    import torch
    qdrant_emb = torch.tensor(qdrant_vecs, device="cpu")

    for item in VLM_TEST_ITEMS:
        q_emb = model.encode([item["query"]], convert_to_tensor=False, show_progress_bar=False)
        import torch as _t
        q_tensor = _t.tensor(q_emb, device="cpu")
        sims = cos_sim(q_tensor, qdrant_emb)[0]
        sorted_idx = sims.argsort(descending=True)
        top_ids = [qdrant_ids[j] for j in sorted_idx[:5]]
        expected = item["expected"]
        if expected in qdrant_ids:
            exp_idx = qdrant_ids.index(expected)
            s = sims[exp_idx].item()
        else:
            s = 0.0
        scores_list.append(s)
        rank = (top_ids.index(expected) + 1) if expected in top_ids else 999
        if rank == 1: top1_ok += 1
        if rank <= 3: top3_ok += 1
        group = item["name"][0]
        group_results[group].append({"rank": rank, "score": s})
        details.append({"name": item["name"], "expected": expected, "top1": top_ids[0], "rank": rank, "score": s})

    n = len(VLM_TEST_ITEMS)
    group_stats = {}
    for g, items in group_results.items():
        gt1 = sum(1 for x in items if x["rank"] == 1)
        gavg = sum(x["score"] for x in items) / len(items)
        group_stats[g] = {"top1": gt1, "total": 10, "avg": gavg}
    return {"top1": top1_ok, "top3": top3_ok, "total": n, "avg": sum(scores_list)/n, "group_stats": group_stats, "details": details}


def combined_score(vlm, kw_top1=None):
    vt1 = vlm["top1"]; vn = vlm["total"]; vavg = vlm["avg"]
    kw_f = (kw_top1 / 10) if kw_top1 is not None else 1.0
    base = (vt1/vn)*0.55 + vavg*0.35 + kw_f*0.10
    gs = vlm["group_stats"]
    a_r = gs["A"]["top1"]/gs["A"]["total"]
    c_r = gs["C"]["top1"]/gs["C"]["total"]
    return base - max(0, a_r - c_r)*0.10


# ══════════════════════════════════════════════════════════════
print("=" * 100)
print("  VLM 쿼리 테스트: v3(content) vs v4(embed_text/Qdrant)")
print("=" * 100)

# ── v3 baseline ──
print("\n[1] boost_best_vlm_v3 로딩 (content 전체 임베딩)...")
v3_model = SentenceTransformer(os.path.join(BASE_DIR, "boost_best_vlm_v3"))
v3_model.max_seq_length = 384
v3_chunk_emb = v3_model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
v3_kw_top1, v3_kw_avg = eval_keyword(v3_model, v3_chunk_emb)
v3_vlm = eval_vlm_content(v3_model, v3_chunk_emb)
del v3_model, v3_chunk_emb
import gc, torch; gc.collect(); torch.cuda.empty_cache()

# ── v4 content 방식 ──
print("[2] boost_best_vlm_v4 로딩 (content 전체 임베딩)...")
v4_model = SentenceTransformer(os.path.join(BASE_DIR, "boost_best_vlm_v4"))
v4_model.max_seq_length = 384
v4_chunk_emb = v4_model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
v4_kw_top1, v4_kw_avg = eval_keyword(v4_model, v4_chunk_emb)
v4_vlm_content = eval_vlm_content(v4_model, v4_chunk_emb)
del v4_chunk_emb; gc.collect(); torch.cuda.empty_cache()

# ── v4 Qdrant embed_text 방식 ──
print("[3] boost_best_vlm_v4 + Qdrant embed_text 검색...")
client = QdrantClient(path=QDRANT_PATH)
v4_vlm_qdrant = eval_vlm_qdrant(v4_model, client)
# keyword는 Qdrant 방식에서도 동일 모델 사용
v4q_kw_top1, v4q_kw_avg = v4_kw_top1, v4_kw_avg  # 키워드는 content 기반과 동일 모델
client.close()
del v4_model; gc.collect(); torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 비교표 출력
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  종합 비교표")
print(f"{'=' * 100}\n")

cw = 22
header = f"{'지표':<30} {'v3(content)':>{cw}} {'v4(content)':>{cw}} {'v4(embed_text)':>{cw}}"
print(header)
print("-" * len(header))

# 키워드
print(f"{'키워드 Top1 (/10)':<30} {v3_kw_top1:>{cw}} {v4_kw_top1:>{cw}} {v4q_kw_top1:>{cw}}")
print(f"{'키워드 Avg':<30} {v3_kw_avg:>{cw}.4f} {v4_kw_avg:>{cw}.4f} {v4q_kw_avg:>{cw}.4f}")
print("-" * len(header))

# VLM
for label, v3r, v4cr, v4qr in [
    ("VLM Top1 (/30)", v3_vlm["top1"], v4_vlm_content["top1"], v4_vlm_qdrant["top1"]),
    ("VLM Top3 (/30)", v3_vlm["top3"], v4_vlm_content["top3"], v4_vlm_qdrant["top3"]),
]:
    print(f"{label:<30} {v3r:>{cw}} {v4cr:>{cw}} {v4qr:>{cw}}")
print(f"{'VLM Avg':<30} {v3_vlm['avg']:>{cw}.4f} {v4_vlm_content['avg']:>{cw}.4f} {v4_vlm_qdrant['avg']:>{cw}.4f}")
print("-" * len(header))

# 그룹별
for g in ["A", "B", "C"]:
    g_label = {"A": "A (학습 多)", "B": "B (학습 少)", "C": "C (극소/복합)"}[g]
    v3g = v3_vlm["group_stats"][g]
    v4cg = v4_vlm_content["group_stats"][g]
    v4qg = v4_vlm_qdrant["group_stats"][g]
    print(f"{'그룹'+g_label+' Top1':<30} {v3g['top1']:>{cw-3}}/{v3g['total']} {v4cg['top1']:>{cw-3}}/{v4cg['total']} {v4qg['top1']:>{cw-3}}/{v4qg['total']}")
    print(f"{'그룹'+g_label+' Avg':<30} {v3g['avg']:>{cw}.4f} {v4cg['avg']:>{cw}.4f} {v4qg['avg']:>{cw}.4f}")

print("-" * len(header))

# Combined
v3_comb = combined_score(v3_vlm, v3_kw_top1)
v4c_comb = combined_score(v4_vlm_content, v4_kw_top1)
v4q_comb = combined_score(v4_vlm_qdrant, v4q_kw_top1)
print(f"{'Combined Score':<30} {v3_comb:>{cw}.4f} {v4c_comb:>{cw}.4f} {v4q_comb:>{cw}.4f}")

# 과적합
print(f"\n{'과적합 (A-C Top1 Gap)':<30}", end="")
for vlm_r in [v3_vlm, v4_vlm_content, v4_vlm_qdrant]:
    a = vlm_r["group_stats"]["A"]["top1"] / vlm_r["group_stats"]["A"]["total"]
    c = vlm_r["group_stats"]["C"]["top1"] / vlm_r["group_stats"]["C"]["total"]
    gap = (a - c) * 100
    print(f" {gap:>+{cw-1}.1f}%p", end="")
print()


# ══════════════════════════════════════════════════════════════
# 고질적 오답 5건
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  고질적 오답 5건 상세")
print(f"{'=' * 100}\n")

targets = {
    "차20-2": "A5-차선변경끼어들기접촉",
    "차3-1":  "B2-적색우회전녹색직진교차충돌",
    "차4-1":  "B3-동일방향직진우회전교차충돌",
    "차31-2": "B5-일방통행로역주행정면충돌",
    "차11-2": "B9-비신호동일폭교차로우측차우선위반",
}

v3_map = {d["name"]: d for d in v3_vlm["details"]}
v4c_map = {d["name"]: d for d in v4_vlm_content["details"]}
v4q_map = {d["name"]: d for d in v4_vlm_qdrant["details"]}

print(f"  {'Chunk':<8} {'테스트명':<42} {'':>4} {'v3(content)':>16} {'v4(content)':>16} {'v4(embed_txt)':>16}")
print(f"  {'-'*8} {'-'*42} {'-'*4} {'-'*16} {'-'*16} {'-'*16}")

for cid, tname in targets.items():
    d3 = v3_map.get(tname, {})
    d4c = v4c_map.get(tname, {})
    d4q = v4q_map.get(tname, {})

    def fmt(d):
        if not d:
            return "N/A"
        r = d.get("rank", 999)
        s = d.get("score", 0)
        t = d.get("top1", "?")
        mark = "O" if r == 1 else f"X({t})"
        return f"r={r} {s:.2f} {mark}"

    print(f"  {cid:<8} {tname:<42} rank {fmt(d3):>16} {fmt(d4c):>16} {fmt(d4q):>16}")


# ══════════════════════════════════════════════════════════════
# 쿼리별 전체 상세 (v4 Qdrant)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  v4(embed_text/Qdrant) 쿼리별 상세")
print(f"{'=' * 100}\n")

print(f"  {'그룹':<4} {'이름':<42} {'정답':>6} {'Top1':>8} {'Rank':>5} {'Score':>7} {'판정'}")
print(f"  {'-'*4} {'-'*42} {'-'*6} {'-'*8} {'-'*5} {'-'*7} {'-'*6}")

for d in v4_vlm_qdrant["details"]:
    d3 = v3_map.get(d["name"], {})
    mark = "O" if d["rank"] == 1 else ("T3" if d["rank"] <= 3 else "X")
    change = ""
    if d3:
        if d["rank"] == 1 and d3.get("rank", 999) != 1:
            change = " ++FIX"
        elif d["rank"] != 1 and d3.get("rank") == 1:
            change = " --REG"
    print(f"  {d['name'][0]:<4} {d['name']:<42} {d['expected']:>6} {d['top1']:>8} {d['rank']:>5} {d['score']:>7.4f} {mark}{change}")


# ── JSON 저장 ──
result = {
    "v3_content": {
        "kw_top1": v3_kw_top1, "kw_avg": v3_kw_avg,
        "vlm_top1": v3_vlm["top1"], "vlm_top3": v3_vlm["top3"], "vlm_avg": v3_vlm["avg"],
        "combined": v3_comb,
        "group_stats": v3_vlm["group_stats"],
        "details": v3_vlm["details"],
    },
    "v4_content": {
        "kw_top1": v4_kw_top1, "kw_avg": v4_kw_avg,
        "vlm_top1": v4_vlm_content["top1"], "vlm_top3": v4_vlm_content["top3"], "vlm_avg": v4_vlm_content["avg"],
        "combined": v4c_comb,
        "group_stats": v4_vlm_content["group_stats"],
        "details": v4_vlm_content["details"],
    },
    "v4_qdrant_embed_text": {
        "kw_top1": v4q_kw_top1, "kw_avg": v4q_kw_avg,
        "vlm_top1": v4_vlm_qdrant["top1"], "vlm_top3": v4_vlm_qdrant["top3"], "vlm_avg": v4_vlm_qdrant["avg"],
        "combined": v4q_comb,
        "group_stats": v4_vlm_qdrant["group_stats"],
        "details": v4_vlm_qdrant["details"],
    },
}
with open(os.path.join(BASE_DIR, "test_qdrant_vlm_result.json"), "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: test_qdrant_vlm_result.json")
