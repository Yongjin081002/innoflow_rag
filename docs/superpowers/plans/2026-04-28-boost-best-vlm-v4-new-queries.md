# boost_best_vlm_v4 New Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated script that evaluates `boost_best_vlm_v4` with the 20 unseen VLM-style queries and prints a readable table plus summary metrics.

**Architecture:** Keep the new evaluator separate from existing comparison scripts. Reuse the 20-query dataset already defined in `test_vlm_new_queries.py`, add small pure functions for result formatting so they can be tested without loading the model, and write JSON output for reproducibility.

**Tech Stack:** Python 3, `sentence-transformers`, `torch`, standard library `json` and `os`

---

### Task 1: Add regression tests for reusable helpers

**Files:**
- Create: `innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py`
- Test: `innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_new_vlm_queries_returns_20_items():
    queries = load_new_vlm_queries()
    assert len(queries) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py -v`
Expected: FAIL because the new module and helper functions do not exist yet.

- [ ] **Step 3: Expand test coverage for table rows and summary**

```python
def test_build_table_rows_marks_top1_top3_and_top5():
    ...
```

- [ ] **Step 4: Run test to verify it still fails for missing implementation**

Run: `python3 -m unittest /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py -v`
Expected: FAIL with import or attribute errors from the new module.

- [ ] **Step 5: Commit**

```bash
git add /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py
git commit -m "test: add new query evaluation helper coverage"
```

### Task 2: Implement the dedicated evaluator script

**Files:**
- Create: `innoflow_rag/test_boost_best_vlm_v4_new_queries.py`
- Modify: `innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py`

- [ ] **Step 1: Write minimal implementation**

```python
def load_new_vlm_queries():
    from test_vlm_new_queries import NEW_VLM_QUERIES
    return NEW_VLM_QUERIES
```

- [ ] **Step 2: Add pure helpers for summary and row formatting**

```python
def compute_summary(details):
    ...

def build_table_rows(details):
    ...
```

- [ ] **Step 3: Implement model evaluation and JSON persistence**

```python
def evaluate_model(model_path):
    ...
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python3 -m unittest /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries.py /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries_unit.py
git commit -m "feat: add boost_best_vlm_v4 new query evaluator"
```

### Task 3: Verify end-to-end execution with the real model

**Files:**
- Modify: `innoflow_rag/test_boost_best_vlm_v4_new_queries.py`

- [ ] **Step 1: Run the script**

Run: `cd /home/minsung0830/innoflow_rag && python3 test_boost_best_vlm_v4_new_queries.py`
Expected: printed table for 20 items and summary metrics.

- [ ] **Step 2: Check output artifact**

Run: `cd /home/minsung0830/innoflow_rag && ls -1 boost_best_vlm_v4_new_queries_result.json`
Expected: file exists.

- [ ] **Step 3: Inspect a small slice of the JSON**

Run: `cd /home/minsung0830/innoflow_rag && sed -n '1,80p' boost_best_vlm_v4_new_queries_result.json`
Expected: includes `model`, `summary`, and `details`.

- [ ] **Step 4: Adjust formatting only if verification reveals readability issues**

```python
print(...)
```

- [ ] **Step 5: Commit**

```bash
git add /home/minsung0830/innoflow_rag/test_boost_best_vlm_v4_new_queries.py /home/minsung0830/innoflow_rag/boost_best_vlm_v4_new_queries_result.json
git commit -m "chore: verify new query evaluation output"
```
