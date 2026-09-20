"""panel-selection 接口端到端测试与分支限界单测。"""

from __future__ import annotations

import itertools
import os
import random

import pytest

from app.panel import select_panel


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("panel_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# 主批次风险画像（阈值 ct=4/tt=3，已用 analyze_pair 独立核算）：
#   F(pos0)=GGGGGAAAAA  自 low(0,0)；R(pos1)=GGGGGTTTTT  自 low(0,0)
#   H(pos2)=ACGTACGTAC  自 high(8,8)；A(pos3)=ATACACGTCA  自 low(1,1)
#   B(pos4)=AAGTAAGTGT  自 low(1,1)；C(pos5)=TCTATTGACT  自 low(1,1)
#   F-R high(5,5)；H-B high(3,3)；A-B high(4,4)；A-C high(4,4)；B-C high(3,3)
#   其余异二聚体均 low（3′端连续 ≤2）
PAYLOAD = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
        {"name": "H", "sequence": "ACGTACGTAC"},
        {"name": "A", "sequence": "ATACACGTCA"},
        {"name": "B", "sequence": "AAGTAAGTGT"},
        {"name": "C", "sequence": "TCTATTGACT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}

# 含 medium 配对的第二批次（阈值相同）：
#   G1(pos0)=GAGACTAGAA 自 low(1,1)；G2(pos1)=GCTGCTAGTG 自 low(1,1)
#   N1(pos2)=CTAATGCTAA 自 low(1,1)；N4(pos3)=CCATGTCCGT 自 low(1,1)
#   H(pos4) 自 high(8,8)；M1(pos5)=TTGGGGTTAA 自 high(4,4)
#   G1-G2 medium(5,1)；N1-M1 high(3,3)；N4-H high(3,3)；其余异二聚体 low
MEDIUM_PAYLOAD = {
    "primers": [
        {"name": "G1", "sequence": "GAGACTAGAA"},
        {"name": "G2", "sequence": "GCTGCTAGTG"},
        {"name": "N1", "sequence": "CTAATGCTAA"},
        {"name": "N4", "sequence": "CCATGTCCGT"},
        {"name": "H", "sequence": "ACGTACGTAC"},
        {"name": "M1", "sequence": "TTGGGGTTAA"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}


@pytest.fixture(scope="module")
def batch_id(client):
    return client.post("/api/batches", json=PAYLOAD).json()["id"]


@pytest.fixture(scope="module")
def medium_batch_id(client):
    return client.post("/api/batches", json=MEDIUM_PAYLOAD).json()["id"]


def _select(client, batch_id, groups):
    return client.post(
        f"/api/batches/{batch_id}/panel-selection",
        json={"groups": [{"primer_names": names} for names in groups]},
    )


# ---------- 基本求解与目标字典序 ----------

def test_clean_optimum_no_risky_pairs(client, batch_id):
    # F+A 全 low：(0,0,1)；F-R 与 H 自二聚体的备选组合均含 high
    resp = _select(client, batch_id, [["F", "H"], ["R", "A"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["batch_id"] == batch_id
    assert data["group_count"] == 2
    assert data["selected_count"] == 2
    assert data["combination_count"] == 4
    assert data["selected_pair_count"] == 3
    assert data["high_pair_count"] == 0
    assert data["medium_pair_count"] == 0
    assert data["max_three_prime_run"] == 1
    assert [
        (s["group_index"], s["name"], s["position"]) for s in data["selected"]
    ] == [(0, "F", 0), (1, "A", 3)]
    assert data["selected"][0]["sequence"] == "GGGGGAAAAA"
    assert data["risky_pairs"] == []


def test_position_tiebreak_in_group_request_order(client, batch_id):
    # F+A(0,3)、F+B(0,4)、R+A(1,3) 同为 (0,0,1)，位置元组最小者 F、A 入选
    resp = _select(client, batch_id, [["F", "R"], ["A", "B"]])
    assert resp.status_code == 200, resp.text
    assert [s["name"] for s in resp.json()["selected"]] == ["F", "A"]

    # 决胜元组按组请求顺序排列：A(pos3) 在组0、F(pos0) 在组1 时元组为 (3,0)，
    # 小于 A+R 的 (3,1) 与 B+F 的 (4,0)
    resp = _select(client, batch_id, [["A", "B"], ["F", "R"]])
    assert resp.status_code == 200, resp.text
    assert [s["name"] for s in resp.json()["selected"]] == ["A", "F"]


def test_forced_high_minimizes_count_then_three_prime(client, batch_id):
    # 任何组合至少 1 个 high；(1,0,t) 中 t 最小为 F,B,C 的 3（B-C high(3,3)）
    resp = _select(client, batch_id, [["F", "A"], ["R", "B"], ["C", "H"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [s["name"] for s in data["selected"]] == ["F", "B", "C"]
    assert data["high_pair_count"] == 1
    assert data["medium_pair_count"] == 0
    assert data["max_three_prime_run"] == 3
    assert data["selected_pair_count"] == 6
    assert data["combination_count"] == 8
    assert len(data["risky_pairs"]) == 1
    pair = data["risky_pairs"][0]
    assert pair["pair_index"] == 20
    assert pair["name_top"] == "B"
    assert pair["name_bottom"] == "C"
    assert pair["is_self_dimer"] is False
    assert pair["risk_level"] == "high"
    assert pair["three_prime_run_length"] == 3
    assert pair["alignment"]["top"]


def test_self_dimer_high_counts_and_appears_in_risky_pairs(
    client, medium_batch_id
):
    # 组0 必选自 high 引物（H 自 high(8,8)、M1 自 high(4,4)）：
    # M1+G1 仅 M1 自二聚体 1 个 high(t=4)，优于 H+G1 的 (1,0,8) 与
    # M1+N1 的 2 个 high（M1 自 + N1-M1）
    resp = _select(client, medium_batch_id, [["H", "M1"], ["G1", "N1"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [s["name"] for s in data["selected"]] == ["M1", "G1"]
    assert data["high_pair_count"] == 1
    assert data["max_three_prime_run"] == 4
    assert len(data["risky_pairs"]) == 1
    pair = data["risky_pairs"][0]
    assert pair["is_self_dimer"] is True
    assert pair["name_top"] == pair["name_bottom"] == "M1"
    assert pair["risk_level"] == "high"
    assert pair["three_prime_run_length"] == 4


def test_medium_count_dominates_three_prime_run(client, medium_batch_id):
    # G1+G2 为 (0,1,1)：medium 数 1 输给 (0,0,1)，尽管其 3′端连续最小；
    # (0,0,1) 并列 N1+G2(2,1) 与 N1+N4(2,3)，位置元组取 N1、G2
    resp = _select(client, medium_batch_id, [["G1", "N1"], ["G2", "N4"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [s["name"] for s in data["selected"]] == ["N1", "G2"]
    assert data["high_pair_count"] == 0
    assert data["medium_pair_count"] == 0
    assert data["max_three_prime_run"] == 1
    assert data["risky_pairs"] == []


def test_medium_pair_returned_in_risky_pairs(client, medium_batch_id):
    # 备选组合均含自 high 引物：G1+G2 以 (0,1,1) 胜出，medium 配对如实返回
    resp = _select(client, medium_batch_id, [["G1", "H"], ["G2", "M1"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [s["name"] for s in data["selected"]] == ["G1", "G2"]
    assert data["high_pair_count"] == 0
    assert data["medium_pair_count"] == 1
    assert data["max_three_prime_run"] == 1
    assert len(data["risky_pairs"]) == 1
    pair = data["risky_pairs"][0]
    assert (pair["name_top"], pair["name_bottom"]) == ("G1", "G2")
    assert pair["risk_level"] == "medium"
    assert pair["is_self_dimer"] is False


def test_eight_groups_combination_count(client):
    # 8 组（接口上限）× 2 候选 = 256 组合；入选 8 条，配对数 8*9/2 = 36
    primers = [
        {
            "name": f"P{i}",
            "sequence": "".join("ACGT"[(i + j) % 4] for j in range(10)),
        }
        for i in range(16)
    ]
    bid = client.post(
        "/api/batches",
        json={
            "primers": primers,
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    ).json()["id"]
    resp = _select(
        client, bid, [[f"P{2 * g}", f"P{2 * g + 1}"] for g in range(8)]
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["group_count"] == 8
    assert data["combination_count"] == 256
    assert data["selected_count"] == 8
    assert data["selected_pair_count"] == 36
    assert [s["group_index"] for s in data["selected"]] == list(range(8))
    # 每组恰选一条且跨组不重复
    positions = [s["position"] for s in data["selected"]]
    assert len(set(positions)) == 8
    for g, s in enumerate(data["selected"]):
        assert s["position"] in (2 * g, 2 * g + 1)


# ---------- 只读与确定性 ----------

def test_batch_not_modified_and_deterministic(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    body = {
        "groups": [
            {"primer_names": ["F", "A"]},
            {"primer_names": ["R", "B"]},
            {"primer_names": ["C", "H"]},
        ]
    }
    r1 = client.post(f"/api/batches/{batch_id}/panel-selection", json=body)
    r2 = client.post(f"/api/batches/{batch_id}/panel-selection", json=body)
    assert r1.status_code == 200
    assert r1.json() == r2.json()

    after = client.get(f"/api/batches/{batch_id}").json()
    assert after["pair_count"] == before["pair_count"]
    assert after["primer_count"] == before["primer_count"]
    assert [
        (p["pair_index"], p["name_top"], p["name_bottom"], p["risk_level"])
        for p in after["pairs"]
    ] == [
        (p["pair_index"], p["name_top"], p["name_bottom"], p["risk_level"])
        for p in before["pairs"]
    ]


# ---------- 错误处理 ----------

def test_batch_not_found_404(client):
    resp = _select(client, "no-such-batch", [["F", "R"], ["A", "B"]])
    assert resp.status_code == 404


def test_unknown_name_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "ghost"], ["A", "B"]])
    assert resp.status_code == 422
    assert "ghost" in resp.json()["detail"]


def test_duplicate_within_group_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "F"], ["A", "B"]])
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"]


def test_duplicate_across_groups_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "R"], ["A", "F"]])
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"]


def test_blank_name_422(client, batch_id):
    resp = client.post(
        f"/api/batches/{batch_id}/panel-selection",
        json={
            "groups": [
                {"primer_names": ["   ", "F"]},
                {"primer_names": ["A", "B"]},
            ]
        },
    )
    assert resp.status_code == 422


def test_group_count_bounds_422(client, batch_id):
    one = _select(client, batch_id, [["F", "R"]])
    assert one.status_code == 422

    nine = _select(client, batch_id, [["F", "R"]] * 9)
    assert nine.status_code == 422


def test_group_size_bounds_422(client, batch_id):
    one_name = _select(client, batch_id, [["F"], ["A", "B"]])
    assert one_name.status_code == 422

    six_names = _select(
        client, batch_id, [["F", "R", "H", "A", "B", "C"], ["A", "B"]]
    )
    assert six_names.status_code == 422


def test_combination_limit_422(client):
    # 服务层防线：9 组 × 5 候选 = 5^9 = 1953125 > 390625（路由层组数上限
    # 为 8，结构性组合数不超过 5^8；直接调用服务函数验证超限分支）
    primers = [
        {
            "name": f"P{i}",
            "sequence": "".join("ACGT"[(i + j) % 4] for j in range(10)),
        }
        for i in range(45)
    ]
    bid = client.post(
        "/api/batches",
        json={
            "primers": primers,
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    ).json()["id"]

    from app.services import plan_batch_panel_selection

    groups = [[f"P{i}" for i in range(g * 5, g * 5 + 5)] for g in range(9)]
    with pytest.raises(ValueError, match="超过上限"):
        plan_batch_panel_selection(batch_id=bid, groups=groups)


# ---------- 分支限界精确求解单测 ----------

def _full_metrics(n, overrides):
    metrics = {(a, b): ("low", 0) for a in range(n) for b in range(a, n)}
    metrics.update(overrides)
    return metrics


def test_solver_objective_lexicographic_order():
    # 组合目标值：(0,2)→(1,0,5)；(0,3)→(0,2,2)；(1,2)→(0,1,9)；(1,3)→(0,1,2)
    # high 数优先于 medium 数、medium 数优先于 3′端连续 → (1,3) 唯一最优
    metrics = _full_metrics(
        4,
        {
            (3, 3): ("medium", 2),
            (0, 2): ("high", 5),
            (0, 3): ("medium", 1),
            (1, 2): ("medium", 9),
            (1, 3): ("low", 2),
        },
    )
    sel = select_panel([[0, 1], [2, 3]], metrics)
    assert sel.selected == (1, 3)
    assert (sel.high_count, sel.medium_count, sel.max_three_prime_run) == (0, 1, 2)


def test_solver_tiebreak_smallest_position_tuple():
    # 全部 low 且 3′端相同：目标值并列，取组请求顺序下位置元组最小者；
    # 组内候选未按位置给出时求解器内部排序
    metrics = _full_metrics(4, {(a, b): ("low", 1) for a in range(4)
                                for b in range(a, 4)})
    sel = select_panel([[3, 1], [2, 0]], metrics)
    assert sel.selected == (1, 0)
    assert (sel.high_count, sel.medium_count, sel.max_three_prime_run) == (0, 0, 1)


def test_solver_self_pair_metrics_used():
    # 位置 0 自 high(7)、位置 1 自 low(1)：两组各一条时自 high 计入目标
    metrics = _full_metrics(2, {(0, 0): ("high", 7), (1, 1): ("low", 1)})
    sel = select_panel([[0], [1]], metrics)
    assert sel.selected == (0, 1)
    assert (sel.high_count, sel.medium_count, sel.max_three_prime_run) == (1, 0, 7)


def test_solver_matches_brute_force_randomized():
    rng = random.Random(20260920)
    for _ in range(4000):
        k = rng.randint(2, 5)
        sizes = [rng.randint(2, 4) for _ in range(k)]
        positions = list(range(sum(sizes)))
        rng.shuffle(positions)
        groups = []
        idx = 0
        for size in sizes:
            groups.append(sorted(positions[idx:idx + size]))
            idx += size
        n = sum(sizes)
        metrics = {}
        for a in range(n):
            for b in range(a, n):
                metrics[(a, b)] = (
                    rng.choice(["low", "low", "medium", "high"]),
                    rng.randint(0, 9),
                )

        sel = select_panel(groups, metrics)

        best = None
        for combo in itertools.product(*groups):
            h = m = t = 0
            for x, y in itertools.combinations_with_replacement(combo, 2):
                level, run = metrics[(x, y) if x <= y else (y, x)]
                if level == "high":
                    h += 1
                elif level == "medium":
                    m += 1
                if run > t:
                    t = run
            candidate = ((h, m, t), combo)
            if best is None or candidate < best:
                best = candidate
        (eh, em, et), expected_combo = best
        assert (
            sel.high_count,
            sel.medium_count,
            sel.max_three_prime_run,
        ) == (eh, em, et)
        assert sel.selected == expected_combo


def test_solver_scale_8_groups_of_5_is_fast():
    import time

    rng = random.Random(3)
    n = 40
    metrics = {}
    for a in range(n):
        for b in range(a, n):
            metrics[(a, b)] = (
                rng.choice(["low", "medium", "high"]),
                rng.randint(0, 9),
            )
    groups = [list(range(g * 5, g * 5 + 5)) for g in range(8)]
    start = time.time()
    sel = select_panel(groups, metrics)
    assert time.time() - start < 5
    assert len(sel.selected) == 8
    # 结果必须真的是每组一条
    for g, pos in enumerate(sel.selected):
        assert pos in groups[g]
