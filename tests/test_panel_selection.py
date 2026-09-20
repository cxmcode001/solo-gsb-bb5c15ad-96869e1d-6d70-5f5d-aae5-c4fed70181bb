"""panel-selection 接口端到端测试与分支限界单测。"""

from __future__ import annotations

import itertools
import os
import random
import time

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


# 10 引物批次（阈值 ct=4/tt=3），风险画像已经 analyze_pair 独立核算：
#   自二聚体：H high(8,8)、M1 high(6,6)，其余 low
#   high 异二聚体：F-R(5,5)、F-M1(3,3)、R-M1(3,3)、R-M2(4,4)、H-B(3,3)、
#     A-B(4,4)、A-C(4,4)、B-C(3,3)、B-D1(4,4)、B-D2(4,4)
#   medium 异二聚体：F-M2(4,0)、M1-M2(4,0)；其余配对均 low
PAYLOAD = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},    # pos 0
        {"name": "R", "sequence": "GGGGGTTTTT"},    # pos 1
        {"name": "H", "sequence": "ACGTACGTAC"},    # pos 2
        {"name": "A", "sequence": "ATACACGTCA"},    # pos 3
        {"name": "B", "sequence": "AAGTAAGTGT"},    # pos 4
        {"name": "C", "sequence": "TCTATTGACT"},    # pos 5
        {"name": "M1", "sequence": "CCCGGGGCCC"},   # pos 6
        {"name": "M2", "sequence": "AACCCCAAAA"},   # pos 7
        {"name": "D1", "sequence": "ACACACACAC"},   # pos 8
        {"name": "D2", "sequence": "CACACACACA"},   # pos 9
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}


@pytest.fixture(scope="module")
def batch_id(client):
    return client.post("/api/batches", json=PAYLOAD).json()["id"]


@pytest.fixture(scope="module")
def free_batch_id(client):
    # 40 条 {A,C} 字母表引物：任意配对（含自二聚体）run=0、t3=0、low
    primers = [
        {
            "name": f"P{i}",
            "sequence": format(i, "010b").replace("0", "A").replace("1", "C"),
        }
        for i in range(40)
    ]
    resp = client.post(
        "/api/batches",
        json={
            "primers": primers,
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _select(client, batch_id, groups):
    return client.post(
        f"/api/batches/{batch_id}/panel-selection",
        json={"groups": [{"primer_names": names} for names in groups]},
    )


def _saved_pairs(client, batch_id):
    """GET 批次结果 -> (名称->位置, (u,v)->(risk,t3), pair_index 键->配对)。"""
    data = client.get(f"/api/batches/{batch_id}").json()
    pos = {p["name"]: p["position"] for p in data["primers"]}
    metrics = {}
    by_key = {}
    for p in data["pairs"]:
        key = (min(pos[p["name_top"]], pos[p["name_bottom"]]),
               max(pos[p["name_top"]], pos[p["name_bottom"]]))
        metrics[key] = (p["risk_level"], p["three_prime_run_length"])
        by_key[key] = p
    return pos, metrics, by_key


def _brute(groups_pos, metrics):
    """穷举全部组合，按 (high, medium, max_t3)、位置元组取最优（独立参考）。"""
    best = None
    best_sel = None
    for combo in itertools.product(*[sorted(g) for g in groups_pos]):
        h = m = t = 0
        for i, u in enumerate(combo):
            for v in combo[i:]:
                level, run = metrics[(min(u, v), max(u, v))]
                if level == "high":
                    h += 1
                elif level == "medium":
                    m += 1
                if run > t:
                    t = run
        obj = (h, m, t)
        if best is None or obj < best:
            best, best_sel = obj, combo
    return best_sel, best


# ---------- 基本选择结果 ----------

def test_basic_selection_avoids_high(client, batch_id):
    # (F,R)=high、(H,*)=H 自 high；(F,A) 全 low，仅 A 自/A-F 的 t3=1
    resp = _select(client, batch_id, [["F", "H"], ["R", "A"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["batch_id"] == batch_id
    assert data["group_count"] == 2
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 0,
        "max_three_prime_run": 1,
    }
    assert [(s["group_index"], s["name"], s["position"]) for s in data["selected"]] == [
        (0, "F", 0),
        (1, "A", 3),
    ]
    assert data["pairs"] == []


def test_medium_avoided_when_zero_medium_possible(client, batch_id):
    # (F,M2)=medium、(M1,*)=M1 自 high；(F,D2) 全 low 且 t3=0
    resp = _select(client, batch_id, [["F", "M1"], ["M2", "D2"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 0,
        "max_three_prime_run": 0,
    }
    assert [s["name"] for s in data["selected"]] == ["F", "D2"]
    assert data["pairs"] == []


def test_medium_count_outranks_max_three_prime(client, batch_id):
    # (F,B)=(0,0,1) 与 (F,M2)=(0,1,0)：medium 数优先于最大 3′长度
    resp = _select(client, batch_id, [["H", "F"], ["B", "M2"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 0,
        "max_three_prime_run": 1,
    }
    assert [s["name"] for s in data["selected"]] == ["F", "B"]


def test_position_tiebreak_ignores_request_order(client, batch_id):
    # (M2,D1) 与 (D2,D1) 同为 (0,0,0)：按批次位置元组取 (7,8) 的 M2、D1，
    # 与组内候选的请求顺序无关
    body = [["D2", "F", "M2"], ["D1", "B"]]
    r1 = _select(client, batch_id, body)
    assert r1.status_code == 200, r1.text
    data = r1.json()
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 0,
        "max_three_prime_run": 0,
    }
    assert [(s["name"], s["position"]) for s in data["selected"]] == [
        ("M2", 7),
        ("D1", 8),
    ]
    # 组内候选顺序打乱结果不变
    r2 = _select(client, batch_id, [["M2", "D2", "F"], ["B", "D1"]])
    assert r2.json() == data


def test_unavoidable_self_high_and_pairs_content(client, batch_id):
    # (H,*)=H 自 high(8,8)、(M1,R)=M1 自 high+R-M1 high；
    # (M1,D2)=(1,0,6) 最优，pairs 仅含 M1 自二聚体
    resp = _select(client, batch_id, [["H", "M1"], ["R", "D2"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objectives"] == {
        "high_pair_count": 1,
        "medium_pair_count": 0,
        "max_three_prime_run": 6,
    }
    assert [s["name"] for s in data["selected"]] == ["M1", "D2"]
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair["is_self_dimer"] is True
    assert pair["name_top"] == pair["name_bottom"] == "M1"
    assert pair["risk_level"] == "high"
    assert pair["three_prime_run_length"] == 6


def test_medium_pair_returned_with_full_fields(client, batch_id):
    # (F,M2)=(0,1,0) 最优；pairs 恰为 F-M2 medium 配对，与批次配对同构
    resp = _select(client, batch_id, [["F", "H"], ["R", "M2"]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 1,
        "max_three_prime_run": 0,
    }
    assert [s["name"] for s in data["selected"]] == ["F", "M2"]

    _, _, by_key = _saved_pairs(client, batch_id)
    saved = by_key[(0, 7)]
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair == saved
    assert pair["risk_level"] == "medium"
    assert pair["continuous_complementary_length"] == 4
    assert pair["three_prime_run_length"] == 0


def test_whitespace_in_names_is_normalized(client, batch_id):
    resp = _select(client, batch_id, [[" F ", "H"], ["R", " A\n"]])
    assert resp.status_code == 200, resp.text
    assert [s["name"] for s in resp.json()["selected"]] == ["F", "A"]


# ---------- 与穷举参考一致 ----------

@pytest.mark.parametrize(
    "groups",
    [
        [["F", "R", "H"], ["A", "B"]],
        [["H", "M1"], ["F", "M2", "D1"], ["B", "C"]],
        [["F", "D2"], ["M2", "A"], ["H", "C"], ["B", "D1"]],
        [["R", "M1", "D2", "A"], ["F", "H"], ["C", "M2", "B"]],
    ],
)
def test_matches_brute_force(client, batch_id, groups):
    pos, metrics, by_key = _saved_pairs(client, batch_id)
    groups_pos = [[pos[name] for name in names] for names in groups]
    exp_sel, exp_obj = _brute(groups_pos, metrics)

    resp = _select(client, batch_id, groups)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert tuple(s["position"] for s in data["selected"]) == exp_sel
    assert [s["group_index"] for s in data["selected"]] == list(range(len(groups)))
    assert (
        data["objectives"]["high_pair_count"],
        data["objectives"]["medium_pair_count"],
        data["objectives"]["max_three_prime_run"],
    ) == exp_obj

    # pairs 恰为面板内全部 high/medium 配对，按 pair_index 升序
    keys = []
    for i, u in enumerate(exp_sel):
        keys.append((u, u))
        for v in exp_sel[i + 1:]:
            keys.append((min(u, v), max(u, v)))
    exp_pairs = sorted(
        (by_key[k] for k in keys if metrics[k][0] != "low"),
        key=lambda p: p["pair_index"],
    )
    assert data["pairs"] == exp_pairs
    assert data["objectives"]["high_pair_count"] == sum(
        1 for p in exp_pairs if p["risk_level"] == "high"
    )
    assert data["objectives"]["medium_pair_count"] == sum(
        1 for p in exp_pairs if p["risk_level"] == "medium"
    )


# ---------- 上界边界：8 组 × 5 条 = 390625 ----------

def test_max_combinations_boundary_is_accepted(client, free_batch_id):
    groups = [
        [f"P{5 * g + k}" for k in range(4, -1, -1)] for g in range(8)
    ]
    resp = _select(client, free_batch_id, groups)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["group_count"] == 8
    # 全部配对 low(0,0)：目标 (0,0,0)，按位置取每组最小位置
    assert data["objectives"] == {
        "high_pair_count": 0,
        "medium_pair_count": 0,
        "max_three_prime_run": 0,
    }
    assert [s["position"] for s in data["selected"]] == [5 * g for g in range(8)]
    assert data["pairs"] == []


# ---------- 确定性与只读 ----------

def test_deterministic_repeated_requests(client, batch_id):
    groups = [["H", "F", "M1"], ["R", "M2", "D1"], ["A", "B", "C", "D2"]]
    r1 = _select(client, batch_id, groups)
    r2 = _select(client, batch_id, groups)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()


def test_batch_not_modified(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    resp = _select(client, batch_id, [["F", "H"], ["R", "M2"]])
    assert resp.status_code == 200
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
    resp = _select(client, "no-such-batch", [["F", "H"], ["R", "A"]])
    assert resp.status_code == 404


def test_unknown_name_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "ghost"], ["R", "A"]])
    assert resp.status_code == 422
    assert "ghost" in resp.json()["detail"]


def test_duplicate_within_group_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "F"], ["R", "A"]])
    assert resp.status_code == 422


def test_duplicate_across_groups_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "H"], ["R", "F"]])
    assert resp.status_code == 422
    assert "F" in resp.text


def test_group_count_bounds_422(client, batch_id):
    one = _select(client, batch_id, [["F", "H"]])
    assert one.status_code == 422

    nine = _select(
        client,
        batch_id,
        [["F", "R"], ["H", "A"], ["B", "C"], ["M1", "M2"], ["D1", "D2"],
         ["F", "R"], ["H", "A"], ["B", "C"], ["M1", "M2"]],
    )
    assert nine.status_code == 422


def test_group_size_bounds_422(client, batch_id):
    single = _select(client, batch_id, [["F"], ["R", "A"]])
    assert single.status_code == 422

    six = _select(client, batch_id, [["F", "R", "H", "A", "B", "C"], ["M1", "M2"]])
    assert six.status_code == 422


def test_blank_name_422(client, batch_id):
    resp = _select(client, batch_id, [["F", "  "], ["R", "A"]])
    assert resp.status_code == 422


# ---------- 分支限界单测 ----------

def _brute_solver(groups, metrics):
    best = None
    best_sel = None
    for combo in itertools.product(*groups):
        h = m = t = 0
        for i, u in enumerate(combo):
            for v in combo[i:]:
                level, run = metrics[(min(u, v), max(u, v))]
                if level == "high":
                    h += 1
                elif level == "medium":
                    m += 1
                if run > t:
                    t = run
        obj = (h, m, t)
        if best is None or obj < best:
            best, best_sel = obj, combo
    return best_sel, best


def _random_instance(rng, group_count, max_size):
    sizes = [rng.randint(2, max_size) for _ in range(group_count)]
    ids = rng.sample(range(60), sum(sizes))
    groups = []
    k = 0
    for size in sizes:
        groups.append(sorted(ids[k:k + size]))
        k += size
    used = [v for group in groups for v in group]
    metrics = {}
    for i, u in enumerate(used):
        for v in used[i:]:
            level = rng.choices(
                ["low", "medium", "high"], weights=[6, 2, 2]
            )[0]
            metrics[(min(u, v), max(u, v))] = (level, rng.randint(0, 9))
    return groups, metrics


def test_solver_matches_brute_force_randomized():
    rng = random.Random(20260920)
    for _ in range(1500):
        groups, metrics = _random_instance(rng, rng.randint(2, 4), 4)
        sol = select_panel(groups, metrics, max_combinations=390625)
        exp_sel, exp_obj = _brute_solver(groups, metrics)
        assert sol.selection == exp_sel
        assert sol.objectives == exp_obj


def test_solver_matches_brute_force_at_medium_scale():
    rng = random.Random(20260921)
    for _ in range(20):
        groups, metrics = _random_instance(rng, 6, 4)  # 至多 4^6=4096 组合
        sol = select_panel(groups, metrics, max_combinations=390625)
        exp_sel, exp_obj = _brute_solver(groups, metrics)
        assert sol.selection == exp_sel
        assert sol.objectives == exp_obj


def test_solver_combination_limit():
    groups = [[0, 1], [2, 3]]
    metrics = {(u, v): ("low", 0) for u in range(4) for v in range(u, 4)}
    with pytest.raises(ValueError, match="超过上限"):
        select_panel(groups, metrics, max_combinations=3)
    # 等于上限可正常求解
    sol = select_panel(groups, metrics, max_combinations=4)
    assert sol.selection == (0, 2)
    assert sol.objectives == (0, 0, 0)


def test_solver_tiebreak_prefers_smaller_positions():
    # 全部 low 且 3′相同：纯位置决胜，与组内候选顺序无关
    metrics = {(u, v): ("low", 2) for u in range(10) for v in range(u, 10)}
    sol = select_panel([[2, 4, 7], [1, 9]], metrics, max_combinations=100)
    assert sol.selection == (2, 1)
    assert sol.objectives == (0, 0, 2)


def test_solver_objective_priority_medium_over_three_prime():
    # (0,2)=(0,1,0)，(0,3)=(0,0,5)，(1,*)=(1,0,9)：medium 数优先于最大 3′
    metrics = {
        (0, 0): ("low", 0),
        (1, 1): ("high", 9),
        (2, 2): ("low", 0),
        (3, 3): ("low", 0),
        (0, 2): ("medium", 0),
        (0, 3): ("low", 5),
        (1, 2): ("low", 0),
        (1, 3): ("low", 0),
    }
    sol = select_panel([[0, 1], [2, 3]], metrics, max_combinations=100)
    assert sol.selection == (0, 3)
    assert sol.objectives == (0, 0, 5)


def test_solver_max_scale_boundary_is_fast():
    # 8 组 × 5 条 = 390625 组合（恰为上限），全 low 时立即剪枝
    groups = [list(range(5 * g, 5 * g + 5)) for g in range(8)]
    metrics = {(u, v): ("low", 0) for u in range(40) for v in range(u, 40)}
    start = time.time()
    sol = select_panel(groups, metrics, max_combinations=390625)
    assert time.time() - start < 2
    assert sol.selection == tuple(5 * g for g in range(8))
    assert sol.objectives == (0, 0, 0)
