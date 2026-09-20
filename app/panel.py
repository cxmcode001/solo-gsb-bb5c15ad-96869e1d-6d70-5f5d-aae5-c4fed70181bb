"""多重 PCR 面板的联合选择（确定性分支限界精确解）。

问题模型
--------
请求给出 2～8 个目标组，每组 2～5 条批次内候选引物（跨组不重复）。从每组
恰选一条组成多重 PCR 面板；面板的配对集合为入选引物的**自二聚体**与入选
引物两两间的**异二聚体**，全部取自批次已保存结果（不重算对齐）。

目标（按字典序依次最小化）
------------------------
1. ``high`` 配对数；
2. ``medium`` 配对数；
3. 面板全部配对（含 low）的最大 3′连续互补长度。

同分决胜
--------
目标值完全相同的解之间，按各组请求顺序取入选引物的批次位置组成元组，
字典序最小者为唯一解（各组顶点互不相同，故无并列）。

分支限界
--------
按组序深度优先、组内按候选顶点升序枚举，完整选择元组即按位置字典序被
访问，故首个达到最优目标的解就是同分唯一解。目标随配对加入单调不减：
部分解的三元目标一旦不优于当前最优即可剪枝；另对每组预计算候选自二聚体
的最小 high/medium 计数与最小 3′连续长度，按后缀聚合成剩余组的下界加强
剪枝。搜索组合数（各组规模之积）超过上限时抛出 ValueError。
"""

from __future__ import annotations

from typing import NamedTuple


class PanelSelection(NamedTuple):
    """精确求解结果。

    selection: 各组入选顶点（批次位置），按组序排列；
    objectives: (high 配对数, medium 配对数, 最大 3′连续互补长度)。
    """

    selection: tuple[int, ...]
    objectives: tuple[int, int, int]


def select_panel(
    groups: list[list[int]],
    pair_metrics: dict[tuple[int, int], tuple[str, int]],
    max_combinations: int,
) -> PanelSelection:
    """确定性分支限界求面板联合选择的最优解。

    groups 为各组候选顶点（批次位置）列表，组内必须已按升序排列、跨组
    无重复顶点；pair_metrics 以无序顶点对 ``(u, v)``（u <= v，自二聚体
    u == v）为键，值为 ``(risk_level, 3′连续互补长度)``，须覆盖全部候选
    的自二聚体与候选两两异二聚体。搜索组合数超过 max_combinations 时
    抛出 ValueError。
    """
    group_count = len(groups)
    combinations = 1
    for group in groups:
        combinations *= len(group)
    if combinations > max_combinations:
        raise ValueError(
            f"搜索组合数 {combinations} 超过上限 {max_combinations}"
        )

    def metrics(u: int, v: int) -> tuple[str, int]:
        return pair_metrics[(u, v) if u <= v else (v, u)]

    # 每组候选自二聚体的逐维下界（异二聚体贡献非负，不计入仍合法）
    min_self_high = [
        min(1 if metrics(v, v)[0] == "high" else 0 for v in group)
        for group in groups
    ]
    min_self_medium = [
        min(1 if metrics(v, v)[0] == "medium" else 0 for v in group)
        for group in groups
    ]
    min_self_three = [min(metrics(v, v)[1] for v in group) for group in groups]
    # 后缀聚合：rem_*[g] 为第 g..group_count-1 组的下界贡献
    rem_high = [0] * (group_count + 1)
    rem_medium = [0] * (group_count + 1)
    rem_three = [0] * (group_count + 1)
    for g in range(group_count - 1, -1, -1):
        rem_high[g] = rem_high[g + 1] + min_self_high[g]
        rem_medium[g] = rem_medium[g + 1] + min_self_medium[g]
        rem_three[g] = max(rem_three[g + 1], min_self_three[g])

    best_objectives: tuple[int, int, int] | None = None
    best_selection: tuple[int, ...] = ()
    chosen: list[int] = []

    def dfs(g: int, high: int, medium: int, three: int) -> None:
        nonlocal best_objectives, best_selection

        if best_objectives is not None:
            # 下界不优于当前最优：子树内不可能有更优解；目标相同的解按
            # 位置字典序也必晚于已访问的最优解（枚举顺序即字典序）
            lower = (
                high + rem_high[g],
                medium + rem_medium[g],
                max(three, rem_three[g]),
            )
            if lower >= best_objectives:
                return

        if g == group_count:
            objectives = (high, medium, three)
            if best_objectives is None or objectives < best_objectives:
                best_objectives = objectives
                best_selection = tuple(chosen)
            return

        for v in groups[g]:
            self_level, self_run = metrics(v, v)
            next_high = high + (1 if self_level == "high" else 0)
            next_medium = medium + (1 if self_level == "medium" else 0)
            next_three = max(three, self_run)
            for u in chosen:
                level, run = metrics(u, v)
                if level == "high":
                    next_high += 1
                elif level == "medium":
                    next_medium += 1
                if run > next_three:
                    next_three = run
            chosen.append(v)
            dfs(g + 1, next_high, next_medium, next_three)
            chosen.pop()

    dfs(0, 0, 0, 0)
    assert best_objectives is not None  # 每组非空，必然到达完整解
    return PanelSelection(
        selection=best_selection, objectives=best_objectives
    )
