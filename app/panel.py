"""多重 PCR 面板联合选择的确定性分支限界精确求解。

问题模型
--------
k 个目标组按请求顺序编号 0..k-1，组 g 提供若干候选引物（以批次位置表示，
组内按批次位置升序）。从每组恰选一条引物构成一个面板；面板的全部相关
配对为入选引物的自二聚体与入选引物两两异二聚体，共 k(k+1)/2 对，每对
带有已保存的风险等级（high/medium/low）与 3′端连续互补长度两个标量。

目标（按字典序依次最小化）
--------------------------
1. high 配对数；
2. medium 配对数；
3. 最大 3′端连续互补长度。

同分决胜：把入选引物的批次位置按组请求顺序排成元组，字典序最小者唯一。

分支限界
--------
按组序深度优先、组内按批次位置升序枚举候选，叶子恰好按决胜元组的字典序
被访问。每加入一条引物，增量累加其自二聚体及其与已选引物异二聚体的目标
分量；三个分量都随配对增加而单调不减，故部分解的目标元组是其全部完成的
分量级下界——部分解不严格优于当前最优完成解时，整棵子树不可能产生严格
更优解（目标相等者因访问顺序必在决胜元组上更大），直接剪枝。因此只在
严格更优时替换最优解，首次达到某目标值的叶子即该目标值下决胜元组最小
者，全序无并列，结果稳定可复现。搜索组合数（各组候选数之积）由接口
限制在 390625 以内，最坏情形亦可瞬时完成。
"""

from __future__ import annotations

from typing import NamedTuple


class PanelSelection(NamedTuple):
    """精确求解结果：每组入选引物的批次位置（按组请求顺序）与目标值。"""

    selected: tuple[int, ...]
    high_count: int
    medium_count: int
    max_three_prime_run: int


def select_panel(
    groups: list[list[int]],
    metrics: dict[tuple[int, int], tuple[str, int]],
) -> PanelSelection:
    """确定性分支限界求面板联合选择的最优解。

    ``groups[g]`` 为组 g 的候选引物批次位置（函数内按升序排序，保证枚举
    顺序即决胜顺序）；``metrics[(u, v)]``（u <= v）给出批次位置 u、v 间
    已保存配对的 ``(风险等级, 3′端连续互补长度)``，自二聚体键为 ``(v, v)``。
    任一候选位置的自二聚体与跨组位置对都必须存在于 ``metrics``。
    """
    candidates = [sorted(group) for group in groups]
    k = len(candidates)

    best_key: tuple[int, int, int] | None = None
    best_selected: tuple[int, ...] = ()
    chosen: list[int] = []  # 已选批次位置（按组序追加）

    def pair_metric(u: int, v: int) -> tuple[str, int]:
        return metrics[(u, v) if u <= v else (v, u)]

    def search(g: int, high: int, medium: int, three: int) -> None:
        nonlocal best_key, best_selected

        # 下界剪枝：三个目标分量单调不减，部分解不严格更优时整棵子树无望
        if best_key is not None and (high, medium, three) >= best_key:
            return
        if g == k:
            # 叶子按决胜元组字典序被访问；只在严格更优时替换，同分保留先到者
            best_key = (high, medium, three)
            best_selected = tuple(chosen)
            return

        for pos in candidates[g]:
            h, m, t = high, medium, three
            level, run = pair_metric(pos, pos)  # 入选引物自二聚体
            if level == "high":
                h += 1
            elif level == "medium":
                m += 1
            if run > t:
                t = run
            for other in chosen:  # 与已选引物的异二聚体
                level, run = pair_metric(pos, other)
                if level == "high":
                    h += 1
                elif level == "medium":
                    m += 1
                if run > t:
                    t = run
            chosen.append(pos)
            search(g + 1, h, m, t)
            chosen.pop()

    search(0, 0, 0, 0)
    assert best_key is not None, "各组非空时搜索必然到达叶子"
    return PanelSelection(
        selected=best_selected,
        high_count=best_key[0],
        medium_count=best_key[1],
        max_three_prime_run=best_key[2],
    )
