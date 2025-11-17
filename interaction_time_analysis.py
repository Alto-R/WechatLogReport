import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np


# 设置字体为SimHei以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
CHAT_PATH = OUTPUT_DIR / "chat.json"
STATS_PATH = OUTPUT_DIR / "interaction_time_stats.json"


@dataclass
class Msg:
    time: datetime
    is_self: bool
    sender_name: str
    content: str


def load_messages() -> List[Msg]:
    if not CHAT_PATH.exists():
        raise FileNotFoundError(f"chat.json not found at: {CHAT_PATH}")

    with CHAT_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    msgs: List[Msg] = []
    for m in raw:
        ts = m.get("time")
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue

        content = (m.get("content") or "").strip()
        msgs.append(
            Msg(
                time=dt,
                is_self=bool(m.get("isSelf")),
                sender_name=m.get("senderName") or "",
                content=content,
            )
        )

    msgs.sort(key=lambda x: x.time)
    return msgs


def compute_weekday_hour_counts(msgs: List[Msg]) -> Dict[str, Dict[str, int]]:
    """
    统计每个星期几、每个小时的消息数量。
    weekday: 0-6 对应 周一-周日，存成字符串。
    hour: 0-23，存成两位字符串 "00"-"23"。
    """
    counts: Dict[str, Dict[str, int]] = {
        str(w): {f"{h:02d}": 0 for h in range(24)} for w in range(7)
    }
    for m in msgs:
        w = str(m.time.weekday())
        h = f"{m.time.hour:02d}"
        counts[w][h] += 1
    return counts


def compute_daily_hour_counts(msgs: List[Msg]) -> Dict[str, Dict[str, int]]:
    """
    统计每一天、每个小时的消息数量。
    """
    counts: Dict[str, Dict[str, int]] = {}
    for m in msgs:
        date_key = m.time.strftime("%Y-%m-%d")
        hours = counts.setdefault(date_key, {f"{h:02d}": 0 for h in range(24)})
        h = f"{m.time.hour:02d}"
        hours[h] += 1
    return counts


def compute_sessions(msgs: List[Msg], gap_minutes: int = 30) -> List[List[Msg]]:
    """
    按时间间隔划分会话：相邻两条消息间隔超过 gap_minutes 则视为新会话。
    """
    if not msgs:
        return []

    sessions: List[List[Msg]] = []
    current: List[Msg] = [msgs[0]]

    for prev, cur in zip(msgs, msgs[1:]):
        if (cur.time - prev.time) > timedelta(minutes=gap_minutes):
            sessions.append(current)
            current = [cur]
        else:
            current.append(cur)

    sessions.append(current)
    return sessions


def compute_initiative_and_reply(msgs: List[Msg]) -> Dict[str, Any]:
    """
    互动模式量化：
    - 会话发起次数（谁更常开局）
    - 回复速度统计（我/对方平均回复时间、分布）
    - 消息长度统计（字符数、平均值）
    """
    sessions = compute_sessions(msgs)

    # 会话发起统计
    initiator_counter = Counter()

    # 回复时间统计（秒）
    reply_times_self: List[float] = []   # 我回复对方
    reply_times_other: List[float] = []  # 对方回复我

    # 消息长度统计
    length_self: List[int] = []
    length_other: List[int] = []

    for s in sessions:
        if not s:
            continue

        # 会话发起方
        first = s[0]
        initiator = "self" if first.is_self else "other"
        initiator_counter[initiator] += 1

        # 消息长度 & 回复时间
        for i, m in enumerate(s):
            length = len(m.content)
            if m.is_self:
                length_self.append(length)
            else:
                length_other.append(length)

            # 回复时间：从上一条消息到当前这条
            if i == 0:
                continue
            prev = s[i - 1]
            if prev.is_self == m.is_self:
                # 同一方连续说话，不算“回复”
                continue
            delta = (m.time - prev.time).total_seconds()
            if delta < 0:
                continue
            if m.is_self:
                reply_times_self.append(delta)
            else:
                reply_times_other.append(delta)

    def summarize_reply(times: List[float]) -> Dict[str, Any]:
        if not times:
            return {
                "count": 0,
                "avg_seconds": None,
                "median_seconds": None,
                "p90_seconds": None,
                "histogram": {},
            }
        times_sorted = sorted(times)
        n = len(times_sorted)
        avg = sum(times_sorted) / n

        def percentile(p: float) -> float:
            k = int(p * (n - 1))
            return times_sorted[k]

        # 简单时间段分桶（秒）
        buckets: List[Tuple[str, Tuple[int, int]]] = [
            ("<=1min", (0, 60)),
            ("1-5min", (60, 300)),
            ("5-30min", (300, 1800)),
            ("30-120min", (1800, 7200)),
            (">2h", (7200, 10**9)),
        ]
        hist: Dict[str, int] = {name: 0 for name, _ in buckets}
        for t in times_sorted:
            for name, (lo, hi) in buckets:
                if lo <= t < hi:
                    hist[name] += 1
                    break

        return {
            "count": n,
            "avg_seconds": avg,
            "median_seconds": percentile(0.5),
            "p90_seconds": percentile(0.9),
            "histogram": hist,
        }

    def summarize_lengths(lengths: List[int]) -> Dict[str, Any]:
        if not lengths:
            return {
                "count": 0,
                "avg_chars": None,
                "median_chars": None,
                "p90_chars": None,
            }
        ls = sorted(lengths)
        n = len(ls)
        avg = sum(ls) / n

        def percentile(p: float) -> int:
            k = int(p * (n - 1))
            return ls[k]

        return {
            "count": n,
            "avg_chars": avg,
            "median_chars": percentile(0.5),
            "p90_chars": percentile(0.9),
        }

    stats = {
        "sessions_total": len(sessions),
        "sessions_initiator": {
            "self": int(initiator_counter.get("self", 0)),
            "other": int(initiator_counter.get("other", 0)),
        },
        "reply_time_self": summarize_reply(reply_times_self),
        "reply_time_other": summarize_reply(reply_times_other),
        "length_self": summarize_lengths(length_self),
        "length_other": summarize_lengths(length_other),
    }

    return stats


def plot_weekday_hour_heatmap(weekday_hour_counts: Dict[str, Dict[str, int]]):
    # 数据转为 7x24 矩阵
    matrix = np.zeros((7, 24), dtype=int)
    for w in range(7):
        for h in range(24):
            matrix[w, h] = int(weekday_hour_counts.get(str(w), {}).get(f"{h:02d}", 0))

    plt.figure(figsize=(12, 4))
    im = plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(im, label="消息数量")

    plt.yticks(range(7), ["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
    plt.xticks(range(24), [f"{h:02d}" for h in range(24)], rotation=45)
    plt.xlabel("小时")
    plt.ylabel("星期")
    plt.title("消息活跃度热力图（星期 × 小时）")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "weekday_hour_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"星期 × 小时活跃度热力图已保存到: {out_path}")


def plot_daily_heatmap(daily_hour_counts: Dict[str, Dict[str, int]]):
    if not daily_hour_counts:
        print("每日热力图数据为空，跳过绘制。")
        return

    dates = sorted(daily_hour_counts.keys())
    matrix = np.zeros((len(dates), 24), dtype=int)
    for i, date_key in enumerate(dates):
        for h in range(24):
            matrix[i, h] = int(daily_hour_counts[date_key].get(f"{h:02d}", 0))

    plt.figure(figsize=(14, max(4, len(dates) * 0.12)))
    im = plt.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = plt.colorbar(im, fraction=0.035, pad=0.04)
    cbar.set_label("消息数量", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    plt.xticks(range(24), [f"{h:02d}" for h in range(24)], rotation=0, fontsize=16)

    if len(dates) <= 20:
        yticks = range(len(dates))
    else:
        step = max(1, len(dates) // 20)
        yticks = range(0, len(dates), step)
        plt.yticks(yticks, [dates[i] for i in yticks], fontsize=16)

    plt.xlabel("小时", fontsize=16)
    plt.ylabel("日期", fontsize=16)
    plt.title("消息活跃度热力图（日期 × 小时）", fontsize=16)

    plt.tight_layout()

    out_path = OUTPUT_DIR / "daily_hour_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"日期 × 小时热力图已保存到: {out_path}")


def plot_daily_totals(daily_totals: Dict[str, int]):
    if not daily_totals:
        print("每日总量数据为空，跳过绘制。")
        return

    dates = sorted(daily_totals.keys())
    x = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    y = [daily_totals[d] for d in dates]

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker="o", linewidth=1.2, color="#4C72B0")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if len(dates) > 20:
        ax.xaxis.set_major_locator(MaxNLocator(10))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("消息数量")
    plt.title("每日消息总量趋势")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "daily_trend.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"每日消息趋势图已保存到: {out_path}")


def main():
    msgs = load_messages()
    if not msgs:
        print("chat.json 中没有任何消息。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weekday_hour_counts = compute_weekday_hour_counts(msgs)
    daily_hour_counts = compute_daily_hour_counts(msgs)
    daily_totals = {
        date_key: sum(hours.values()) for date_key, hours in daily_hour_counts.items()
    }
    interaction_stats = compute_initiative_and_reply(msgs)

    stats = {
        "weekday_hour_counts": weekday_hour_counts,
        "daily_hour_counts": daily_hour_counts,
        "daily_totals": daily_totals,
        "interaction_stats": interaction_stats,
    }

    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"时间 & 互动统计已保存到: {STATS_PATH}")

    # 画热力图 & 趋势
    plot_weekday_hour_heatmap(weekday_hour_counts)
    plot_daily_heatmap(daily_hour_counts)
    plot_daily_totals(daily_totals)


if __name__ == "__main__":
    main()
