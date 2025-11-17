import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置字体为SimHei以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
LLM_PATH = OUTPUT_DIR / "llm_analysis.json"


def load_llm_analysis():
    if not LLM_PATH.exists():
        raise FileNotFoundError(f"llm_analysis.json not found at: {LLM_PATH}")

    with LLM_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_topic_distribution(topic_distribution: dict):
    if not topic_distribution:
        print("topic_distribution 为空，跳过话题分布可视化。")
        return

    topics = list(topic_distribution.keys())
    counts = list(topic_distribution.values())

    plt.figure(figsize=(36, 6))
    plt.bar(range(len(topics)), counts, color="#4C72B0")
    plt.xticks(range(len(topics)), topics, rotation=45, ha="right", fontsize=9)
    plt.ylabel("出现次数")
    plt.title("话题分布")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "topics_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"话题分布图已保存到: {out_path}")


def plot_communication_patterns(comm_patterns: dict):
    if not comm_patterns:
        print("communication_patterns 为空，跳过沟通模式可视化。")
        return

    patterns = list(comm_patterns.keys())
    counts = list(comm_patterns.values())

    plt.figure(figsize=(16, 6))
    plt.bar(range(len(patterns)), counts, color="#55A868")
    plt.xticks(range(len(patterns)), patterns, rotation=45, ha="right", fontsize=9)
    plt.ylabel("出现次数")
    plt.title("沟通模式分布")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "communication_patterns.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"沟通模式分布图已保存到: {out_path}")


def plot_emotion_timeline(emotion_timeline: list):
    if not emotion_timeline:
        print("emotion_timeline 为空，跳过情绪时间线可视化。")
        return

    # 收集所有出现过的情绪标签
    emotions_self = [e.get("self_emotion") for e in emotion_timeline if e.get("self_emotion")]
    emotions_other = [e.get("other_emotion") for e in emotion_timeline if e.get("other_emotion")]
    unique_emotions = sorted(set(emotions_self + emotions_other))
    if not unique_emotions:
        print("情绪标签为空，跳过情绪时间线可视化。")
        return

    emotion_to_y = {emo: idx for idx, emo in enumerate(unique_emotions)}

    # 尝试使用时间作为 X 轴
    def parse_start(entry):
        tr = entry.get("time_range") or {}
        start = tr.get("start")
        if not start:
            return None
        try:
            return datetime.fromisoformat(start)
        except Exception:
            return None

    start_times = [parse_start(e) for e in emotion_timeline]
    use_time_axis = any(start_times)

    if use_time_axis:
        x_vals = [
            mdates.date2num(dt) if dt else idx
            for idx, dt in enumerate(start_times, start=1)
        ]
        x_label = "起始时间"
    else:
        x_vals = [e.get("chunk_index") for e in emotion_timeline]
        x_label = "段落序号（chunk_index）"

    y_self = [emotion_to_y.get(e.get("self_emotion")) for e in emotion_timeline]
    y_other = [emotion_to_y.get(e.get("other_emotion")) for e in emotion_timeline]

    plt.figure(figsize=(18, 6))

    plt.plot(x_vals, y_self, label="我", marker="o", color="#4C72B0")
    plt.plot(x_vals, y_other, label="Ta", marker="x", color="#DD8452")

    if use_time_axis:
        ax = plt.gca()
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.xticks(rotation=30, ha="right")

    plt.yticks(range(len(unique_emotions)), unique_emotions)
    plt.xlabel(x_label)
    plt.ylabel("情绪标签")
    plt.title("情绪时间线（按时间排列）" if use_time_axis else "情绪时间线（按段落）")
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "emotion_timeline.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"情绪时间线图已保存到: {out_path}")


def main():
    data = load_llm_analysis()

    topic_distribution = data.get("topic_distribution") or {}
    emotion_timeline = data.get("emotion_timeline") or []
    communication_patterns = data.get("communication_patterns") or {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_topic_distribution(topic_distribution)
    plot_communication_patterns(communication_patterns)
    plot_emotion_timeline(emotion_timeline)


if __name__ == "__main__":
    main()
