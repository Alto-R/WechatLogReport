import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
CHAT_PATH = OUTPUT_DIR / "chat.json"
OUTPUT_PATH = OUTPUT_DIR / "chat_stats.json"


def load_messages(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"chat.json not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("chat.json format error: expected a list of messages")

    return data


def parse_hour(ts: str):
    """Parse ISO time string and return hour (0-23)."""
    try:
        # Python 3.11+ can parse "2025-02-14T00:49:32+08:00" directly
        dt = datetime.fromisoformat(ts)
        return dt.hour
    except Exception:
        return None


def tokenize(content: str):
    """Tokenize message content.

    - If jieba is available, use word segmentation.
    - Otherwise, fall back to character-level tokens.
    """
    content = content.strip()
    if not content:
        return []

    # Skip XML / structured payloads
    if content.lstrip().startswith("<?xml"):
        return []

    try:
        import jieba  # type: ignore

        tokens = jieba.lcut(content)
    except Exception:
        tokens = list(content)

    return tokens


PUNCTUATION = set(
    " \t\r\n0123456789"
    ".,!?;:，。？！；：、"
    "()[]{}（）【】<>《》"
    "\"'“”‘’"
)

STOPWORDS = {
    "哦哦",
    "啊啊",
    '感觉',
    '什么',
    '这个',
    '没有',
    '还是',
    '怎么',
    '可以',
    '旺柴',
    '偷笑',
    '捂脸',
    'Rhapsody',
    'Vent',
    '我们',
    '你们',
}

# 有些单字在对话里比较有信息量，可以保留
ALLOWED_SINGLE_CHARS = {
    "爱",
    "累",
    "困",
    "忙",
    "好",
    "爽",
    "惨",
    "晕",
    "渴",
    "穷",
}


def is_meaningful(token: str):
    if not token:
        return False

    token = token.strip()
    if not token:
        return False

    if token in STOPWORDS:
        return False

    # 丢掉大部分单个字，只保留少数主动允许的
    if len(token) == 1 and token not in ALLOWED_SINGLE_CHARS:
        return False

    # Filter pure punctuation / digits
    if all(ch in PUNCTUATION for ch in token):
        return False

    # Simple emoji / symbol heuristic
    if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
        return False

    return True


def compute_stats(messages):
    hour_counter = Counter()
    keyword_counter = Counter()
    keyword_counter_self = Counter()
    keyword_counter_other = Counter()

    for msg in messages:
        ts = msg.get("time")
        if isinstance(ts, str):
            hour = parse_hour(ts)
            if hour is not None:
                hour_counter[hour] += 1

        # Only consider text messages for keywords
        if msg.get("type") != 1:
            continue

        content = msg.get("content") or ""
        tokens = tokenize(content)
        for tok in tokens:
            if not is_meaningful(tok):
                continue
            keyword_counter[tok] += 1
            if msg.get("isSelf"):
                keyword_counter_self[tok] += 1
            else:
                keyword_counter_other[tok] += 1

    # Hourly stats: ensure all 24 hours exist
    hourly_counts = {f"{h:02d}": int(hour_counter.get(h, 0)) for h in range(24)}

    favorite_hours = []
    if hour_counter:
        max_count = max(hour_counter.values())
        favorite_hours = [h for h, c in hour_counter.items() if c == max_count]

    favorite_hours_readable = [
        f"{h:02d}:00-{h:02d}:59" for h in sorted(favorite_hours)
    ]

    top_keywords = [[w, int(c)] for w, c in keyword_counter.most_common(200)]
    top_keywords_self = [[w, int(c)] for w, c in keyword_counter_self.most_common(200)]
    top_keywords_other = [[w, int(c)] for w, c in keyword_counter_other.most_common(200)]

    stats = {
        "total_messages": len(messages),
        "hourly_counts": hourly_counts,
        "favorite_hours": favorite_hours,
        "favorite_hours_readable": favorite_hours_readable,
        "keywords_top": top_keywords,
        "keywords_top_self": top_keywords_self,
        "keywords_top_other": top_keywords_other,
    }

    return stats


FONT_PATH_CANDIDATES = [
    # Windows 常见中文字体
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
]


def find_font_path():
    for p in FONT_PATH_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    return None
    

def generate_wordcloud_for(freqs, suffix: str):
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        print("未安装 wordcloud，跳过词云图片生成。可使用: pip install wordcloud")
        return

    font_path = find_font_path()
    if not font_path:
        print("未找到合适的中文字体文件，跳过词云图片生成。")
        print("请在 calc_chat_stats.py 中配置 FONT_PATH_CANDIDATES。")
        return

    if not freqs:
        print(f"{suffix} 关键词列表为空，跳过对应词云生成。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_path = OUTPUT_DIR / f"wordcloud_{suffix}.png"

    wc = WordCloud(
        font_path=str(font_path),
        width=1200,
        height=800,
        background_color="white",
        max_words=200,
    )
    wc.generate_from_frequencies(freqs)
    wc.to_file(str(img_path))

    print(f"词云图片已保存到: {img_path}")


def main():
    messages = load_messages(CHAT_PATH)
    stats = compute_stats(messages)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计完成，共 {stats['total_messages']} 条消息")
    print(f"结果已保存到: {OUTPUT_PATH}")
    
    # 生成三张词云图片：总、自己、对方
    total_freqs = {w: c for w, c in stats.get("keywords_top", []) if isinstance(w, str)}
    self_freqs = {w: c for w, c in stats.get("keywords_top_self", []) if isinstance(w, str)}
    other_freqs = {w: c for w, c in stats.get("keywords_top_other", []) if isinstance(w, str)}

    generate_wordcloud_for(total_freqs, "total")
    generate_wordcloud_for(self_freqs, "self")
    generate_wordcloud_for(other_freqs, "other")


if __name__ == "__main__":
    main()
