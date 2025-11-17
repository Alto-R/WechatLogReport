# WechatLogReport

一个帮助你把微信聊天记录跑通「数据统计 → LLM 深度分析 → 报告输出」的脚本集合。它配套了本仓库下的 Windows 导出工具 `chatlog.exe`，以及位于 `analysis/` 目录中的多段 Python 分析脚本，最终会在 `analysis_output/` 写出一整套数据文件、可视化图片与 Markdown 报告，方便你复盘一段关系的沟通模式。

## 目录概览

| 位置 | 说明 |
| --- | --- |
| `chatlog.exe` | 启动一个本地 HTTP 服务，将 PC 版微信的聊天记录暴露为 `http://127.0.0.1:5030` （可自定义）的 API。 |
| `analysis/getdata.py` | 调用上面的 API，将指定好友 + 时间范围的聊天导出成 JSON。 |
| `analysis/calc_chat_stats.py` | 词频、活跃时段、词云等基础统计。 |
| `analysis/interaction_time_analysis.py` | 会话划分、回复速度、活跃度热力图。 |
| `analysis/llm_analysis.py` | 依赖 SiliconFlow，大模型分段阅读聊天，抽取话题、情绪、沟通模式。 |
| `analysis/visualize_llm.py` | 根据 `llm_analysis.json` 画出话题、情绪等图表。 |
| `analysis/llm_profile_story.py` | 用 LLM 写双方人物画像 & 回忆录式故事。 |
| `analysis/final_report.py` | 汇总所有统计与 LLM 结果，生成 `analysis_output/final_report.md`。 |

## 环境准备

1. **操作系统**：`chatlog.exe` 仅在 Windows 下可用；Python 脚本可在任意系统运行，只要能访问同一份导出的 JSON。
2. **Python**：建议 3.11+。
3. **依赖**：项目没有 `requirements.txt`，可以手动安装：

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -U pip
   pip install requests matplotlib numpy wordcloud pillow jieba
   ```

   - 若 `wordcloud` 无法找到中文字体，请在 `analysis/calc_chat_stats.py` 里修改 `FONT_PATH_CANDIDATES`。
   - 使用 LLM 前需要在 `analysis/llm_analysis.py` 把 `API_KEY` 设置为你自己的 SiliconFlow key，并确保模型 `Pro/deepseek-ai/DeepSeek-V3.2-Exp` 对应的配额充足。
4. **微信版本**：微信4.0及以上的版本已经不能用这种方法解码聊天记录了，因此目前只能下载3.9的版本使用，可在`https://github.com/tom-snow/wechat-windows-versions/releases`下载到。

## 准备聊天数据

1. **聊天记录导入**  
   用手机微信将聊天记录导出到电脑端3.9版本微信。可选择联系人或群聊，以及要导出的时间范围。

2. **启动导出器**  
   双击 `chatlog.exe`，它会打开一个监听在 `http://127.0.0.1:5030` 的本地服务。PC 版微信保持登陆，这样导出器就能读取到聊天记录。

3. **配置导出参数**  
   修改 `analysis/getdata.py` 中的 `params`，主要是 `time`（`YYYY-MM-DD~YYYY-MM-DD`）和 `talker`（好友的微信 ID）。如果不确定 ID，可以先用导出器的 UI 查询。

4. **导出 JSON**  
   ```powershell
   python analysis/getdata.py
   ```
   命令会把数据写到 `analysis_output/chat.json`。后续脚本默认从 `analysis_output/bb_chat.json` 读取，所以请把文件改名或复制一份：
   ```powershell
   Rename-Item analysis_output/chat.json bb_chat.json
   ```
若你已经有现成的聊天 JSON，只需放到 `analysis_output/bb_chat.json` 即可。

### 关于 `chatlog.exe` 的详细使用流程

以下步骤节选并凝练自 [sjzar/chatlog](https://github.com/sjzar/chatlog) 官方教程，方便你结合本项目快速跑通数据：

1. **下载 / 安装**  
   - Windows 用户可直接使用仓库中附带的 `chatlog.exe`。  
   - 想自己折腾的可以找github上别人对sjzar的库的备份。
2. **运行与解密**  
   - 直接双击 exe 或在 Windows Terminal 中执行 `chatlog` 进入 TUI，依次选择“获取密钥”“解密数据”“开启 HTTP 服务”。  
   - 偏好命令行的用户可使用子命令：`chatlog key`（拉取密钥）、`chatlog decrypt`（解密数据库）、`chatlog server`（启动 API）。
3. **访问接口**  
   - 核心聊天记录 API：`GET /api/v1/chatlog?time=2025-01-01~2025-11-17&talker=wxid_xxx&format=json`。  
   - 还可调用 `/api/v1/contact`、`/api/v1/chatroom`、`/api/v1/session` 查询联系人/群聊/最近会话，或通过 `/image/<id>`、`/voice/<id>` 等路径下载多媒体。

参数说明：
- `time`: 时间范围，格式为 `YYYY-MM-DD` 或 `YYYY-MM-DD~YYYY-MM-DD`
- `talker`: 聊天对象标识（支持 wxid、群聊 ID、备注名、昵称等）
- `limit`: 返回记录数量
- `offset`: 分页偏移量
- `format`: 输出格式，支持 `json`、`csv` 或纯文本

完成上述准备并确认 `http://127.0.0.1:5030` 可访问后，就可以运行本 README 其余脚本，产出统计与报告。

## 分析流程

按顺序执行以下脚本即可逐步得到所有统计与报告，每一步都会把结果写入 `analysis_output/`（不存在会自动创建）。

1. **基础统计 & 词云**
   ```powershell
   python analysis/calc_chat_stats.py
   ```
   输出 `chat_stats.json` 以及 `wordcloud_total.png` / `wordcloud_self.png` / `wordcloud_other.png`。

2. **时间轴与互动模式**
   ```powershell
   python analysis/interaction_time_analysis.py
   ```
   输出 `interaction_time_stats.json`，并生成 `weekday_hour_heatmap.png`、`daily_hour_heatmap.png`、`daily_trend.png`。

3. **大模型分段分析**  
   在 `analysis/llm_analysis.py` 中填好 `API_KEY`，必要时调整 `CHUNK_SIZE`（每次投喂多少条消息）与 `DEFAULT_SYSTEM_PROMPT`。然后运行：
   ```powershell
   python analysis/llm_analysis.py
   ```
   该脚本会：
   - 按时间切分对话并逐段请求 SiliconFlow；
   - 途中若中断，会把临时结果写到 `analysis_output/llm_analysis_partial.json`，下次可继续；
   - 最终输出 `llm_analysis.json`，包含话题分布、情绪时间线、沟通模式、时间切片等结构化数据。

4. **可视化 LLM 结果**
   ```powershell
   python analysis/visualize_llm.py
   ```
   生成 `topics_distribution.png`、`communication_patterns.png`、`emotion_timeline.png`。

5. **人物画像 & 年度故事（可选）**
   ```powershell
   python analysis/llm_profile_story.py
   ```
   复用 `llm_analysis.py` 中设置的 API Key，从 `chat_stats.json` + `llm_analysis.json` 抽取关键信息，生成：
   - `profile_self.txt`
   - `profile_other.txt`
   - `story.md`

6. **整合终极报告**
   ```powershell
   python analysis/final_report.py
   ```
   读取前面所有 JSON/文本，拼出 Markdown 版《微信关系年度终极报告》，路径为 `analysis_output/final_report.md`。

## 输出清单

| 文件 | 来源脚本 | 内容 |
| --- | --- | --- |
| `bb_chat.json` | `getdata.py` 或手动 | 原始聊天数据（数组，每条消息包含 `time` `isSelf` `content` 等）。 |
| `chat_stats.json` | `calc_chat_stats.py` | 基础统计、词频排行、活跃时段。 |
| `wordcloud_*.png` | `calc_chat_stats.py` | 总体 / 我 / 对方的词云。 |
| `interaction_time_stats.json` | `interaction_time_analysis.py` | 会话划分、回复速度、消息长度分布。 |
| `weekday_hour_heatmap.png` 等 | `interaction_time_analysis.py` | 时段热力图 & 日趋势。 |
| `llm_analysis.json` | `llm_analysis.py` | 大模型对每段对话的总结、话题、情绪、建议。 |
| `topics_distribution.png` 等 | `visualize_llm.py` | LLM 结果的可视化图表。 |
| `profile_self.txt` / `profile_other.txt` / `story.md` | `llm_profile_story.py` | 人物画像与回忆录式故事。 |
| `final_report.md` | `final_report.py` | 汇总所有结论的终极 Markdown 报告。 |

## 小贴士

- 如果想节省 API 费用，可减小 `CHUNK_SIZE` 或先在脚本里过滤掉无意义的消息（例如系统提示）。
- 每次更新 `chat.json` 后建议重新跑所有脚本，保证 `final_report.md` 的数据一致。
- `analysis_output/` 默认未被 git 追踪，可以放心地把生成的隐私数据留在本地。
- 若 `analysis/` 中的脚本需要自定义（比如改 prompt、增加图表），可以直接修改后再运行，无需重装依赖。

> ℹ️ 本仓库附带的 `chatlog.exe` 源自 [@sjzar](https://github.com/sjzar) 维护的开源项目 [sjzar/chatlog](https://github.com/sjzar/chatlog)。该项目提供跨平台的微信聊天记录导出、解密与 HTTP API 服务，以下关于 exe 的使用流程也参考了其官方文档整理而来，特此致谢。若需要详细的指南请自行寻找原库的readme教程。
