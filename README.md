# 🎙️ 小希 — 全链路本地语音 AI 助手

**13,000+ 行 C++17，76 个单元测试，从麦克风采集到语音合成，完整的本地语音交互系统。**

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Voice Pipeline                             │
│                                                                  │
│  ┌──────────┐   ┌──────┐   ┌────────┐   ┌──────────┐           │
│  │ 麦克风 ──┼──→│ VAD  ┼──→│ 声纹识别┼──→│ 情感分析 │           │
│  │ arecord  │   │自适应│   │CAM++   │   │音高/语速 │           │
│  └──────────┘   └──┬───┘   └───┬────┘   └────┬─────┘           │
│                    │            │              │                 │
│  Capture 线程 ═════╪════════════╪══════════════╪══════           │
│  Process 线程 ═════╪════════════╪══════════════╪══════           │
│                    ↓            ↓              ↓                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐        │
│  │ 流式 ASR  │   │ LLM 推理 │   │   14 个 Skill 技能    │        │
│  │Zipformer │   │ qwen2.5  │   │ 时间|计算|天气|笔记|  │        │
│  │chunk 0.5s│   │  + RAG   │   │ 提醒|诗词|笑话|占卜|  │        │
│  └──────────┘   └─────┬────┘   │ 谜语|游戏|搜索|RAG   │        │
│                        ↓        └──────────┬───────────┘        │
│                  ┌─────────────────────────────────┐            │
│                  │   关键字匹配调度（确定性路由）     │            │
│                  │   内容型技能 direct 模式交付      │            │
│                  └───────────────┬─────────────────┘            │
│                                  ↓                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│  │ TTS 播放 │←──│ 韵律融合 │←──│ 回复生成 │                   │
│  │edge_tts  │   │ 情感+语速│   │          │                   │
│  │/Piper    │   └──────────┘   └──────────┘                   │
│  └──────────┘                                                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  长期记忆 (JSON持久化) │ 声纹库 (多用户) │ WebSocket  │       │
│  │  热配置重载 │ 可打断交互 │ 垃圾文本过滤   │  音频推流  │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

## 核心特性

### 🎤 语音管线
| 模块 | 技术 | 说明 |
|------|------|------|
| **ASR** | sherpa-onnx Zipformer CTC | 中文专用，INT8量化，~50ms延迟 |
| **流式ASR** | 分块增量识别 | 0.5s间隔部分结果，边说边出字 |
| **VAD** | 自适应能量阈值 | 自动学习环境噪声基线，支持ASR文本稳定性端点 |
| **声纹** | 3D-Speaker CAM++ | 多用户注册/识别，dim=192 |
| **唤醒词** | 拼音匹配 | 可选，支持400+常用汉字 |
| **情感** | 声学特征分析 | 实时音高/语速/能量 → 融合TTS韵律 |
| **TTS** | edge_tts / Piper + espeak-ng | 多后端可切换，情感驱动韵律 |

### 🧠 智能推理
| 模块 | 技术 | 说明 |
|------|------|------|
| **LLM** | Ollama + qwen2.5:3b | 本地CPU推理，~660ms首响应 |
| **RAG** | 本地ONNX Embedding | 文档语义检索增强生成 |
| **FunctionCalling** | LLM驱动函数调用 | 可选（默认关闭），小模型JSON输出不稳定 |
| **ReAct** | 多步推理 | 可选（默认关闭），LLM自主调用工具链 |
| **Reflection** | 自我反思 | 可选（默认关闭），LLM审视修正回复 |
| **Multi-Agent** | Generator+Critic | 可选（默认关闭），双模型协作 |

### 🔧 14个技能
```
实用类:  时间 · 天气 · 计算器 · 网页搜索 · RAG知识库
生产类:  提醒(后台fork进程) · 笔记(JSON CRUD) · 系统状态(CPU/内存/磁盘)
娱乐类:  笑话 · 故事 · 冷知识 · 毒鸡汤 · 唐诗宋词 · 谜语 · 占卜(星座/塔罗/黄历) · 猜数字/成语接龙
```

**关键字匹配调度**：使用确定性关键字路由技能，避免小模型 Function Calling 输出不稳定。内容型技能(诗词/笑话/故事)使用 `direct` 模式直接交付，绕过LLM避免小模型截断。

### 🏗️ 工程特性
- **多线程交互**：Capture线程持续录音 + Process线程异步推理，支持语音打断
- **双端点检测**：VAD静音 + ASR文本稳定性(1.5s不变→判定说完)，解决噪声下VAD不休眠问题
- **垃圾ASR过滤**：`is_garbage_text()` 检测日文假名/韩文/短英文/纯标点
- **热配置重载**：修改config.json自动热应用VAD/LLM/TTS参数
- **长期记忆**：对话历史+用户记忆JSON持久化，重启不丢
- **WebSocket服务**：支持浏览器/APP客户端音频推流
- **多用户声纹库**：识别说话人→自动切换该用户的system prompt

## 性能基准

```
测试环境: Intel CPU, 4线程, 全本地推理

ASR (Zipformer CTC INT8, 351MB):     ~50 ms/段      (实时率 0.05x)
LLM (qwen2.5:3b, 2GB):               ~660 ms/短回复
TTS (edge_tts/Piper):                 ~480 ms/短句

端到端延迟:
  短交互 (你好→回复):   ~1.5s  (ASR+LLM+TTS)
  正常对话 (20字→30字):  ~3-6s  (流式ASR与LLM重叠)
  长内容 (诗词/故事):    ~1s    (direct模式跳过LLM)
```

## 测试覆盖

| 测试套件 | 用例数 | 覆盖 |
|----------|--------|------|
| test_utf8 | 33 | 垃圾文本检测(中/日/韩/英/标点/空白) |
| test_vad | 7 | VAD状态机(静音/语音开始/结束/自适应/冷却) |
| test_skills | 36 | 技能匹配+执行(时间/计算/诗词/娱乐/占卜/谜语/游戏) |

```bash
./build/test_utf8 && ./build/test_vad && ./build/test_skills
./build/benchmark test_recording.wav  # 性能基准
```

## 快速开始

### 依赖

```bash
sudo apt install libcurl4-openssl-dev libboost-system-dev libspdlog-dev
sudo apt install espeak-ng libespeak-ng-dev nlohmann-json3-dev

# Ollama (LLM服务)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

sherpa-onnx 库已内置在 `src/third_party/sherpa-onnx/`，模型需单独下载：

```bash
# Zipformer CTC 中文ASR模型 (推荐)
mkdir -p src/third_party/sherpa-onnx/zipformer-ctc-zh
wget https://hf-mirror.com/csukuangfj/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/resolve/main/model.int8.onnx \
     -O src/third_party/sherpa-onnx/zipformer-ctc-zh/model.int8.onnx
wget https://hf-mirror.com/csukuangfj/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/resolve/main/tokens.txt \
     -O src/third_party/sherpa-onnx/zipformer-ctc-zh/tokens.txt
```

### 编译运行

```bash
cd src && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 回到项目根目录运行
cd ../..
./build/voice_pipeline
```

### 交互模式

| 输入 | 功能 |
|------|------|
| 任意文字 | 直接和 LLM 对话 |
| `listen` | 交互模式（持续监听 + 语音打断） |
| `enroll` | 注册声纹 |
| `clear` | 清空对话记忆 |
| `quit` | 退出 |

## 目录结构

```
src/
├── pipeline/     # 管线编排 (VoicePipeline)
├── speech/       # ASR · 流式ASR · TTS · 声纹 · 唤醒词 · 情感
├── brain/        # 技能系统 (SkillManager + 14个技能)
├── llm/          # LLM引擎 · FunctionCalling · ReAct · Reflection · MultiAgent
├── memory/       # 对话记忆 · 用户长期记忆 · 文档分块
├── audio/        # 音频IO · VAD · 降噪
├── server/       # WebSocket 语音服务
└── utils/        # UTF8工具 · 配置热重载 · 日志
```

## 配置

`config.json` 支持热重载（修改后自动生效）：

```json
{
  "asr": {
    "model_path": "src/third_party/sherpa-onnx/zipformer-ctc-zh",
    "model_type": "zipformer_ctc"
  },
  "llm": { "host": "http://localhost:11434", "model": "qwen2.5:3b" },
  "tts": { "backend": "edge_tts", "rate": 200 },
  "vad": { "backend": "adaptive", "adaptive_factor": 7.0, "min_energy": 0.025 }
}
```

> **实验性功能**：`function_calling`、`react`、`reflection`、`multi_agent` 默认关闭，详见 `config.json` 中的开关。

### 多机部署

LLM 可与 ASR/TTS 分离部署：

```
┌─ 客户端 ──────────────────────────┐      ┌─ 服务器 ──────────────┐
│  ASR (Zipformer CTC)              │      │                       │
│  TTS (edge_tts / Piper)           │ LAN  │  Ollama (Qwen2.5)    │
│  VAD + 声纹 + 情感                │─────▶│  HTTP API :11434     │
│  VoicePipeline (管线编排)          │      │                       │
└───────────────────────────────────┘      └───────────────────────┘
```

修改 `config.json` 中 `llm.host` 指向服务器 IP 即可。

## 设计决策与权衡

1. **为什么 ASR 从 SenseVoice 换到 Zipformer CTC？**
   SenseVoice 在噪声下严重幻觉(输出日/韩/英乱码)。Zipformer CTC 是中文专用CTC模型，解码天然低幻觉，accuracy更高。

2. **为什么用 qwen2.5:3b 而不是更大的模型？**
   CPU本地推理的延迟/内存权衡。通过关键字兜底调度和`direct`交付模式弥补小模型在Function Calling和长文本上的不足。

3. **为什么技能调度使用关键字匹配而非纯 Function Calling？**
   小模型Function Calling输出不稳定(畸形JSON)。关键字匹配作为确定性方案确保技能始终可用。Function Calling 作为可选增强保留。

4. **为什么 VAD + ASR稳定性双重端点检测？**
   纯VAD在持续噪声中无法检测静音。ASR文本稳定性检测作为互补——文本1.5s不变即判定说完，1.0s if 以。！？结尾。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 语言 | C++17 | 全项目~100个文件 |
| 构建 | CMake | 3.16+ |
| ASR | sherpa-onnx C API v1.13+ | ONNX Runtime |
| LLM | Ollama HTTP API | OpenAI兼容接口 |
| TTS | edge_tts / Piper + espeak-ng | 多后端可切换 |
| 声纹 | CAM++ | sherpa-onnx speaker embedding |
| Embedding | BERT ONNX | 本地512维向量 |
| JSON | nlohmann/json | header-only |
| 日志 | spdlog | header-only |
| HTTP | libcurl | 系统库 |

## 未来方向

### 本地小模型 + 云端大模型混合推理

当前 Demo 使用单一本地模型 (`qwen2.5:3b`) 处理全部请求，在低延迟和复杂推理之间存在天然矛盾。下一步架构演进方向：

```
┌─ 客户端 (本地) ──────────────────────┐      ┌─ 云端 ───────────────┐
│                                       │      │                       │
│  用户语音 ──→ 意图路由器 ──→ 简单任务 ──→ 本地小模型 (qwen2.5:3b)   │
│                    │                  │      │                       │
│                    └──→ 复杂任务 ──→ HTTP ──→ 云端大模型 (Qwen3-235B)│
│                                       │      │                       │
│  响应延迟:  简单 ~1s / 复杂 ~3-5s     │      │  响应延迟: ~0.5-2s    │
└───────────────────────────────────────┘      └───────────────────────┘
```

**核心思路**：绝大多数日常对话（问候、闲聊、简单问答）由本地小模型在 ~1s 内直接响应；仅当意图识别判定需要深度推理（多步计算、复杂规划、知识综合）时，才路由到云端大模型。

**关键设计点**：
1. **意图路由** — ASR 文本首先经过轻量分类器（关键词 + embedding 相似度），决定走本地还是云端。不需要 LLM 参与路由决策，零额外延迟。
2. **本地优先** — 默认走本地，云端仅在本地无法胜任时作为 fallback，而非反过来。保证离线可用性。
3. **响应融合** — 云端返回的结果可选择性缓存到本地知识库 (RAG)，逐渐减少对云端的依赖。
4. **成本控制** — 预估 90%+ 请求落在本地，云端调用量远低于全云端方案。

**对比纯本地 vs 纯云端方案**：

| 维度 | 纯本地 (当前) | 纯云端 | 混合架构 |
|------|:---:|:---:|:---:|
| 简单回复延迟 | ~1s ✅ | ~2-5s ❌ | ~1s ✅ |
| 复杂推理能力 | 弱 ❌ | 强 ✅ | 强 ✅ |
| 离线可用 | ✅ | ❌ | ✅ (降级) |
| 隐私保护 | ✅ | ❌ | ✅ (90%+) |
| 云端成本 | 零 | 高 | 低 |
| 部署复杂度 | 低 | 低 | 中 |

## License

MIT
