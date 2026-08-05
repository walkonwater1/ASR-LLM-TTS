#pragma once
/**
 * 管线配置结构体
 *
 * 支持两种配置方式：
 *   1. 代码直接赋值 — PipelineConfig cfg; cfg.llm_model = "...";
 *   2. JSON 配置文件 — cfg.load_from_file("config.json");
 *
 * JSON 配置文件格式见项目根目录 config.json。
 *
 * Python 对应: src/config.py → PipelineConfig
 */

#include <string>
#include <vector>

struct PipelineConfig {
    // ── ASR ────────────────────────────────────────
    std::string asr_model_path = "src/third_party/sherpa-onnx/zipformer-ctc-zh";
    std::string asr_model_type = "zipformer_ctc";   // "sense_voice" | "zipformer_ctc"

    // ── LLM ────────────────────────────────────────
    std::string ollama_host   = "http://localhost:11434";
    std::string llm_model     = "qwen2.5:3b";
    std::string system_prompt = "你叫小希，是一个活泼开朗的少女，说话可爱俏皮。回复控制在三句话以内。只使用中文回答，不要用日语或任何其他语言。";

    // ── TTS ────────────────────────────────────────
    int         tts_rate         = 200;                           // espeak 语速 (词/分钟)
    std::string tts_voice        = "cmn+f3";                      // espeak 音色
    std::string tts_backend      = "piper";                       // "espeak" / "piper" / "edge_tts"
    std::string piper_model_path = "~/pretrained_models/piper/zh_CN/zh_CN-xiao_ya-medium.onnx";
    std::string edge_tts_voice   = "zh-CN-XiaoyiNeural";         // Edge TTS 音色

    // ── 唤醒词 ─────────────────────────────────────
    std::string wake_word = "zhan qi lai";   // 空字符串 = 关闭（单选模式，兼容旧配置）
    std::vector<std::string> wake_words = {  // 多唤醒词（中文原文）
        "你好爱秋", "你好小希", "小希小希"
    };
    std::string wake_reply     = "我在";       // 唤醒后简短回应
    int    idle_sleep_seconds  = 90;          // 无交互自动休眠秒数
    std::string sleep_message  = "我去休息啦";  // 休眠提示语

    // ── 声纹验证 ───────────────────────────────────
    std::string sv_enroll_dir = "speaker_voice";
    float       sv_threshold  = 0.35f;       // 相似度阈值 (越低越严格)

    // ── 音频 ───────────────────────────────────────
    int sample_rate = 16000;

    // ── VAD（语音活动检测 / 打断灵敏度）─────────────
    std::string vad_backend          = "energy";   // "energy" 或 "adaptive"
    float vad_energy_threshold   = 0.003f;   // RMS 阈值，越大越不敏感（仅 energy 模式）
    int   vad_min_speech_frames  = 8;        // 最小语音帧数 (~160ms)
    int   vad_min_silence_frames = 20;       // 静音多少帧后判结束 (~400ms)
    int   vad_pre_speech_frames  = 15;       // 语音开始前保留帧数 (~300ms)
    float vad_adaptive_factor    = 3.0f;     // 阈值 = 噪声基线 × factor（仅 adaptive 模式）
    float vad_min_energy         = 0.002f;   // 绝对最小能量阈值（仅 adaptive 模式）
    int   vad_cooldown_frames    = 25;       // 语音段结束后强制静音帧数 (~500ms)

    // ── 交互模式 ───────────────────────────────────
    bool  barge_in_enabled      = true;    // 播放时允许语音打断
    float barge_in_energy_ratio = 1.5f;    // 能量门控：当前帧/回声基线 > 此值 → 启动打断（1.5=较灵敏）
    int   max_response_chars    = 80;      // 回复最大字数（超出截断），0=不限制
    bool  barge_in_semantic     = true;    // 语义打断：ASR识别到打断意图短语才打断（false=纯能量打断）
    std::vector<std::string> barge_in_phrases = {
        "别说了", "不想听", "不要说了", "停下", "停",
        "闭嘴", "够了", "别念了", "别讲了", "别吵了",
        "我不想听了", "别再说了", "可以停了", "别说了行不行",
        "好了好了", "行了行了", "知道了知道了"
    };

    // ── 对话记忆 ───────────────────────────────────
    int max_rounds = 10;
    int max_tokens = 2048;
    std::string memory_persist_dir = "data/memory";  // 持久化目录
    bool memory_long_term_enabled  = true;            // 长期记忆开关
    bool memory_auto_extract       = true;            // 自动提取个人信息

    // ── 技能 ───────────────────────────────────────
    bool skill_weather    = true;
    bool skill_time       = true;
    bool skill_web_search = false;
    bool skill_rag        = false;
    std::string rag_docs_dir = "knowledge_base";

    // ── Function Calling (LLM 驱动工具选择) ──────────
    bool fc_enabled     = true;            // 启用 function calling
    std::string fc_model = "";             // 工具选择模型（空=复用 llm_model）

    // ── ReAct (多步推理) ──────────────────────────────
    bool react_enabled  = true;            // 启用 ReAct 多步推理
    int  react_max_steps = 5;              // 最大推理步数

    // ── Reflection (自我反思) ─────────────────────────
    bool reflect_enabled = true;           // 启用回复后反思修正
    std::string reflect_model = "";        // 反思模型（空=复用 llm_model）

    // ── Multi-Agent (双Agent协作) ─────────────────────
    bool multi_agent_enabled = true;       // 启用双 Agent 协作优化
    std::string ma_critic_model = "";      // Critic 模型（空=复用 llm_model）
    int  ma_max_rounds = 2;               // 最大协作轮数

    // ── 流式 ASR ──────────────────────────────────────
    bool   streaming_asr_enabled = true;            // 启用流式 ASR
    std::string streaming_asr_backend = "chunked";  // "online" | "chunked"
    std::string streaming_asr_model  = "";          // online 模型路径（空=复用 asr_model_path）
    float  streaming_min_chunk  = 0.8f;             // chunked: 最小触发长度 (秒)
    float  streaming_chunk_intv = 0.5f;             // chunked: 部分识别间隔 (秒)
    int    asr_endpoint_punct_frames   = 15;        // 句末标点后稳定帧数→端点 (20ms/帧, 15=300ms)
    int    asr_endpoint_nopunct_frames = 30;        // 无标点稳定帧数→端点 (30=600ms)

    // ── Embedding (Layer 4.2) ──────────────────────────
    std::string embedding_backend   = "ollama";     // "ollama" | "onnx"
    std::string embedding_model_dir = "models/embedding";  // ONNX 模型目录

    // ── 文件加载 ───────────────────────────────────

    /// 从 JSON 文件加载配置（未出现在文件中的键保持默认值）
    /// @return true 加载成功，false 文件不存在或格式错误（回退到默认值）
    bool load_from_file(const std::string& path);

    /// 智能查找配置文件：当前目录 → 上级目录 → 环境变量
    /// @return 实际加载的文件路径，空 = 未找到（使用纯默认值）
    static std::string auto_load_path();
};
