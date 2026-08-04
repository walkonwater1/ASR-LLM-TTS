/**
 * 语音交互管线编排器
 *
 * Python 对应: src/pipeline.py → VoicePipeline
 *
 * 数据流:
 *   单次模式:
 *     文字:   prompt ──→ LLM ──→ TTS ──→ 播放 → 更新记忆
 *     语音:   麦克风 ──→ ASR ──→ KWS ──→ SV ──→ LLM ──→ TTS ──→ 播放
 *
 *   交互模式:
 *     Capture 线程:  arecord → VAD → 语音段 → queue
 *     Process 线程:  queue → ASR → KWS → LLM → TTS → 异步播放
 *                    新语音到达时 → 打断播放 + generation++ → 旧结果丢弃
 */

#include "voice_pipeline.h"
#include "denoiser.h"
#include "wav_utils.h"
#include "utf8_utils.h"
#include "skill_memory.h"

#include <iostream>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <fstream>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include "logger.h"

// ── 辅助：去掉 LLM 回复中的 emoji 表情（TTS 无法发音）──

static std::string strip_emoji(const std::string& text)
{
    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_bytes = 1;
        if (c >= 0xF0 && i + 3 < text.size()) char_bytes = 4;
        else if (c >= 0xE0 && i + 2 < text.size()) char_bytes = 3;
        else if (c >= 0xC0 && i + 1 < text.size()) char_bytes = 2;

        bool keep = false;
        if (char_bytes == 1) {
            // ASCII: 保留可打印字符和换行
            keep = (c >= 0x20 && c <= 0x7E) || c == '\n' || c == '\r';
        } else if (char_bytes == 3) {
            // 只保留 CJK 汉字和中文标点
            unsigned char b0 = c, b1 = text[i+1], b2 = text[i+2];
            int cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
            // CJK Unified + Extension A + Compatibility
            if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
                (cp >= 0x3400 && cp <= 0x4DBF) ||
                (cp >= 0xF900 && cp <= 0xFAFF)) {
                keep = true;
            }
            // 中文标点：。，！？；：、（）
            if (b0 == 0xE3 && b1 == 0x80) keep = true;  // U+3000–U+303F
            if (b0 == 0xEF && b1 == 0xBC) keep = true;  // U+FF00–U+FFEF（全角标点）
        } else if (char_bytes == 2) {
            keep = true;  // 2字节字符罕见，保留
        }
        // 4字节字符（emoji、特殊符号）全部丢弃

        if (keep) result.append(text, i, char_bytes);
        i += char_bytes;
    }
    return result;
}

// ── 辅助：构建增强的 extra_context（注入用户长期记忆）─

static std::string build_extra_context(
    const std::string& base_extra,
    const UserMemoryStore& user_mem,
    bool long_term_enabled)
{
    std::string extra = base_extra;

    if (long_term_enabled) {
        std::string user_ctx = user_mem.get_all_as_context();
        if (!user_ctx.empty()) {
            // 用户记忆放在最前面，确保 LLM 优先知道
            extra = user_ctx + "\n" + extra;
        }
    }

    return extra;
}

// ── 构造 / 初始化 ────────────────────────────────────

static AsrModelType parse_asr_model_type(const std::string& s)
{
    if (s == "zipformer_ctc") return AsrModelType::ZIPFORMER_CTC;
    return AsrModelType::SENSE_VOICE;  // 默认
}

VoicePipeline::VoicePipeline(const PipelineConfig& cfg)
    : cfg_(cfg)
    , asr_(cfg.asr_model_path, parse_asr_model_type(cfg.asr_model_type))
    , llm_(cfg.ollama_host, cfg.llm_model, cfg.system_prompt)
    , tts_(cfg.tts_rate, cfg.tts_voice, cfg.tts_backend, cfg.piper_model_path, cfg.edge_tts_voice)
    , kws_(cfg.wake_word)
    , speaker_(cfg.sv_enroll_dir, cfg.sv_threshold)
    , memory_(cfg.max_rounds, cfg.max_tokens)
    , recorder_(cfg.sample_rate)
{
    // 根据配置启用/禁用技能
    skill_mgr_.set_enabled("weather",    cfg.skill_weather);
    skill_mgr_.set_enabled("time",       cfg.skill_time);
    skill_mgr_.set_enabled("web_search", cfg.skill_web_search);
}

bool VoicePipeline::initialize()
{
    LOG_INFO("============ 初始化管线 ============");

    if (!asr_.initialize()) {
        LOG_ERROR("❌ ASR 初始化失败");
    }

    if (!speaker_.initialize()) {
        LOG_ERROR("❌ 声纹模型初始化失败");
    }

    // ── 初始化多用户声纹库 (Layer 3.4) ──────────────────
    {
        float vp_threshold = cfg_.sv_threshold;
        std::string vp_dir = cfg_.sv_enroll_dir.empty()
            ? "voiceprint_library" : cfg_.sv_enroll_dir + "_library";
        if (voiceprint_.initialize(vp_dir, vp_threshold)) {
            voiceprint_.set_event_callback(
                [](const std::string& event, const std::string& detail) {
                    std::cout << "   [Voiceprint] 事件: " << event
                              << " → " << detail << std::endl;
                });
        }
    }

    if (!tts_.initialize()) {
        LOG_ERROR("❌ TTS 初始化失败");
        return false;
    }

    // ── 初始化 EmbeddingEngine + RAG ──────────────────
    if (cfg_.skill_rag) {
        if (cfg_.embedding_backend == "onnx") {
            LOG_INFO("[Embedding] 初始化 (本地 ONNX) ...");
            embed_ = EmbeddingEngine::create_onnx(cfg_.embedding_model_dir);
            if (!embed_) {
                LOG_WARN("⚠️ ONNX 嵌入初始化失败，降级使用 Ollama");
                embed_ = std::make_shared<EmbeddingEngine>(cfg_.ollama_host, cfg_.llm_model);
            }
        } else {
            LOG_INFO("[Embedding] 初始化 (Ollama /api/embed) ...");
            embed_ = std::make_shared<EmbeddingEngine>(cfg_.ollama_host, cfg_.llm_model);
        }
        skill_mgr_.register_rag(embed_, cfg_.rag_docs_dir);
        skill_mgr_.set_enabled("rag", true);
    }

    // ── 初始化 Function Calling ──────────────────────
    if (cfg_.fc_enabled) {
        std::string fc_model = cfg_.fc_model.empty() ? cfg_.llm_model : cfg_.fc_model;
        std::cout << "[FunctionCalling] 启用 LLM 驱动工具选择 (模型: "
                  << fc_model << ")" << std::endl;
        fc_ = std::make_shared<FunctionCaller>(cfg_.ollama_host, fc_model);
        skill_mgr_.set_function_caller(fc_);
    } else {
        LOG_INFO("[FunctionCalling] 禁用，使用关键字匹配降级方案");
        skill_mgr_.set_function_calling_enabled(false);
    }

    // ── 初始化 ReAct 多步推理 ────────────────────────
    if (cfg_.react_enabled) {
        std::cout << "[ReAct] 🧠 启用多步推理 (最多 "
                  << cfg_.react_max_steps << " 步)" << std::endl;
        react_ = std::make_shared<ReActEngine>(
            cfg_.ollama_host, cfg_.llm_model, cfg_.system_prompt);
    } else {
        LOG_INFO("[ReAct] 禁用，使用单步推理");
    }

    // ── 初始化 Reflection 反思 ────────────────────────
    if (cfg_.reflect_enabled) {
        std::string rmodel = cfg_.reflect_model.empty()
            ? cfg_.llm_model : cfg_.reflect_model;
        std::cout << "[Reflect] 🔍 启用回复反思修正 (模型: "
                  << rmodel << ")" << std::endl;
        reflect_ = std::make_shared<ReflectionEngine>(cfg_.ollama_host, rmodel);
    } else {
        LOG_INFO("[Reflect] 禁用");
    }

    // ── 初始化 Multi-Agent ───────────────────────────
    if (cfg_.multi_agent_enabled) {
        std::string cmodel = cfg_.ma_critic_model.empty()
            ? cfg_.llm_model : cfg_.ma_critic_model;
        std::cout << "[MultiAgent] 🤝 启用双 Agent 协作 (Critic: "
                  << cmodel << ", 最多" << cfg_.ma_max_rounds << "轮)" << std::endl;
        multi_agent_ = std::make_shared<MultiAgentEngine>(cfg_.ollama_host);
    } else {
        LOG_INFO("[MultiAgent] 禁用");
    }

    // ── 初始化流式 ASR (Layer 3.2) ────────────────────
    if (cfg_.streaming_asr_enabled) {
        std::string sa_model = cfg_.streaming_asr_model.empty()
            ? cfg_.asr_model_path : cfg_.streaming_asr_model;
        std::cout << "[StreamASR] 🎤 流式语音识别 (后端: "
                  << cfg_.streaming_asr_backend << ")" << std::endl;

        StreamingASRConfig sa_cfg;
        sa_cfg.backend              = cfg_.streaming_asr_backend;
        sa_cfg.model_path           = sa_model;
        sa_cfg.model_type           = cfg_.asr_model_type;
        sa_cfg.min_chunk_seconds    = cfg_.streaming_min_chunk;
        sa_cfg.chunk_interval       = cfg_.streaming_chunk_intv;

        if (stream_asr_.initialize(sa_cfg)) {
            // 设置部分结果回调：实时显示识别文本
            stream_asr_.set_partial_callback([](const char* text) {
                std::cout << "   🎤 " << text << "\r" << std::flush;
            });
        } else {
            LOG_INFO("   ⚠️ 流式 ASR 初始化失败，回退到离线模式");
        }
    }

    // ── 初始化长期记忆 (持久化) ─────────────────────
    memory_dir_ = cfg_.memory_persist_dir;
    if (!memory_dir_.empty()) {
        // 创建持久化目录
        std::string mkdir_cmd = "mkdir -p " + memory_dir_;
        (void)system(mkdir_cmd.c_str());

        // 从磁盘加载已有记忆
        load_all_memory();
    }

    // 注册记忆技能（需要 UserMemoryStore 引用）
    if (cfg_.memory_long_term_enabled) {
        skill_mgr_.add_skill(std::make_unique<MemorySkill>(&user_memory_));
        skill_mgr_.set_enabled("memory", true);
        LOG_INFO("[Memory] 🧠 长期记忆已启用 (持久化目录: {})", memory_dir_);
    }

    std::cout << "[5/5] Ollama: " << cfg_.llm_model
              << " (" << cfg_.ollama_host << ")" << std::endl;

    LOG_INFO("   特性: ");
    if (kws_.enabled()) std::cout << "唤醒词=\"" << cfg_.wake_word << "\" ";
    std::cout << "声纹(阈值=" << cfg_.sv_threshold << ") ";
    std::cout << "记忆(最近" << cfg_.max_rounds << "轮";
    if (cfg_.memory_long_term_enabled) std::cout << "+长期";
    std::cout << ")";
    if (cfg_.skill_rag) std::cout << " RAG(" << cfg_.rag_docs_dir << ")";
    LOG_INFO("std::endl");

    std::cout << "   " << speaker_.status_text() << std::endl;
    std::cout << "   [Voiceprint] " << voiceprint_.status_text() << std::endl;
    LOG_INFO("std::endl");

    initialized_ = true;
    return true;
}

// ── 单次模式 ────────────────────────────────────────

std::string VoicePipeline::process_text(const std::string& text)
{
    if (!initialized_) return "";

    std::string reply;

    auto t_start = std::chrono::steady_clock::now();

    // ── 策略 0: 关键字技能匹配（最优先，不依赖LLM）─────
    //     先让关键字 match 抢答（计算/系统/笔记/提醒/笑话等）
    //     这些确定性任务不需要 LLM 理解，关键字就够了
    SkillResult sr = skill_mgr_.detect_and_execute(text);
    std::string extra = SkillManager::get_system_context();
    bool skill_hit = sr.hit;
    if (skill_hit) {
        extra += "\n" + sr.result_text;
        std::cout << "   [Skill] \"" << text << "\" → " << sr.skill_name << std::endl;
        // 内容型技能（笑话/故事/诗词等）：直接返回，不经过LLM避免截断和延迟
        if (sr.direct) reply = sr.result_text;
    }

    auto t_after_skill = std::chrono::steady_clock::now();

    // 注入用户长期记忆
    extra = build_extra_context(extra, user_memory_, cfg_.memory_long_term_enabled);

    // ── 策略 1: ReAct 多步推理（skill 未命中时才用）───
    //     复杂多步任务（如"查天气然后提醒我"）需要 ReAct
    if (reply.empty() && react_ && !skill_hit) {
        auto tools = skill_mgr_.collect_function_defs();
        if (!tools.empty()) {
            auto exec_fn = [this](const std::string& name, const nlohmann::json& args) {
                return skill_mgr_.execute_tool(name, args, "");
            };

            std::string ctx = memory_.get_context();
            std::string react_ctx = ctx.empty() ? extra : ctx + "\n" + extra;
            auto result = react_->run(text, tools, exec_fn, react_ctx, cfg_.react_max_steps);

            if (result.success && !result.final_answer.empty()) {
                reply = result.final_answer;
            } else if (!result.success) {
                LOG_INFO("   [ReAct] 推理失败，降级到单步模式");
            }
        }
    }

    // ── 策略 2: 直接 LLM 回复（skill命中/ReAct失败时）─
    if (reply.empty()) {
        std::string context = memory_.get_context();
        reply = llm_.chat(text, context, extra);
    }

    auto t_after_llm = std::chrono::steady_clock::now();

    if (reply.empty()) return "";

    // ── 质量优化：Multi-Agent 协作 或 Reflection 反思 ──
    if (multi_agent_) {
        std::string cmodel = cfg_.ma_critic_model.empty()
            ? cfg_.llm_model : cfg_.ma_critic_model;
        auto ma = multi_agent_->collaborate(
            text, reply, /*tool_context*/"",
            cfg_.system_prompt, cfg_.llm_model, cmodel, cfg_.ma_max_rounds);
        reply = ma.final_answer;
    } else if (reflect_) {
        auto r = reflect_->reflect(text, reply);
        reply = r.improved;
    }

    auto t_after_quality = std::chrono::steady_clock::now();

    memory_.add(text, reply);
    save_all_memory();  // 持久化
    reply = strip_emoji(reply);
    speak_and_play(reply, text);

    auto t_end = std::chrono::steady_clock::now();

    // ── 延迟汇总 ──
    auto skill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_skill - t_start).count();
    auto llm_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_llm - t_after_skill).count();
    auto qual_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_quality - t_after_llm).count();
    auto tts_play_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_after_quality).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    LOG_INFO("⏱️  [延迟] 技能:{}ms | LLM:{}ms | 质检:{}ms | TTS+播放:{}ms | 总计:{}ms",
             skill_ms, llm_ms, qual_ms, tts_play_ms, total_ms);

    return reply;
}

std::string VoicePipeline::transcribe_file(const std::string& wav_path)
{
    if (!initialized_) return "";
    return asr_.transcribe(wav_path);
}

std::string VoicePipeline::process_text_for_ws(const std::string& text,
                                                 const std::string& wav_path)
{
    if (!initialized_) return "";

    std::string reply;

    auto t_pipeline_start = std::chrono::steady_clock::now();

    // ── 策略 0: 关键字技能匹配（最优先）───────────────
    SkillResult sr = skill_mgr_.detect_and_execute(text);
    std::string extra = SkillManager::get_system_context();
    bool skill_hit = sr.hit;
    if (skill_hit) {
        extra += "\n" + sr.result_text;
        std::cout << "   [Skill] \"" << text << "\" → " << sr.skill_name << std::endl;
        // 内容型技能（笑话/故事/诗词等）：直接返回，不经过LLM避免截断
        if (sr.direct) reply = sr.result_text;
    }

    auto t_after_skill = std::chrono::steady_clock::now();

    // 注入用户长期记忆
    extra = build_extra_context(extra, user_memory_, cfg_.memory_long_term_enabled);

    // ── 策略 1: ReAct 多步推理（skill 未命中时才用）───
    if (reply.empty() && react_ && !skill_hit) {
        auto tools = skill_mgr_.collect_function_defs();
        if (!tools.empty()) {
            auto exec_fn = [this](const std::string& name, const nlohmann::json& args) {
                return skill_mgr_.execute_tool(name, args, "");
            };

            std::string ctx = memory_.get_context();
            std::string react_ctx = ctx.empty() ? extra : ctx + "\n" + extra;
            auto result = react_->run(text, tools, exec_fn, react_ctx, cfg_.react_max_steps);

            if (result.success && !result.final_answer.empty()) {
                reply = result.final_answer;
            } else if (!result.success) {
                LOG_INFO("   [ReAct] 推理失败，降级到单步模式");
            }
        }
    }

    // ── 策略 2: 直接 LLM 回复 ───────────────────────
    if (reply.empty()) {

        std::string context = memory_.get_context();
        reply = llm_.chat(text, context, extra);
    }

    auto t_after_llm = std::chrono::steady_clock::now();

    if (reply.empty()) return "";

    // ── 质量优化：Multi-Agent 协作 或 Reflection 反思 ──
    if (multi_agent_) {
        std::string cmodel = cfg_.ma_critic_model.empty()
            ? cfg_.llm_model : cfg_.ma_critic_model;
        auto ma = multi_agent_->collaborate(
            text, reply, /*tool_context*/"",
            cfg_.system_prompt, cfg_.llm_model, cmodel, cfg_.ma_max_rounds);
        reply = ma.final_answer;
    } else if (reflect_) {
        auto r = reflect_->reflect(text, reply);
        reply = r.improved;
    }

    auto t_after_quality = std::chrono::steady_clock::now();

    // 回复截断（与 process_loop 保持一致）
    if (cfg_.max_response_chars > 0 && (int)reply.size() > cfg_.max_response_chars * 3) {
        size_t cut = (size_t)cfg_.max_response_chars * 3;
        for (size_t off = 0; off < 60 && cut + off < reply.size(); off++) {
            size_t pos = cut + off;
            if (pos + 2 < reply.size() && (unsigned char)reply[pos] == 0xE3) {
                unsigned char c1 = reply[pos+1], c2 = reply[pos+2];
                if ((c1 == 0x80 && c2 == 0x82) ||
                    (c1 == 0xBC && c2 == 0x81) ||
                    (c1 == 0xBC && c2 == 0x9F)) {
                    reply = reply.substr(0, pos + 3);
                    break;
                }
            }
        }
    }

    memory_.add(text, reply);
    save_all_memory();  // 持久化
    reply = strip_emoji(reply);

    // TTS 合成到文件（不本地播放）
    if (!tts_.synthesize(reply, wav_path, text)) {
        LOG_ERROR("   ❌ TTS 合成失败");
        return reply;  // 仍然返回 LLM 文本
    }

    auto t_after_tts = std::chrono::steady_clock::now();

    // ── 延迟汇总 ──
    auto skill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_skill - t_pipeline_start).count();
    auto llm_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_llm - t_after_skill).count();
    auto qual_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_quality - t_after_llm).count();
    auto tts_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_tts - t_after_quality).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_tts - t_pipeline_start).count();
    LOG_INFO("⏱️  [延迟] 技能:{}ms | LLM:{}ms | 质检:{}ms | TTS:{}ms | 总计:{}ms",
             skill_ms, llm_ms, qual_ms, tts_ms, total_ms);

    return reply;
}

std::string VoicePipeline::process_voice()
{
    if (!initialized_) return "";

    const std::string wav_file = "temp_recording.wav";

    if (!recorder_.record(wav_file)) {
        return "";
    }

    std::string prompt = asr_.transcribe(wav_file);
    if (prompt.empty()) {
        LOG_INFO("   ⚠️ 未识别到语音");
        std::remove(wav_file.c_str());
        return "";
    }

    if (!kws_.detect(prompt)) {
        speak_and_play("请说出正确的唤醒词");
        std::remove(wav_file.c_str());
        return "";
    }

    // ── 多用户声纹识别 (Layer 3.4) ──────────────────────
    if (voiceprint_.has_any()) {
        auto id = voiceprint_.identify(wav_file);
        if (id.identified()) {
            // 自动切换到识别的用户
            if (id.name != voiceprint_.active_speaker()) {
                voiceprint_.set_active_speaker(id.name);
            }
            std::cout << "   👤 说话人: " << id.display_name
                      << " (置信度=" << id.confidence << ")" << std::endl;
        } else {
            speak_and_play("抱歉，我没有识别出你的声音，请先注册声纹");
            std::remove(wav_file.c_str());
            return "";
        }
    } else if (speaker_.has_enrolled()) {
        // 旧版单用户验证
        if (!speaker_.verify(wav_file)) {
            speak_and_play("声纹验证失败，我无法为您服务");
            std::remove(wav_file.c_str());
            return "";
        }
    } else {
        speak_and_play("请先注册声纹，输入 enroll 开始注册");
        std::remove(wav_file.c_str());
        return "";
    }

    std::remove(wav_file.c_str());

    std::string reply;

    // ── 策略 0: 关键字技能匹配（最优先）───────────────
    SkillResult sr = skill_mgr_.detect_and_execute(prompt);
    std::string extra = SkillManager::get_system_context();
    bool skill_hit = sr.hit;
    if (skill_hit) {
        extra += "\n" + sr.result_text;
        std::cout << "   [Skill] \"" << prompt << "\" → " << sr.skill_name << std::endl;
        // 内容型技能（笑话/故事/诗词等）：直接返回，不经过LLM避免截断和延迟
        if (sr.direct) reply = sr.result_text;
    }

    // 注入用户长期记忆
    extra = build_extra_context(extra, user_memory_, cfg_.memory_long_term_enabled);

    // ── 策略 1: ReAct 多步推理（skill 未命中时才用）───
    if (reply.empty() && react_ && !skill_hit) {
        auto tools = skill_mgr_.collect_function_defs();
        if (!tools.empty()) {
            auto exec_fn = [this](const std::string& name, const nlohmann::json& args) {
                return skill_mgr_.execute_tool(name, args, "");
            };

            std::string ctx = memory_.get_context();
            std::string react_ctx = ctx.empty() ? extra : ctx + "\n" + extra;
            auto result = react_->run(prompt, tools, exec_fn, react_ctx, cfg_.react_max_steps);

            if (result.success && !result.final_answer.empty()) {
                reply = result.final_answer;
            }
        }
    }

    // ── 降级：直接 LLM 回复 ─────────────────────────
    if (reply.empty()) {
        std::string context = memory_.get_context();
        reply = llm_.chat(prompt, context, extra);
    }

    if (reply.empty()) return "";

    // ── 质量优化 ───────────────────────────────────
    if (multi_agent_) {
        std::string cmodel = cfg_.ma_critic_model.empty()
            ? cfg_.llm_model : cfg_.ma_critic_model;
        auto ma = multi_agent_->collaborate(
            prompt, reply, "", cfg_.system_prompt,
            cfg_.llm_model, cmodel, cfg_.ma_max_rounds);
        reply = ma.final_answer;
    } else if (reflect_) {
        auto r = reflect_->reflect(prompt, reply);
        reply = r.improved;
    }

    memory_.add(prompt, reply);
    save_all_memory();  // 持久化
    reply = strip_emoji(reply);
    speak_and_play(reply, prompt);

    return reply;
}

std::string VoicePipeline::process_voice_file(const std::string& wav_path)
{
    if (!initialized_) return "";

    std::string prompt = asr_.transcribe(wav_path);
    if (prompt.empty()) {
        LOG_INFO("   ⚠️ 未识别到语音");
        return "";
    }

    // 跳过唤醒词和声纹检查（WS 客户端已自行管理）

    std::string reply;

    // ── 策略 0: 关键字技能匹配（最优先）───────────────
    SkillResult sr = skill_mgr_.detect_and_execute(prompt);
    std::string extra = SkillManager::get_system_context();
    bool skill_hit = sr.hit;
    if (skill_hit) {
        extra += "\n" + sr.result_text;
        std::cout << "   [Skill] \"" << prompt << "\" → " << sr.skill_name << std::endl;
        // 内容型技能（笑话/故事/诗词等）：直接返回，不经过LLM避免截断和延迟
        if (sr.direct) reply = sr.result_text;
    }

    extra = build_extra_context(extra, user_memory_, cfg_.memory_long_term_enabled);

    // ── 策略 1: ReAct（skill 未命中时才用）───────────
    if (reply.empty() && react_ && !skill_hit) {
        auto tools = skill_mgr_.collect_function_defs();
        if (!tools.empty()) {
            auto exec_fn = [this](const std::string& name, const nlohmann::json& args) {
                return skill_mgr_.execute_tool(name, args, "");
            };
            std::string ctx = memory_.get_context();
            std::string react_ctx = ctx.empty() ? extra : ctx + "\n" + extra;
            auto result = react_->run(prompt, tools, exec_fn, react_ctx, cfg_.react_max_steps);
            if (result.success && !result.final_answer.empty()) {
                reply = result.final_answer;
            }
        }
    }

    // ── 策略 2: 直接 LLM 回复 ───────────────────────
    if (reply.empty()) {
        std::string context = memory_.get_context();
        reply = llm_.chat(prompt, context, extra);
    }

    if (reply.empty()) return "";

    if (multi_agent_) {
        std::string cmodel = cfg_.ma_critic_model.empty()
            ? cfg_.llm_model : cfg_.ma_critic_model;
        auto ma = multi_agent_->collaborate(
            prompt, reply, "", cfg_.system_prompt,
            cfg_.llm_model, cmodel, cfg_.ma_max_rounds);
        reply = ma.final_answer;
    } else if (reflect_) {
        auto r = reflect_->reflect(prompt, reply);
        reply = r.improved;
    }

    memory_.add(prompt, reply);
    save_all_memory();  // 持久化
    reply = strip_emoji(reply);
    speak_and_play(reply, prompt);

    return reply;
}

bool VoicePipeline::enroll_speaker()
{
    const std::string wav_file = "temp_enroll.wav";

    LOG_INFO("   🎙️ 请说出你的名字（3秒后开始录音）...");
    speak_and_play("请在提示音后说出你的名字");

    if (!recorder_.record(wav_file)) {
        return false;
    }

    // 先用 ASR 获取用户名
    std::string user_name = asr_.transcribe(wav_file);
    if (user_name.empty()) {
        speak_and_play("没有识别到你说的名字，请重试");
        std::remove(wav_file.c_str());
        return false;
    }

    // 清理用户名（过滤中文标点和空格）
    std::string clean_name;
    for (size_t i = 0; i < user_name.size(); ) {
        unsigned char c = (unsigned char)user_name[i];
        // 跳过 ASCII 空格
        if (c == ' ') { i++; continue; }
        // 跳过 UTF-8 中文标点（3字节序列）
        if (c == 0xE3 || c == 0xEF) {
            i += 3;
            continue;
        }
        clean_name += user_name[i];
        i++;
    }

    // 重新录音（用于声纹注册，建议 >3 秒）
    LOG_INFO("   🎙️ 请说一段话（用于声纹注册，3秒后开始）...");
    speak_and_play("请说一小段话，用于声纹注册");
    std::remove(wav_file.c_str());

    if (!recorder_.record(wav_file)) {
        return false;
    }

    // 注册到声纹库
    if (!voiceprint_.enroll(wav_file, clean_name, clean_name)) {
        // 降级到旧版单用户注册
        if (!speaker_.enroll(wav_file)) {
            std::remove(wav_file.c_str());
            return false;
        }
        speak_and_play("声纹注册完成！");
    } else {
        std::string msg = "声纹注册完成！欢迎你，" + clean_name;
        speak_and_play(msg);
    }

    std::remove(wav_file.c_str());
    return true;
}

void VoicePipeline::clear_memory()
{
    memory_.clear();
    user_memory_.clear();
    LOG_INFO("   🧹 对话记忆已清空");
}

void VoicePipeline::save_all_memory()
{
    if (memory_dir_.empty()) return;

    std::string chat_path = memory_dir_ + "/chat_history.json";
    std::string user_path = memory_dir_ + "/user_memory.json";

    memory_.save_to_file(chat_path);
    user_memory_.save_to_file(user_path);
}

void VoicePipeline::load_all_memory()
{
    if (memory_dir_.empty()) return;

    // 每次启动重置对话上下文，只加载长期用户记忆
    std::string user_path = memory_dir_ + "/user_memory.json";
    user_memory_.load_from_file(user_path);

    LOG_INFO("[Memory] 对话上下文已重置，长期记忆已加载");
}

// ── 热配置重载 (Layer 4.4) ──────────────────────────────

void VoicePipeline::reload_config(const PipelineConfig& new_cfg)
{
    LOG_INFO("[Config] ============ 热重载配置 ============");

    int hot_changes = 0;
    int warn_changes = 0;

    // 辅助：比较 string 字段
    auto diff_str = [&](const char* name,
                         const std::string& old_val, const std::string& new_val,
                         std::string& target) {
        if (old_val != new_val) {
            LOG_INFO("[Config] {}: \"{}\" → \"{}\"", name, old_val, new_val);
            target = new_val;
            return true;
        }
        return false;
    };

    // 辅助：比较数值字段
    auto diff_val = [&](const char* name, auto old_val, auto new_val, auto& target) {
        if (old_val != new_val) {
            LOG_INFO("[Config] {}: {} → {}", name, old_val, new_val);
            target = new_val;
            return true;
        }
        return false;
    };

    // 辅助：仅记录警告（需重启字段）
    auto warn_if = [&](const char* name, auto old_val, auto new_val) {
        if (old_val != new_val) {
            LOG_WARN("[Config] ⚠️  {}: \"{}\" → \"{}\" (需重启生效)", name, old_val, new_val);
            return true;
        }
        return false;
    };

    // ── Tier 1: 完全热安全 ────────────────────────────

    // LLM
    if (diff_str("system_prompt", cfg_.system_prompt, new_cfg.system_prompt, cfg_.system_prompt)) {
        llm_.set_system_prompt(cfg_.system_prompt);
        hot_changes++;
    }

    // TTS
    if (diff_str("tts_backend", cfg_.tts_backend, new_cfg.tts_backend, cfg_.tts_backend))
        hot_changes++;
    diff_val("tts_rate", cfg_.tts_rate, new_cfg.tts_rate, cfg_.tts_rate);

    // 唤醒词
    if (diff_str("wake_word", cfg_.wake_word, new_cfg.wake_word, cfg_.wake_word))
        hot_changes++;

    // 声纹阈值
    if (diff_val("sv_threshold", cfg_.sv_threshold, new_cfg.sv_threshold, cfg_.sv_threshold))
        hot_changes++;

    // 对话记忆
    bool mem_changed = false;
    if (cfg_.max_rounds != new_cfg.max_rounds) {
        LOG_INFO("[Config] max_rounds: {} → {}", cfg_.max_rounds, new_cfg.max_rounds);
        cfg_.max_rounds = new_cfg.max_rounds;
        mem_changed = true;
    }
    if (cfg_.max_tokens != new_cfg.max_tokens) {
        LOG_INFO("[Config] max_tokens: {} → {}", cfg_.max_tokens, new_cfg.max_tokens);
        cfg_.max_tokens = new_cfg.max_tokens;
        mem_changed = true;
    }
    if (mem_changed) {
        memory_.set_limits(cfg_.max_rounds, cfg_.max_tokens);
        hot_changes++;
    }
    // 长期记忆持久化路径
    if (diff_str("memory_persist_dir", cfg_.memory_persist_dir, new_cfg.memory_persist_dir, cfg_.memory_persist_dir)) {
        memory_dir_ = cfg_.memory_persist_dir;
        if (!memory_dir_.empty()) {
            std::string mkdir_cmd = "mkdir -p " + memory_dir_;
            (void)system(mkdir_cmd.c_str());
        }
        hot_changes++;
    }
    diff_val("memory_long_term_enabled", cfg_.memory_long_term_enabled, new_cfg.memory_long_term_enabled, cfg_.memory_long_term_enabled);
    diff_val("memory_auto_extract", cfg_.memory_auto_extract, new_cfg.memory_auto_extract, cfg_.memory_auto_extract);

    // ReAct
    if (diff_val("react_max_steps", cfg_.react_max_steps, new_cfg.react_max_steps, cfg_.react_max_steps))
        hot_changes++;

    // Multi-Agent
    if (diff_str("ma_critic_model", cfg_.ma_critic_model, new_cfg.ma_critic_model, cfg_.ma_critic_model))
        hot_changes++;
    if (diff_val("ma_max_rounds", cfg_.ma_max_rounds, new_cfg.ma_max_rounds, cfg_.ma_max_rounds))
        hot_changes++;

    // 交互模式
    diff_val("barge_in_enabled", cfg_.barge_in_enabled, new_cfg.barge_in_enabled, cfg_.barge_in_enabled);
    diff_val("barge_in_energy_ratio", cfg_.barge_in_energy_ratio, new_cfg.barge_in_energy_ratio, cfg_.barge_in_energy_ratio);
    diff_val("max_response_chars", cfg_.max_response_chars, new_cfg.max_response_chars, cfg_.max_response_chars);

    // VAD 参数（capture_loop 每帧读取 cfg_）
    diff_val("vad_energy_threshold", cfg_.vad_energy_threshold, new_cfg.vad_energy_threshold, cfg_.vad_energy_threshold);
    diff_val("vad_min_speech_frames", cfg_.vad_min_speech_frames, new_cfg.vad_min_speech_frames, cfg_.vad_min_speech_frames);
    diff_val("vad_min_silence_frames", cfg_.vad_min_silence_frames, new_cfg.vad_min_silence_frames, cfg_.vad_min_silence_frames);
    diff_val("vad_pre_speech_frames", cfg_.vad_pre_speech_frames, new_cfg.vad_pre_speech_frames, cfg_.vad_pre_speech_frames);
    diff_val("vad_adaptive_factor", cfg_.vad_adaptive_factor, new_cfg.vad_adaptive_factor, cfg_.vad_adaptive_factor);
    diff_val("vad_min_energy", cfg_.vad_min_energy, new_cfg.vad_min_energy, cfg_.vad_min_energy);
    diff_val("vad_cooldown_frames", cfg_.vad_cooldown_frames, new_cfg.vad_cooldown_frames, cfg_.vad_cooldown_frames);

    // 流式 ASR 运行时参数
    diff_val("streaming_min_chunk", cfg_.streaming_min_chunk, new_cfg.streaming_min_chunk, cfg_.streaming_min_chunk);
    diff_val("streaming_chunk_intv", cfg_.streaming_chunk_intv, new_cfg.streaming_chunk_intv, cfg_.streaming_chunk_intv);

    // 技能开关
    auto diff_skill = [&](const char* name, bool old_val, bool new_val, bool& target) {
        if (old_val != new_val) {
            LOG_INFO("[Config] skill_{}: {} → {}", name, old_val, new_val);
            target = new_val;
            skill_mgr_.set_enabled(name, new_val);
            return true;
        }
        return false;
    };
    diff_skill("weather",    cfg_.skill_weather,    new_cfg.skill_weather,    cfg_.skill_weather);
    diff_skill("time",       cfg_.skill_time,       new_cfg.skill_time,       cfg_.skill_time);
    diff_skill("web_search", cfg_.skill_web_search, new_cfg.skill_web_search, cfg_.skill_web_search);

    // ── Tier 2: 需重建对象（更新字段但警告）──────────

    if (warn_if("fc_enabled", cfg_.fc_enabled, new_cfg.fc_enabled)) {
        cfg_.fc_enabled = new_cfg.fc_enabled;
        warn_changes++;
    }
    if (diff_str("fc_model", cfg_.fc_model, new_cfg.fc_model, cfg_.fc_model))
        warn_changes++;

    if (warn_if("react_enabled", cfg_.react_enabled, new_cfg.react_enabled)) {
        cfg_.react_enabled = new_cfg.react_enabled;
        warn_changes++;
    }
    if (warn_if("reflect_enabled", cfg_.reflect_enabled, new_cfg.reflect_enabled)) {
        cfg_.reflect_enabled = new_cfg.reflect_enabled;
        warn_changes++;
    }
    if (diff_str("reflect_model", cfg_.reflect_model, new_cfg.reflect_model, cfg_.reflect_model))
        warn_changes++;

    if (warn_if("multi_agent_enabled", cfg_.multi_agent_enabled, new_cfg.multi_agent_enabled)) {
        cfg_.multi_agent_enabled = new_cfg.multi_agent_enabled;
        warn_changes++;
    }
    if (warn_if("streaming_asr_enabled", cfg_.streaming_asr_enabled, new_cfg.streaming_asr_enabled)) {
        cfg_.streaming_asr_enabled = new_cfg.streaming_asr_enabled;
        warn_changes++;
    }
    if (warn_if("skill_rag", cfg_.skill_rag, new_cfg.skill_rag)) {
        cfg_.skill_rag = new_cfg.skill_rag;
        warn_changes++;
    }
    if (diff_str("rag_docs_dir", cfg_.rag_docs_dir, new_cfg.rag_docs_dir, cfg_.rag_docs_dir))
        warn_changes++;

    // ── Tier 3: 需重启 ──────────────────────────────

    warn_if("asr_model_path",     cfg_.asr_model_path,     new_cfg.asr_model_path);
    warn_if("ollama_host",        cfg_.ollama_host,        new_cfg.ollama_host);
    warn_if("llm_model",          cfg_.llm_model,          new_cfg.llm_model);
    warn_if("piper_model_path",   cfg_.piper_model_path,   new_cfg.piper_model_path);
    warn_if("tts_voice",          cfg_.tts_voice,          new_cfg.tts_voice);
    warn_if("sv_enroll_dir",      cfg_.sv_enroll_dir,      new_cfg.sv_enroll_dir);
    warn_if("embedding_backend",  cfg_.embedding_backend,  new_cfg.embedding_backend);
    warn_if("embedding_model_dir",cfg_.embedding_model_dir,new_cfg.embedding_model_dir);
    warn_if("sample_rate",        cfg_.sample_rate,        new_cfg.sample_rate);

    // vad_backend 也需要重建 VAD 对象
    if (warn_if("vad_backend", cfg_.vad_backend, new_cfg.vad_backend)) {
        cfg_.vad_backend = new_cfg.vad_backend;
        warn_changes++;
    }
    if (warn_if("streaming_asr_backend", cfg_.streaming_asr_backend, new_cfg.streaming_asr_backend)) {
        cfg_.streaming_asr_backend = new_cfg.streaming_asr_backend;
        warn_changes++;
    }
    if (diff_str("streaming_asr_model", cfg_.streaming_asr_model, new_cfg.streaming_asr_model, cfg_.streaming_asr_model))
        warn_changes++;

    // ── 汇总 ─────────────────────────────────────────

    LOG_INFO("[Config] ✅ 热应用完成 ({} 项已生效, {} 项需重启)",
             hot_changes, warn_changes);
}

void VoicePipeline::speak_and_play(const std::string& text,
                                     const std::string& user_context)
{
    LOG_INFO("   🔊 播放中...");

    if (cfg_.tts_backend == "piper") {
        tts_.synthesize(text, "", user_context);
    } else {
        // espeak / edge_tts 需要输出 WAV 文件后播放
        const std::string tts_file = "temp_reply.wav";
        if (tts_.synthesize(text, tts_file, user_context)) {
            AudioPlayer::play(tts_file);
            std::remove(tts_file.c_str());
        }
    }
}

// ── 交互模式（可打断）───────────────────────────────

void VoicePipeline::run_interactive()
{
    if (!initialized_) {
        LOG_ERROR("❌ 管线未初始化");
        return;
    }

    // 避免僵尸进程
    signal(SIGCHLD, SIG_IGN);

    interactive_running_ = true;
    is_playing_ = false;
    player_pid_ = -1;
    generation_ = 0;

    // 清空队列
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        segment_queue_ = std::queue<Segment>();
    }

    LOG_INFO("std::endl");
    LOG_INFO("============================================================");
    LOG_INFO("  🎤 交互模式已启动（支持语音打断）");
    LOG_INFO("  直接说话即可交互，机器人说话时你可以随时打断");
    LOG_INFO("  按 Ctrl+C 退出交互模式");
    LOG_INFO("============================================================");
    LOG_INFO("std::endl");

    // 启动两个工作线程
    capture_thread_ = std::thread(&VoicePipeline::capture_loop, this);
    process_thread_ = std::thread(&VoicePipeline::process_loop, this);

    // 等待线程结束
    if (capture_thread_.joinable())  capture_thread_.join();
    if (process_thread_.joinable())  process_thread_.join();

    std::cout << std::endl << "👋 交互模式已退出" << std::endl;
}

void VoicePipeline::stop_interactive()
{
    LOG_INFO("\n⏹️  收到停止信号...");
    interactive_running_ = false;

    // 停止当前播放
    pid_t pid = player_pid_.load();
    if (pid > 0) {
        AudioPlayer::stop_async(pid);
        player_pid_ = -1;
    }

    // 唤醒 process 线程使其退出
    queue_cv_.notify_all();
}

// ── Capture 线程 ─────────────────────────────────────

void VoicePipeline::capture_loop()
{
    // 打开 arecord 管道: 持续输出 raw 16-bit PCM
    FILE* pipe = popen(
        "arecord -f S16_LE -r 16000 -c 1 -t raw -q 2>/dev/null", "r");
    if (!pipe) {
        LOG_ERROR("❌ 无法启动录音设备");
        interactive_running_ = false;
        queue_cv_.notify_all();
        return;
    }

    // 从 PipelineConfig 构造 VAD 配置
    VADConfig vad_cfg;
    vad_cfg.energy_threshold       = cfg_.vad_energy_threshold;
    vad_cfg.min_speech_frames      = cfg_.vad_min_speech_frames;
    vad_cfg.min_silence_frames     = cfg_.vad_min_silence_frames;
    vad_cfg.pre_speech_frames      = cfg_.vad_pre_speech_frames;
    vad_cfg.adaptive_factor        = cfg_.vad_adaptive_factor;
    vad_cfg.min_energy_threshold   = cfg_.vad_min_energy;
    vad_cfg.silence_cooldown_frames = cfg_.vad_cooldown_frames;

    auto vad = create_vad(cfg_.vad_backend, vad_cfg);

    const int frame_samples = vad_cfg.frame_size_samples;   // 20ms @16kHz
    const int frame_bytes   = frame_samples * 2;  // int16 = 2 bytes

    std::vector<int16_t> raw_buf(frame_samples);
    std::vector<float>   float_buf(frame_samples);

    // 音频预处理器: 去直流偏置 + 噪声门
    Denoiser denoiser(16000);

    // 播放状态跟踪 + 打断检测
    bool was_playing = false;
    float echo_baseline  = 0.0f;      // 回声能量基线
    int   echo_frames    = 0;         // 基线学习帧数
    int   barge_in_frames = 0;        // 连续高于基线的帧数

    // ── ASR 文本稳定性端点检测 ──────────────────
    // VAD 静音不可靠时，用 ASR 部分结果是否稳定来判断"说完了"
    //   - 文本以 。！？结尾 + 稳定 0.8s → 触发端点
    //   - 文本稳定 1.5s → 触发端点
    std::string asr_last_partial;     // 上一帧的部分识别文本
    int   asr_stable_frames  = 0;    // 文本连续不变的帧数

    std::cout << "🎙️  麦克风已开启，开始监听... (VAD: " << cfg_.vad_backend << ")" << std::endl;

    while (interactive_running_) {
        // 读一帧音频
        size_t nread = fread(raw_buf.data(), 1, frame_bytes, pipe);
        if (nread != (size_t)frame_bytes) {
            if (interactive_running_) {
                LOG_WARN("⚠️ 录音读取异常");
            }
            break;
        }

        // int16 → float [-1, 1]
        for (int i = 0; i < frame_samples; ++i) {
            float_buf[i] = raw_buf[i] / 32768.0f;
        }

        // 音频预处理: 去直流 + 噪声门
        denoiser.process_frame(float_buf.data(), frame_samples);

        // ── 播放结束检测：重置 VAD 丢弃回声 ──
        bool playing = is_playing_.load();
        if (!playing && was_playing) {
            vad->reset();
            if (stream_asr_.initialized()) {
                stream_asr_.cancel();
            }
            asr_last_partial.clear();
            asr_stable_frames = 0;
            speech_timeout_warned_ = false;
            asr_endpoint_logged_ = false;
        }
        was_playing = playing;

        // 持续喂预录音缓冲区
        vad->feed_pre_buffer(float_buf.data(), frame_samples);

        // VAD 处理
        VADState vad_state = vad->process_frame(float_buf.data(), frame_samples);
        bool in_speech = vad->in_speech();

        // ── 流式 ASR：始终喂帧（即使在推理中也持续识别）────
        bool busy = process_busy_.load();
        if (stream_asr_.initialized() && in_speech) {
            if (vad_state == VADState::SPEECH_START) {
                stream_asr_.start_utterance();
                asr_last_partial.clear();
                asr_stable_frames = 0;
                speech_timeout_warned_ = false;
                asr_endpoint_logged_ = false;
            }
            stream_asr_.feed(float_buf.data(), frame_samples);

            // ── ASR 文本稳定性端点检测 ──────────────────
            // 当 ASR 部分结果连续稳定（不变）一段时间，认为语义完成
            // 不再依赖 VAD 检测到静音，解决"一直说话/噪声时 VAD 不休"的问题
            const char* p = stream_asr_.partial();
            std::string cur_partial = (p && p[0]) ? std::string(p) : "";
            if (!cur_partial.empty() && cur_partial == asr_last_partial) {
                asr_stable_frames++;
            } else if (!cur_partial.empty()) {
                asr_stable_frames = 0;
                asr_last_partial = cur_partial;
            }
        }

        // ── ASR 端点触发（与 VAD 互补）──────────────────
        bool asr_endpoint = false;
        if (!asr_last_partial.empty() && asr_stable_frames > 0) {
            // 检查是否以句尾标点结尾: 。！？
            bool ends_with_period = false;
            size_t len = asr_last_partial.size();
            if (len >= 3) {
                unsigned char b0 = static_cast<unsigned char>(asr_last_partial[len-3]);
                unsigned char b1 = static_cast<unsigned char>(asr_last_partial[len-2]);
                unsigned char b2 = static_cast<unsigned char>(asr_last_partial[len-1]);
                // 。U+3002: E3 80 82
                if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) ends_with_period = true;
                // ！U+FF01: EF BC 81
                if (b0 == 0xEF && b1 == 0xBC && b2 == 0x81) ends_with_period = true;
                // ？U+FF1F: EF BC 9F
                if (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F) ends_with_period = true;
            }
            // 句尾标点 → 0.8s；普通稳定 → 1.5s
            int threshold = ends_with_period ? 40 : 75;
            if (asr_stable_frames >= threshold) {
                asr_endpoint = true;
            }
        }

        // ── 安全阀：最大语音时长 10 秒 ──────────────────
        bool speech_timeout = false;
        if (in_speech && vad->speech_sample_count() > 10 * 16000) {
            if (!speech_timeout_warned_) {
                LOG_WARN("⚠️ 语音段超过10秒，强制截断 (VAD可能卡在噪声中)");
                speech_timeout_warned_ = true;
            }
            speech_timeout = true;
        }

        // ── 打断检测：播放中检测用户语音 → 停止播放 ────
        if (cfg_.barge_in_enabled && playing) {
            // 计算当前帧 RMS
            float rms = 0.0f;
            for (int i = 0; i < frame_samples; ++i)
                rms += float_buf[i] * float_buf[i];
            rms = std::sqrt(rms / frame_samples);

            // 前 500ms 学习回声基线，不允许打断
            if (echo_frames < 25) {
                echo_baseline = (echo_baseline * echo_frames + rms) / (echo_frames + 1);
                echo_frames++;
                barge_in_frames = 0;
            } else if (rms > echo_baseline * cfg_.barge_in_energy_ratio) {
                // 能量显著高于回声基线 → 可能是用户语音
                barge_in_frames++;
            } else {
                // 更新回声基线（慢速自适应）
                echo_baseline = echo_baseline * 0.95f + rms * 0.05f;
                barge_in_frames = 0;
            }

            // 持续 300ms（15 帧）高于基线 → 触发打断
            if (barge_in_frames > 15) {
                pid_t pid = player_pid_.exchange(-1);
                if (pid > 0) {
                    AudioPlayer::stop_async(pid);
                    LOG_INFO("\n🔴 检测到真实语音，已打断当前播放！");
                }
                // 关键：立即允许 capture 线程收集新语音
                is_playing_ = false;
                process_busy_ = false;
                playing = false;
                barge_in_frames = 0;
                echo_frames = 0;
                // 重置 VAD + ASR 以捕获打断语音
                vad->reset();
                speech_timeout_warned_ = false;
                asr_endpoint_logged_ = false;
                if (stream_asr_.initialized()) {
                    stream_asr_.cancel();
                    stream_asr_.start_utterance();
                }
            }
        } else {
            barge_in_frames = 0;
            echo_frames = 0;
        }

        // 语音段结束 → 入队（推理中也允许收集，但播放中除外以防止回声误触发）
        // 三种触发条件: VAD静音 / ASR文本稳定(语义完成) / 10s安全超时
        bool asr_trigger = asr_endpoint && vad->speech_sample_count() > 24000; // 至少1.5s语音
        if (asr_trigger && !vad->segment_ready() && !asr_endpoint_logged_) {
            LOG_INFO("🔚 ASR端点: 文本稳定 {}帧 → \"{}\"", asr_stable_frames, asr_last_partial);
            asr_endpoint_logged_ = true;
        }
        if ((vad->segment_ready() || speech_timeout || asr_trigger) && (!busy || !playing)) {
            auto segment = vad->pop_segment();
            speech_timeout_warned_ = false;  // 新语音段入队，重置告警去重
            asr_endpoint_logged_ = false;

            // 检查最小长度：至少 0.5 秒
            float duration = (float)segment.size() / 16000.0f;
            if (duration < 0.5f) {
                std::cout << "   ⏭️ 语音段太短 (" << duration << "s)，跳过" << std::endl;
                if (stream_asr_.initialized()) stream_asr_.cancel();
                continue;
            }

            int gen = generation_.fetch_add(1) + 1;

            // 重置 ASR 稳定性检测（新语音段开始）
            asr_last_partial.clear();
            asr_stable_frames = 0;

            // 流式 ASR 最终识别（在 capture 线程完成，隐藏 ASR 延迟）
            std::string stream_text;
            if (stream_asr_.initialized()) {
                stream_text = stream_asr_.finalize();
            }

            // ── 垃圾识别过滤 ──
            // 文本级检测（纯标点/短ASCII/日韩文）+ 能量级检测
            bool is_garbage = utf8::is_garbage_text(stream_text);
            if (!is_garbage && !stream_text.empty()) {
                // 计算语音段平均 RMS 能量
                float sum_sq = 0.0f;
                for (auto s : segment) sum_sq += s * s;
                float avg_rms = std::sqrt(sum_sq / segment.size());

                // 能量极低（< 0.005）→ 几乎肯定是回声/静音
                if (avg_rms < 0.005f && stream_text.size() <= 10) {
                    is_garbage = true;
                }
                // 短文本 + 低于 min_energy_threshold → SenseVoice 静音幻觉
                else if (avg_rms < vad_cfg.min_energy_threshold
                         && stream_text.size() <= 6) {
                    is_garbage = true;
                }
            }
            if (is_garbage) {
                std::cout << "\n   🗑️  丢弃噪音段 #" << gen
                          << " (" << duration << "s)"
                          << " → \"" << stream_text << "\"" << std::endl;
                continue;
            }

            std::cout << "\n📝 捕获语音段 #" << gen
                      << " (" << duration << "s)"
                      << " → \"" << stream_text << "\"" << std::endl;

            // ── 声学情感分析 (Layer 3.3) ──────────────────
            VoiceEmotionResult voice_emo;
            if (!segment.empty()) {
                voice_emo = voice_emo_.analyze(
                    segment.data(), (int)segment.size(), 16000);
                if (voice_emo.confidence > 0.0f) {
                    std::cout << "   🎭 " << voice_emo.label
                              << " (" << voice_emo.detail << ")"
                              << " 置信度=" << (int)(voice_emo.confidence * 100) << "%"
                              << std::endl;
                }
            }

            {
                std::lock_guard<std::mutex> lk(queue_mutex_);
                Segment seg;
                seg.samples    = std::move(segment);
                seg.text       = stream_text;   // 预识别文本
                seg.voice_emo  = voice_emo;     // 声学情感
                seg.generation = gen;
                segment_queue_.push(std::move(seg));
            }
            queue_cv_.notify_one();
        }
    }

    pclose(pipe);

    // 通知 process 线程退出
    interactive_running_ = false;
    queue_cv_.notify_all();
}

// ── Process 线程 ─────────────────────────────────────

void VoicePipeline::process_loop()
{
    while (true) {
        Segment seg;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this] {
                return !segment_queue_.empty() || !interactive_running_;
            });

            if (!interactive_running_ && segment_queue_.empty()) {
                break;  // 退出
            }

            seg = std::move(segment_queue_.front());
            segment_queue_.pop();
        }

        // 标记处理中（capture 线程仍然收集新段，但播放中除外防止回声）
        process_busy_ = true;

        int my_gen = seg.generation;

        // 检查是否已有更新的语音段（当前段已过期）
        if (my_gen < generation_.load()) {
            std::cout << "   ⏭️ 语音段 #" << my_gen << " 已过期，丢弃" << std::endl;
            process_busy_ = false;
            continue;
        }

        // 1) 获取识别文本：优先使用流式 ASR 预识别结果
        std::string prompt;
        if (!seg.text.empty()) {
            // 流式 ASR 已在 capture 线程完成识别
            prompt = seg.text;
        } else {
            // 传统路径：写入 WAV → 离线 ASR
            const std::string wav_file = "temp_interactive_" + std::to_string(my_gen) + ".wav";
            if (!wav_utils::write_wav_float(wav_file, seg.samples, 16000)) {
                LOG_ERROR("   ❌ 写入 WAV 失败");
                process_busy_ = false;
                continue;
            }
            prompt = asr_.transcribe(wav_file);
            std::remove(wav_file.c_str());
        }

        if (prompt.empty()) {
            LOG_INFO("   ⚠️ 未识别到有效语音");
            process_busy_ = false;
            continue;
        }

        // 再次检查是否过期（ASR 可能耗时）
        if (my_gen < generation_.load()) {
            std::cout << "   ⏭️ 语音段 #" << my_gen << " 在 ASR 后过期" << std::endl;
            process_busy_ = false;
            continue;
        }

        // 3) 唤醒词检查（可选的）／声纹识别
        if (kws_.enabled() && !kws_.detect(prompt)) {
            LOG_INFO("   ⚠️ 未检测到唤醒词，跳过");
            process_busy_ = false;
            continue;
        }

        // 声纹识别（交互模式中可选，仅在声纹库有用户时启用）
        if (voiceprint_.has_any() && !seg.samples.empty()) {
            const std::string vp_wav = "temp_vp_" + std::to_string(my_gen) + ".wav";
            if (wav_utils::write_wav_float(vp_wav, seg.samples, 16000)) {
                auto id = voiceprint_.identify(vp_wav);
                std::remove(vp_wav.c_str());
                if (id.identified()) {
                    if (id.name != voiceprint_.active_speaker()) {
                        voiceprint_.set_active_speaker(id.name);
                    }
                }
            }
        }

        // 4) 推理：关键字技能 → ReAct → 直接 LLM
        auto t_pipeline_start = std::chrono::steady_clock::now();

        // 获取活跃用户的 system prompt（声纹识别出的用户）
        std::string speaker_prompt = voiceprint_.active_system_prompt();
        std::string context = memory_.get_context();
        std::string reply;

        // ── 策略 0: 关键字技能匹配（最优先）───────────
        SkillResult sr = skill_mgr_.detect_and_execute(prompt);
        std::string extra = SkillManager::get_system_context();
        if (!speaker_prompt.empty()) {
            extra = speaker_prompt + "\n" + extra;
        }
        bool skill_hit = sr.hit;
        if (skill_hit) {
            extra += "\n" + sr.result_text;
            std::cout << "   [Skill] \"" << prompt << "\" → " << sr.skill_name << std::endl;
            // 内容型技能（笑话/故事/诗词等）：直接返回，不经过LLM避免截断和延迟
            if (sr.direct) reply = sr.result_text;
        }

        auto t_after_skill = std::chrono::steady_clock::now();

        // 注入用户长期记忆
        extra = build_extra_context(extra, user_memory_, cfg_.memory_long_term_enabled);

        // ── 策略 1: ReAct 多步推理（skill 未命中时才用）─
        if (reply.empty() && react_ && !skill_hit) {
            auto t_react = std::chrono::steady_clock::now();
            int react_steps = 0;
            auto tools = skill_mgr_.collect_function_defs();
            if (!tools.empty()) {
                auto exec_fn = [this](const std::string& name, const nlohmann::json& args) {
                    return skill_mgr_.execute_tool(name, args, "");
                };
                std::string react_ctx = context.empty() ? extra : context + "\n" + extra;
                auto result = react_->run(prompt, tools, exec_fn, react_ctx, cfg_.react_max_steps);
                react_steps = (int)result.steps.size();
                if (result.success && !result.final_answer.empty()) {
                    reply = result.final_answer;
                }
            }
            auto t_react_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_react).count();
            std::cout << "   [ReAct] " << t_react_ms << "ms"
                      << " (" << react_steps << " steps)" << std::endl;
        }

        // ── 策略 2: 直接 LLM 回复 ───────────────────
        if (reply.empty()) {
            reply = llm_.chat(prompt, context, extra);
        }

        auto t_after_llm = std::chrono::steady_clock::now();

        if (reply.empty()) {
            process_busy_ = false;
            continue;
        }

        // 再次检查是否过期（ReAct/LLM 可能耗时较长）
        if (my_gen < generation_.load()) {
            std::cout << "   ⏭️ 语音段 #" << my_gen << " 在推理后过期，丢弃回复" << std::endl;
            process_busy_ = false;
            continue;
        }

        // 5.5) 质量优化：Multi-Agent 或 Reflection
        if (multi_agent_) {
            auto t_ma = std::chrono::steady_clock::now();
            std::string cmodel = cfg_.ma_critic_model.empty()
                ? cfg_.llm_model : cfg_.ma_critic_model;
            auto ma = multi_agent_->collaborate(
                prompt, reply, "", cfg_.system_prompt,
                cfg_.llm_model, cmodel, cfg_.ma_max_rounds);
            reply = ma.final_answer;
            auto t_ma_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_ma).count();
            std::cout << "   [MultiAgent] " << t_ma_ms << "ms" << std::endl;
        } else if (reflect_) {
            auto t_ref = std::chrono::steady_clock::now();
            auto r = reflect_->reflect(prompt, reply);
            reply = r.improved;
            auto t_ref_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_ref).count();
            std::cout << "   [Reflection] " << t_ref_ms << "ms" << std::endl;
        }

        auto t_after_quality = std::chrono::steady_clock::now();

        // 6) 回复截断
        if (cfg_.max_response_chars > 0 && (int)reply.size() > cfg_.max_response_chars * 3) {
            size_t cut = (size_t)cfg_.max_response_chars * 3;
            for (size_t off = 0; off < 60 && cut + off < reply.size(); off++) {
                size_t pos = cut + off;
                if (pos + 2 < reply.size() && (unsigned char)reply[pos] == 0xE3) {
                    unsigned char c1 = reply[pos+1], c2 = reply[pos+2];
                    if ((c1 == 0x80 && c2 == 0x82) ||   // 。
                        (c1 == 0xBC && c2 == 0x81) ||    // ！
                        (c1 == 0xBC && c2 == 0x9F)) {    // ？
                        reply = reply.substr(0, pos + 3);
                        std::cout << "   ✂️  回复截断为 " << reply.size()/3 << " 字" << std::endl;
                        break;
                    }
                }
            }
        }

        // 7) 更新记忆
        memory_.add(prompt, reply);
        save_all_memory();  // 持久化到磁盘

        reply = strip_emoji(reply);
        std::cout << "   🤖 回复: \"" << reply << "\"" << std::endl;

        // 7) TTS + 播放（传递声学情感用于韵律融合）
        const VoiceEmotionResult* ve = seg.voice_emo.confidence > 0.0f ? &seg.voice_emo : nullptr;
        if (cfg_.tts_backend == "piper") {
            // Piper：写出 WAV → 异步播放（可被打断）
            const std::string tts_wav = "temp_reply_interactive_piper.wav";
            if (tts_.synthesize(reply, tts_wav, prompt, ve)) {
                auto t_after_tts = std::chrono::steady_clock::now();

                // 播放前最后检查是否过期（推理中可能已有新语音段入队）
                if (my_gen < generation_.load()) {
                    std::cout << "   ⏭️ 语音段 #" << my_gen << " 在播放前过期" << std::endl;
                    std::remove(tts_wav.c_str());
                    process_busy_ = false;
                    continue;
                }
                LOG_INFO("   🔊 播放中...");
                is_playing_ = true;
                pid_t pid = AudioPlayer::play_async(tts_wav);
                player_pid_ = pid;
                if (pid > 0) {
                    int status;
                    waitpid(pid, &status, 0);
                    pid_t expected = pid;
                    player_pid_.compare_exchange_strong(expected, -1);
                }
                is_playing_ = false;
                std::remove(tts_wav.c_str());

                auto t_after_play = std::chrono::steady_clock::now();

                // ── 延迟汇总 ──
                auto skill_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_skill - t_pipeline_start).count();
                auto llm_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_llm - t_after_skill).count();
                auto qual_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_quality - t_after_llm).count();
                auto tts_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_tts - t_after_quality).count();
                auto play_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_play - t_after_tts).count();
                auto total_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_play - t_pipeline_start).count();
                LOG_INFO("⏱️  [延迟] 技能:{}ms | LLM:{}ms | 质检:{}ms | TTS:{}ms | 播放:{}ms | 总计:{}ms",
                         skill_ms, llm_ms, qual_ms, tts_ms, play_ms, total_ms);
            }
        } else {
            const std::string tts_file = "temp_reply_interactive_" + std::to_string(my_gen) + ".wav";
            if (!tts_.synthesize(reply, tts_file, prompt, ve)) {
                LOG_ERROR("   ❌ TTS 合成失败");
                process_busy_ = false;
                continue;
            }

            auto t_after_tts = std::chrono::steady_clock::now();

            // 最后检查（TTS 后）
            if (my_gen < generation_.load()) {
                std::cout << "   ⏭️ 语音段 #" << my_gen << " 在 TTS 后过期" << std::endl;
                std::remove(tts_file.c_str());
                process_busy_ = false;
                continue;
            }

            // 7) 异步播放（可被打断）
            LOG_INFO("   🔊 播放中...");
            is_playing_ = true;
            pid_t pid = AudioPlayer::play_async(tts_file);
            player_pid_ = pid;

            if (pid > 0) {
                int status;
                waitpid(pid, &status, 0);
                pid_t expected = pid;
                player_pid_.compare_exchange_strong(expected, -1);
            }

            is_playing_ = false;
            std::remove(tts_file.c_str());

            auto t_after_play = std::chrono::steady_clock::now();

            // ── 延迟汇总 ──
            auto skill_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_skill - t_pipeline_start).count();
            auto llm_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_llm - t_after_skill).count();
            auto qual_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_quality - t_after_llm).count();
            auto tts_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_tts - t_after_quality).count();
            auto play_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_play - t_after_tts).count();
            auto total_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_play - t_pipeline_start).count();
            LOG_INFO("⏱️  [延迟] 技能:{}ms | LLM:{}ms | 质检:{}ms | TTS:{}ms | 播放:{}ms | 总计:{}ms",
                     skill_ms, llm_ms, qual_ms, tts_ms, play_ms, total_ms);
        }

        // 处理完成 → capture 线程可以继续收集新语音段
        process_busy_ = false;
    }
}
