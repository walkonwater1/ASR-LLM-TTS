#include "logger.h"
/**
 * 管线配置 — JSON 文件加载
 *
 * 使用 nlohmann/json（项目已有依赖）解析配置文件。
 * 未在文件中出现的键保持默认值，实现最小化配置覆盖。
 */

#include "config.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>

using json = nlohmann::json;

// ── 辅助：安全读 JSON 字段，不存在则跳过 ────────────

template <typename T>
static void try_get(const json& j, const char* key, T& target)
{
    if (j.contains(key)) {
        auto& val = j[key];
        if (!val.is_null()) {
            if constexpr (std::is_same_v<T, int>)
                target = val.get<int>();
            else if constexpr (std::is_same_v<T, bool>)
                target = val.get<bool>();
            else if constexpr (std::is_same_v<T, float>)
                target = val.get<float>();
            else if constexpr (std::is_same_v<T, std::string>)
                target = val.get<std::string>();
        }
    }
}

bool PipelineConfig::load_from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "⚠️ 配置文件 JSON 解析失败: " << e.what() << std::endl;
        return false;
    }

    // ── ASR ────────────────────────────────────────
    if (j.contains("asr")) {
        auto& a = j["asr"];
        try_get(a, "model_path", asr_model_path);
        try_get(a, "model_type", asr_model_type);
    }

    // ── LLM ────────────────────────────────────────
    if (j.contains("llm")) {
        auto& l = j["llm"];
        try_get(l, "host",          ollama_host);
        try_get(l, "model",         llm_model);
        try_get(l, "system_prompt", system_prompt);
    }

    // ── TTS ────────────────────────────────────────
    if (j.contains("tts")) {
        try_get(j["tts"], "rate",            tts_rate);
        try_get(j["tts"], "voice",           tts_voice);
        try_get(j["tts"], "backend",         tts_backend);
        try_get(j["tts"], "piper_model",     piper_model_path);
        try_get(j["tts"], "edge_tts_voice",  edge_tts_voice);
    }

    // ── 唤醒词 ─────────────────────────────────────
    if (j.contains("wake_word")) {
        try_get(j["wake_word"], "keyword", wake_word);
    }

    // ── 声纹验证 ───────────────────────────────────
    if (j.contains("speaker_verification")) {
        auto& s = j["speaker_verification"];
        try_get(s, "enroll_dir", sv_enroll_dir);
        try_get(s, "threshold",  sv_threshold);
    }

    // ── 音频 ───────────────────────────────────────
    if (j.contains("audio")) {
        try_get(j["audio"], "sample_rate", sample_rate);
    }

    // ── VAD ────────────────────────────────────────
    if (j.contains("vad")) {
        auto& v = j["vad"];
        try_get(v, "backend",            vad_backend);
        try_get(v, "energy_threshold",   vad_energy_threshold);
        try_get(v, "min_speech_frames",  vad_min_speech_frames);
        try_get(v, "min_silence_frames", vad_min_silence_frames);
        try_get(v, "pre_speech_frames",  vad_pre_speech_frames);
        try_get(v, "adaptive_factor",    vad_adaptive_factor);
        try_get(v, "min_energy",         vad_min_energy);
        try_get(v, "cooldown_frames",    vad_cooldown_frames);
    }

    // ── 交互模式 ───────────────────────────────────
    if (j.contains("interactive")) {
        auto& it = j["interactive"];
        try_get(it, "barge_in_enabled",      barge_in_enabled);
        try_get(it, "barge_in_energy_ratio", barge_in_energy_ratio);
        try_get(it, "max_response_chars",    max_response_chars);
    }

    // ── 对话记忆 ───────────────────────────────────
    if (j.contains("memory")) {
        auto& m = j["memory"];
        try_get(m, "max_rounds", max_rounds);
        try_get(m, "max_tokens", max_tokens);
        try_get(m, "persist_dir", memory_persist_dir);
        try_get(m, "long_term_enabled", memory_long_term_enabled);
        try_get(m, "auto_extract", memory_auto_extract);
    }

    // ── Function Calling ────────────────────────────
    if (j.contains("function_calling")) {
        auto& fc = j["function_calling"];
        try_get(fc, "enabled", fc_enabled);
        try_get(fc, "model",   fc_model);
    }

    // ── ReAct ────────────────────────────────────────
    if (j.contains("react")) {
        auto& r = j["react"];
        try_get(r, "enabled",    react_enabled);
        try_get(r, "max_steps",  react_max_steps);
    }

    // ── Reflection ───────────────────────────────────
    if (j.contains("reflection")) {
        auto& r = j["reflection"];
        try_get(r, "enabled", reflect_enabled);
        try_get(r, "model",   reflect_model);
    }

    // ── Multi-Agent ──────────────────────────────────
    if (j.contains("multi_agent")) {
        auto& m = j["multi_agent"];
        try_get(m, "enabled",       multi_agent_enabled);
        try_get(m, "critic_model",  ma_critic_model);
        try_get(m, "max_rounds",    ma_max_rounds);
    }

    // ── 流式 ASR ────────────────────────────────────
    if (j.contains("streaming_asr")) {
        auto& s = j["streaming_asr"];
        try_get(s, "enabled",   streaming_asr_enabled);
        try_get(s, "backend",   streaming_asr_backend);
        try_get(s, "model",     streaming_asr_model);
        try_get(s, "min_chunk", streaming_min_chunk);
        try_get(s, "chunk_intv", streaming_chunk_intv);
    }

    // ── 技能 ────────────────────────────────────────
    // ── Embedding (Layer 4.2) ──────────────────────────
    if (j.contains("embedding")) {
        auto& e = j["embedding"];
        try_get(e, "backend",   embedding_backend);
        try_get(e, "model_dir", embedding_model_dir);
    }

    if (j.contains("skills")) {
        auto& s = j["skills"];
        if (s.contains("weather")) {
            try_get(s["weather"], "enabled", skill_weather);
        }
        if (s.contains("time")) {
            try_get(s["time"], "enabled", skill_time);
        }
        if (s.contains("web_search")) {
            try_get(s["web_search"], "enabled", skill_web_search);
        }
        if (s.contains("rag")) {
            auto& r = s["rag"];
            try_get(r, "enabled",  skill_rag);
            try_get(r, "docs_dir", rag_docs_dir);
        }
    }

    return true;
}

std::string PipelineConfig::auto_load_path()
{
    // 按优先级查找 config.json:
    //   1. 当前工作目录
    //   2. 可执行文件所在目录
    //   3. CONFIG_PATH 环境变量

    const std::string filename = "config.json";

    // 1. 当前目录
    {
        std::ifstream f(filename);
        if (f.is_open()) return filename;
    }

    // 2. 环境变量
    const char* env = std::getenv("VOICE_PIPELINE_CONFIG");
    if (env && env[0] != '\0') {
        std::ifstream f(env);
        if (f.is_open()) return env;
    }

    // 均未找到
    return "";
}
