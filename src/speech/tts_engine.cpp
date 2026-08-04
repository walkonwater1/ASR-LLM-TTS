/**
 * 语音合成引擎 — 三后端（espeak-ng / Piper neural TTS / Edge TTS）
 *
 * espeak:   libespeak-ng 直接调用（快速但电音）
 * Piper:    常驻 Python 进程（模型只加载一次，后续调用低延迟）
 * edge_tts: 微软云 TTS，one-shot Python 子进程（音质最好）
 */

#include "tts_engine.h"
#include "prosody.h"
#include "wav_utils.h"

#include "espeak_min.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cctype>
#include <algorithm>
#include <unordered_map>
#include "logger.h"

// ── TTS 文本预处理 ────────────────────────────────────
//
// 问题: LLM 输出直接喂给 Piper，导致:
//   1. 阿拉伯数字发音奇怪（Piper 是按字面读的）
//   2. 符号/英文混排被读成乱码（°C, km/h 等）
//   3. 没有句末标点 → 断句不正常
//
// 解决: 在合成前清洗文本，转成干净的纯中文 + 标点

/// 阿拉伯数字 0-999 → 中文
static std::string num_to_chinese(int n)
{
    static const char* digits[] = {
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"
    };
    if (n == 0) return "零";
    if (n < 0) return "负" + num_to_chinese(-n);

    std::string result;

    if (n >= 1000) {
        // 只支持到 999，超出部分分节处理
        int q = n / 1000;
        result += num_to_chinese(q) + "千";
        n %= 1000;
        if (n > 0 && n < 100) result += "零";
    }

    if (n >= 100) {
        result += digits[n / 100] + std::string("百");
        n %= 100;
        if (n > 0 && n < 10) result += "零";
    }

    if (n >= 10) {
        // 十几的特殊处理: 12 → "十二" 不是 "一十二"
        if (n < 20 && !result.empty()) {
            result += "十";
        } else {
            result += digits[n / 10] + std::string("十");
        }
        n %= 10;
    }

    if (n > 0) {
        result += digits[n];
    }

    return result;
}

/// 替换字符串中的阿拉伯数字为中文读法
static std::string replace_numbers(const std::string& text)
{
    std::string result;
    result.reserve(text.size() * 2);

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // UTF-8 多字节字符：直接跳过
        if (c >= 0x80) {
            size_t start = i;
            // 跳过连续的多字节字符
            while (i < text.size() && (static_cast<unsigned char>(text[i]) >= 0x80)) {
                ++i;
            }
            result.append(text, start, i - start);
            continue;
        }

        // ASCII 数字
        if (std::isdigit(c)) {
            size_t start = i;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            int num = std::stoi(text.substr(start, i - start));
            result += num_to_chinese(num);
        }
        // 小数点
        else if (c == '.' && i + 1 < text.size()
                 && std::isdigit(static_cast<unsigned char>(text[i+1]))) {
            result += "点";
            ++i;
        }
        else {
            result += text[i];
            ++i;
        }
    }
    return result;
}

/// 判断 3 字节 UTF-8 字符是否为 CJK 汉字（排除 emoji/符号/平仮名等）
static bool is_cjk_char_3byte(const char* p)
{
    unsigned char b0 = p[0], b1 = p[1], b2 = p[2];
    int cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    // CJK Unified Ideographs:      U+4E00–U+9FFF
    // CJK Extension A:             U+3400–U+4DBF
    // CJK Compatibility Ideographs:U+F900–U+FAFF
    return (cp >= 0x4E00 && cp <= 0x9FFF)
        || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0xF900 && cp <= 0xFAFF);
}

/// 替换常见符号和英文缩写为中文
static std::string replace_symbols(const std::string& text)
{
    static const std::vector<std::pair<std::string, std::string>> replacements = {
        {"°C", "度"},
        {"℃", "度"},
        {"°F", "华氏度"},
        {"km/h", "公里每小时"},
        {"km", "公里"},
        {"cm", "厘米"},
        {"mm", "毫米"},
        {"m/s", "米每秒"},
        {"%", "百分之"},
        {"wttr.in", ""},
        {"http://", ""},
        {"https://", ""},
        {".com", ""},
        {".cn", ""},
        {".org", ""},
    };

    std::string result = text;
    for (auto& [pat, repl] : replacements) {
        size_t pos = 0;
        while ((pos = result.find(pat, pos)) != std::string::npos) {
            result.replace(pos, pat.length(), repl);
            pos += repl.length();
        }
    }

    // 去掉残留的纯英文单词（中音 Piper 模型无英文音素）
    // 保留中文 + 标点，ASCII 字母连续序列全部丢弃
    // 但先判断文本是否以中文为主 — 如果英文占比过高（如错误消息），
    // 跳过 ASCII 剥离，避免把 "Couldn't connect to server" 变成乱码
    std::string cleaned;
    cleaned.reserve(result.size());

    // 统计 CJK 字符数 vs ASCII 字母数
    size_t cjk_count = 0, ascii_alpha_count = 0;
    for (size_t j = 0; j < result.size(); ) {
        unsigned char cj = static_cast<unsigned char>(result[j]);
        if (cj >= 0xE0 && cj < 0xF0 && j + 2 < result.size()) {
            if (is_cjk_char_3byte(&result[j])) ++cjk_count;
            j += 3;
        } else if (cj >= 0x80) {
            size_t cl = 1;
            if (cj >= 0xF0) cl = 4;
            else if (cj >= 0xE0) cl = 3;
            else if (cj >= 0xC0) cl = 2;
            j += (j + cl <= result.size()) ? cl : 1;
        } else {
            if (std::isalpha(cj)) ++ascii_alpha_count;
            ++j;
        }
    }
    bool mostly_chinese = (cjk_count > 0) && (cjk_count >= ascii_alpha_count);

    size_t i = 0;
    while (i < result.size()) {
        unsigned char c = static_cast<unsigned char>(result[i]);

        // UTF-8 多字节字符：整体保留
        if (c >= 0x80) {
            size_t char_len = 1;
            if (c >= 0xF0)      char_len = 4;
            else if (c >= 0xE0) char_len = 3;
            else if (c >= 0xC0) char_len = 2;
            if (i + char_len <= result.size()) {
                cleaned.append(result, i, char_len);
                i += char_len;
            } else {
                cleaned += result[i];
                ++i;
            }
        }
        // ASCII 字母：仅当文本以中文为主时才丢弃
        else if (mostly_chinese && std::isalpha(c)) {
            while (i < result.size() && std::isalpha(static_cast<unsigned char>(result[i]))) {
                ++i;
            }
        }
        else {
            cleaned += result[i];
            ++i;
        }
    }

    return cleaned;
}

/// 确保句末有标点，改善断句
static std::string ensure_ending_punctuation(const std::string& text)
{
    if (text.empty()) return text;

    // 中文句末标点集合
    static const std::string endings = "。！？….!?~～）)」』》】\"'";

    char last = text.back();

    // 检查最后一个字符是否标点
    // UTF-8: 中文标点占用3字节，检查最后一个字节
    if (endings.find(last) != std::string::npos) {
        return text;
    }

    // 没有标点 → 加句号
    return text + "。";
}

/// 判断 UTF-8 字符是否可被 Piper 中文模型发音
static bool is_pronounceable(const char* p, size_t char_bytes)
{
    if (char_bytes == 3) {
        // CJK 汉字 → 可发音
        if (is_cjk_char_3byte(p)) return true;
        // 中文标点 → 保留（用于断句）
        unsigned char a = p[0], b = p[1], c = p[2];
        if (a == 0xE3 && b == 0x80) return (c == 0x82 || c == 0x81);  // 。，
        if (a == 0xEF && b == 0xBC) return (c == 0x81 || c == 0x9F || c == 0x8C || c == 0x9B); // ！？，；
        // 全角符号（，、：等）保留
        return false;
    }
    if (char_bytes == 4) return false;   // 补充平面 emoji → 无法发音
    if (char_bytes == 1) {
        unsigned char c = static_cast<unsigned char>(p[0]);
        // 允许 ASCII 标点、数字、字母（Piper 会自行处理）
        return std::isprint(c);
    }
    return false;
}

/// 去掉 TTS 无法发音的字符（emoji、特殊符号等）
static std::string remove_unpronounceable(const std::string& text)
{
    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_bytes = 1;
        if (c >= 0xF0 && i + 3 < text.size()) char_bytes = 4;
        else if (c >= 0xE0 && i + 2 < text.size()) char_bytes = 3;
        else if (c >= 0xC0 && i + 1 < text.size()) char_bytes = 2;

        if (is_pronounceable(&text[i], char_bytes)) {
            result.append(text, i, char_bytes);
        }
        i += char_bytes;
    }
    return result;
}

/// 在逗号过少的长句中插入逗号改善停顿
static std::string add_breathing_pauses(const std::string& text)
{
    // 中文标点 UTF-8 序列
    auto is_cn_punct = [](const char* p) -> bool {
        unsigned char a = p[0], b = p[1], c = p[2];
        if (a == 0xE3 && b == 0x80) return (c == 0x82 || c == 0x81); // 。，
        if (a == 0xEF && b == 0xBC) return (c == 0x81 || c == 0x9F || c == 0x8C || c == 0x9B); // ！？，；
        return false;
    };

    // 遍历文本，每 ~20 个 CJK 汉字后如果是连续汉字无标点，插入逗号
    std::string result;
    result.reserve(text.size() + text.size() / 10);

    size_t chars_since_pause = 0;  // 距离上次停顿的 CJK 汉字数

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_bytes = 1;
        if (c >= 0xF0 && i + 3 < text.size()) char_bytes = 4;
        else if (c >= 0xE0 && i + 2 < text.size()) char_bytes = 3;
        else if (c >= 0xC0 && i + 1 < text.size()) char_bytes = 2;

        if (char_bytes >= 3) {
            // 中文标点 → 重置计数器
            if (is_cn_punct(&text[i])) {
                chars_since_pause = 0;
            } else if (is_cjk_char_3byte(&text[i])) {
                // 只对真正的 CJK 汉字计数（排除 emoji/符号）
                ++chars_since_pause;
            }
            // else: 非 CJK 的多字节字符（emoji/符号）→ 不计数
        } else if (c == ',' || c == '.' || c == '!' || c == '?') {
            // ASCII 标点 → 重置
            chars_since_pause = 0;
        }
        // ASCII 字符不增加汉字计数

        result.append(text, i, char_bytes);
        i += char_bytes;

        // 每 20 个汉字无标点时插入逗号（在非标点位置之后）
        // 阈值设高避免打断词语（如"天气情况"、"提供"、"其他"等）
        if (chars_since_pause >= 20 && i < text.size()) {
            unsigned char nc = static_cast<unsigned char>(text[i]);
            // 确保后一个字符不是标点，也不是英文
            if (nc >= 0x80 || nc == ' ') {
                result += "\xEF\xBC\x8C"; // UTF-8 逗号 ，
                chars_since_pause = 0;     // 重置计数器
            }
        }
    }

    return result;
}

/// 综合预处理
static std::string preprocess_tts_text(const std::string& raw_text)
{
    std::string text = raw_text;

    // 0. 去掉 emoji/特殊符号（Piper 中文模型无法发音，且会干扰停顿计数）
    text = remove_unpronounceable(text);

    // 1. 替换符号（在数字转换之前，避免干扰）
    text = replace_symbols(text);

    // 2. 数字 → 中文
    text = replace_numbers(text);

    // 3. 补标点
    text = ensure_ending_punctuation(text);

    // 4. 长句加停顿
    text = add_breathing_pauses(text);

    // 5. 清理多余空格和连续标点
    // UTF-8 句号 。= E3 80 82, 逗号 ，= EF BC 8C, 感叹号 ！= EF BC 81
    std::string cleaned;
    cleaned.reserve(text.size());
    std::string prev_utf8; // track previous multi-byte punct
    char prev_byte = 0;

    auto is_cn_period = [](const char* p) -> bool {
        return (unsigned char)p[0] == 0xE3 && (unsigned char)p[1] == 0x80
            && (unsigned char)p[2] == 0x82;
    };
    auto is_cn_comma = [](const char* p) -> bool {
        return (unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBC
            && (unsigned char)p[2] == 0x8C;
    };
    auto is_cn_excl = [](const char* p) -> bool {
        return (unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBC
            && (unsigned char)p[2] == 0x81;
    };

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_bytes = 1;
        if (c >= 0xF0 && i + 3 < text.size()) char_bytes = 4;
        else if (c >= 0xE0 && i + 2 < text.size()) char_bytes = 3;
        else if (c >= 0xC0 && i + 1 < text.size()) char_bytes = 2;

        if (char_bytes >= 3 && i + 2 < text.size()
            && (is_cn_period(&text[i]) || is_cn_comma(&text[i]) || is_cn_excl(&text[i]))) {
            std::string current(text, i, char_bytes);
            if (current != prev_utf8) {
                cleaned += current;
                prev_utf8 = current;
            }
            // else: skip duplicate punctuation
        } else if (c == ' ' && (prev_byte == ' ' || prev_byte == 0)) {
            // skip duplicate spaces
        } else {
            // 保留整组 UTF-8 字节（中文等多字节字符）
            cleaned.append(text, i, char_bytes);
            prev_byte = c;
            prev_utf8.clear();
        }
        i += char_bytes;
    }

    return cleaned;
}

// ── espeak 回调 ─────────────────────────────────────

static std::vector<int16_t> g_tts_audio;

static int audio_callback(short* wav, int numsamples, espeak_EVENT* /*events*/)
{
    if (wav && numsamples > 0) {
        g_tts_audio.insert(g_tts_audio.end(), wav, wav + numsamples);
    }
    return 0;
}

// ── 路径工具 ─────────────────────────────────────────

static std::string expand_tilde(const std::string& path)
{
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) return path;
    if (path.size() == 1) return home;
    return std::string(home) + path.substr(1);
}

static bool file_exists(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string find_script(const std::string& name)
{
    std::string src_prefix   = "src/" + name;
    std::string scripts_path = "src/scripts/" + name;
    const char* all[] = { name.c_str(), src_prefix.c_str(), scripts_path.c_str(), nullptr };

    for (int i = 0; all[i]; ++i) {
        if (file_exists(all[i])) return all[i];
    }
    return name;  // fallback
}

// ── TTSEngine ────────────────────────────────────────

TTSEngine::TTSEngine(int rate, const std::string& voice,
                     const std::string& backend, const std::string& piper_model,
                     const std::string& edge_tts_voice)
    : rate_(rate)
    , voice_(voice)
    , backend_(backend)
    , piper_model_(piper_model)
    , edge_tts_voice_(edge_tts_voice)
{
    if (!piper_model_.empty()) {
        piper_model_ = expand_tilde(piper_model_);
    }

    ProsodyConfig pcfg;
    pcfg.base_rate = rate;
    pcfg.min_rate  = rate * 70 / 100;
    pcfg.max_rate  = rate * 140 / 100;
    prosody_ = new ProsodyController(pcfg);
}

TTSEngine::~TTSEngine()
{
    delete prosody_;
    if (backend_ == "piper") {
        shutdown_piper();
    }
    if (initialized_ && backend_ == "espeak") {
        espeak_Terminate();
    }
    // edge_tts: no persistent process to clean up
}

bool TTSEngine::initialize()
{
    if (backend_ == "piper") {
        return init_piper();
    } else if (backend_ == "edge_tts") {
        return init_edge_tts();
    } else {
        return init_espeak();
    }
}

bool TTSEngine::synthesize(const std::string& text, const std::string& output_path,
                            const std::string& user_context,
                            const VoiceEmotionResult* voice_emo)
{
    if (!initialized_) return false;

    // ── 韵律分析：文本情感 + 声学情感 → 融合 → 调整语速 + 增强文本 ──
    std::string synth_text = text;
    if (prosody_ && prosody_enabled_) {
        // 1) 文本情感检测
        auto prosody = prosody_->analyze(
            user_context.empty() ? text : user_context,  // 主要信号：用户说了什么
            text);  // 次要信号：LLM 回复（用于标点增强）

        // 2) 如果有声学情感 → 融合
        if (voice_emo && voice_emo->confidence > 0.0f) {
            int text_tone_id = 0;
            switch (prosody.tone) {
            case EmotionTone::HAPPY:      text_tone_id = 1; break;
            case EmotionTone::SAD:        text_tone_id = 2; break;
            case EmotionTone::EMPATHETIC: text_tone_id = 3; break;
            case EmotionTone::URGENT:     text_tone_id = 4; break;
            default: break;
            }

            auto fusion = fuse_emotions(*voice_emo, text_tone_id);

            // 融合后的语调 → 覆盖文本检测结果
            switch (fusion.tone_id) {
            case 1: prosody.tone = EmotionTone::HAPPY;      break;
            case 2: prosody.tone = EmotionTone::SAD;        break;
            case 3: prosody.tone = EmotionTone::EMPATHETIC; break;
            case 4: prosody.tone = EmotionTone::URGENT;     break;
            default: prosody.tone = EmotionTone::NEUTRAL;   break;
            }
            prosody.adjusted_rate = prosody_->rate_for_tone(prosody.tone);
            prosody.tone_label    = prosody_->label_for_tone(prosody.tone);

            std::cout << "   [韵律] 🎤 " << fusion.diagnostic
                      << " → 语速" << prosody.adjusted_rate << std::endl;
        } else {
            std::cout << "   [韵律] " << prosody.tone_label
                      << " → 语速" << prosody.adjusted_rate
                      << " (纯文本检测)" << std::endl;
        }

        // 3) 文本增强（标点）
        synth_text = prosody.enhanced_text;

        // 4) 设置 espeak 语速（edge_tts 无需设置）
        if (backend_ != "piper" && backend_ != "edge_tts") {
            espeak_SetParameter(espeakRATE, prosody.adjusted_rate, 0);
            rate_ = prosody.adjusted_rate;
        }
    }

    auto t_tts_start = std::chrono::steady_clock::now();

    bool ok = false;
    if (backend_ == "piper") {
        ok = synthesize_piper(synth_text, output_path);
    } else if (backend_ == "edge_tts") {
        ok = synthesize_edge_tts(synth_text, output_path);
    } else {
        ok = synthesize_espeak(synth_text, output_path);
    }

    auto t_tts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_tts_start).count();
    std::cout << "   [TTS] " << backend_ << " " << t_tts_ms << "ms"
              << (ok ? "" : " ❌") << std::endl;

    return ok;
}

// ── espeak 后端 ─────────────────────────────────────

bool TTSEngine::init_espeak()
{
    std::cout << "[TTS] 初始化 espeak-ng ... " << std::flush;

    int sample_rate = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, nullptr, 0);
    if (sample_rate <= 0) {
        std::cerr << "❌ espeak_Initialize 失败 (返回: " << sample_rate << ")" << std::endl;
        return false;
    }
    std::cout << "(" << sample_rate << "Hz) " << std::flush;

    espeak_SetVoiceByName(voice_.c_str());
    espeak_SetParameter(espeakRATE, rate_, 0);

    initialized_ = true;
    LOG_INFO("✅");
    return true;
}

bool TTSEngine::synthesize_espeak(const std::string& text, const std::string& output_path)
{
    g_tts_audio.clear();
    espeak_SetSynthCallback(audio_callback);

    espeak_ERROR err = espeak_Synth(text.c_str(), text.size() + 1,
                                    0, POS_CHARACTER, 0,
                                    espeakCHARS_UTF8, nullptr, nullptr);
    if (err != EE_OK) {
        LOG_ERROR("[TTS] espeak_Synth 失败");
        return false;
    }

    espeak_Synchronize();

    if (g_tts_audio.empty()) {
        LOG_ERROR("[TTS] 合成结果为空");
        return false;
    }

    return wav_utils::write_wav(output_path, g_tts_audio);
}

// ── Piper 后端（常驻进程）──────────────��─────────────

bool TTSEngine::init_piper()
{
    std::string voice_name = piper_model_;
    // 从路径提取语音名（如 zh_CN-huayan-medium）
    auto last_slash = voice_name.find_last_of('/');
    if (last_slash != std::string::npos) voice_name = voice_name.substr(last_slash + 1);
    auto dot = voice_name.find(".onnx");
    if (dot != std::string::npos) voice_name = voice_name.substr(0, dot);
    std::cout << "[TTS] Piper (" << voice_name << ") ... " << std::flush;

    // 找 conda Python
    std::string python = "python3";
    const char* home = std::getenv("HOME");
    const char* conda_prefix = std::getenv("CONDA_PREFIX");

    std::vector<std::string> py_candidates;
    if (conda_prefix) py_candidates.push_back(std::string(conda_prefix) + "/bin/python3");
    if (home) {
        py_candidates.push_back(std::string(home) + "/miniconda3/envs/chatAudio/bin/python3");
        py_candidates.push_back(std::string(home) + "/miniconda3/bin/python3");
    }
    py_candidates.push_back("/usr/bin/python3");

    for (const auto& p : py_candidates) {
        if (file_exists(p)) { python = p; break; }
    }

    // 找 piper_server.py
    piper_script_ = find_script("piper_server.py");
    if (!file_exists(piper_script_)) {
        std::cerr << "❌ 脚本未找到: " << piper_script_ << std::endl;
        return false;
    }

    if (!file_exists(piper_model_)) {
        std::cerr << "❌ 模型未找到: " << piper_model_ << std::endl;
        return false;
    }

    // 启动常驻 Python 进程（双向管道）
    std::string cmd = "HF_ENDPOINT=https://hf-mirror.com "
                    + python + " -u " + piper_script_
                    + " \"" + piper_model_ + "\"";

    piper_in_ = popen(cmd.c_str(), "w");   // 我们往 Python 写文本
    if (!piper_in_) {
         // popen "w" 只能写，不能读。需要双向管道用 pipe+fdopen。
         // 改用 fork+pipe 方案。
    }

    // 不能用 popen 做双向！需要用 pipe + fork
    // 重新实现...
    LOG_WARN("⚠️  需要双向管道，改用 fork/pipe 方案");

    // 先清理
    if (piper_in_) { pclose(piper_in_); piper_in_ = nullptr; }

    int to_child[2];   // 父写 → 子读 (stdin)
    int from_child[2]; // 子写 → 父读 (stdout)

    if (pipe(to_child) < 0 || pipe(from_child) < 0) {
        LOG_ERROR("❌ pipe 创建失败");
        return false;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // ── 子进程 ──
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);

        // 清除 LD_LIBRARY_PATH（避免 conda ALSA 冲突）
        unsetenv("LD_LIBRARY_PATH");
        setenv("HF_ENDPOINT", "https://hf-mirror.com", 1);

        // 默认使用 g2pw 精准音素模式（多音字消歧），可通过环境变量覆盖
        setenv("PIPER_PHONEME_MODE", "accurate", 0);  // 0 = 不覆盖已有值

        execlp(python.c_str(), python.c_str(), "-u",
               piper_script_.c_str(), piper_model_.c_str(), nullptr);
        _exit(1);
    }

    // ── 父进程 ──
    close(to_child[0]);
    close(from_child[1]);

    piper_in_  = fdopen(to_child[1], "w");
    piper_out_ = fdopen(from_child[0], "r");

    if (!piper_in_ || !piper_out_) {
        LOG_ERROR("❌ fdopen 失败");
        return false;
    }

    // 读第一行 stderr 输出的 JSON 元数据（通过 pipe 劫持？不，stderr 不走 pipe）
    // 直接硬编码已知参数
    piper_sample_rate_ = 22050;

    initialized_ = true;
    LOG_INFO("✅ (模型已预加载)");
    return true;
}

void TTSEngine::shutdown_piper()
{
    if (piper_in_) {
        fclose(piper_in_);   // 关闭 stdin → Python 收到 EOF → 退出
        piper_in_ = nullptr;
    }
    if (piper_out_) {
        fclose(piper_out_);
        piper_out_ = nullptr;
    }
}

bool TTSEngine::read_exact(void* buf, size_t len)
{
    size_t total = 0;
    auto* p = static_cast<char*>(buf);
    while (total < len) {
        size_t n = fread(p + total, 1, len - total, piper_out_);
        if (n == 0) {
            if (ferror(piper_out_)) {
                LOG_ERROR("[TTS] 读取 Piper 输出失败");
            }
            return false;
        }
        total += n;
    }
    return true;
}

bool TTSEngine::synthesize_piper(const std::string& text, const std::string& output_path)
{
    if (!piper_in_ || !piper_out_) return false;

    // 0. 文本预处理：数字→中文、符号清洗、补标点
    std::string cleaned = preprocess_tts_text(text);
    if (cleaned != text) {
        std::cout << "   [TTS] 预处理: \"" << text << "\" → \"" << cleaned << "\"" << std::endl;
    }

    // 1. 发文本给 Python 进程
    fputs(cleaned.c_str(), piper_in_);
    fputc('\n', piper_in_);
    fflush(piper_in_);

    // 2. 读 4 字节长度头（大端 uint32）
    uint32_t pcm_len_be = 0;
    if (!read_exact(&pcm_len_be, 4)) return false;
    uint32_t pcm_len = (pcm_len_be >> 24) | ((pcm_len_be >> 8) & 0xFF00)
                     | ((pcm_len_be << 8) & 0xFF0000) | (pcm_len_be << 24);

    if (pcm_len == 0) return false;

    // 3. 读 PCM 数据
    std::vector<char> pcm(pcm_len);
    if (!read_exact(pcm.data(), pcm_len)) return false;

    // 4. 如果指定了输出路径 → 写入 WAV 文件（供调用方异步播放/打断）
    if (!output_path.empty()) {
        int num_samples = pcm_len / 2;  // S16_LE → int16_t samples
        std::vector<int16_t> audio(num_samples);
        std::memcpy(audio.data(), pcm.data(), pcm_len);
        return wav_utils::write_wav(output_path, audio, piper_sample_rate_);
    }

    // 5. 管道播放（默认，阻塞直到播完）
    std::string aplay_cmd = "env -u LD_LIBRARY_PATH aplay -q -f S16_LE -r "
                          + std::to_string(piper_sample_rate_) + " -c 1";

    FILE* aplay = popen(aplay_cmd.c_str(), "w");
    if (!aplay) {
        LOG_ERROR("[TTS] 启动 aplay 失败");
        return false;
    }

    fwrite(pcm.data(), 1, pcm_len, aplay);
    fflush(aplay);

    int status = pclose(aplay);
    return status == 0;
}

// ── Edge TTS 后端（微软云 TTS，one-shot 子进程）────────

bool TTSEngine::init_edge_tts()
{
    std::cout << "[TTS] Edge TTS (" << edge_tts_voice_ << ") ... " << std::flush;

    // 找到 edge_tts_cli.py 脚本（转为绝对路径）
    std::string script = find_script("edge_tts_cli.py");
    if (!file_exists(script)) {
        std::cerr << "❌ 脚本未找到: " << script << std::endl;
        return false;
    }

    // 转为绝对路径，避免 CWD 相关的问题
    char abs_path[4096];
    if (realpath(script.c_str(), abs_path)) {
        edge_tts_script_ = abs_path;
    } else {
        edge_tts_script_ = script;
    }
    std::cout << "(" << edge_tts_script_ << ") " << std::flush;

    // 验证 Python 环境和 edge_tts 库可用
    std::string test_cmd = "python3 " + edge_tts_script_ + " --help > /dev/null 2>&1";
    int ret = system(test_cmd.c_str());
    if (ret != 0) {
        std::cerr << "❌ edge_tts 不可用 (exit=" << ret << ", 请确认: pip install edge-tts)" << std::endl;
        return false;
    }

    initialized_ = true;
    LOG_INFO("✅");
    return true;
}

bool TTSEngine::synthesize_edge_tts(const std::string& text, const std::string& output_path)
{
    if (output_path.empty()) {
        LOG_ERROR("[TTS] edge_tts 需要指定 output_path");
        return false;
    }

    // 预处理文本
    std::string cleaned = preprocess_tts_text(text);
    if (cleaned != text) {
        std::cout << "   [TTS] 预处理: \"" << text << "\" → \"" << cleaned << "\"" << std::endl;
    }

    // 调用 one-shot 子进程
    // 注意: 不依赖 pclose() 返回值，因为 voice_pipeline 可能有 SIGCHLD 干扰
    // 直接检查输出文件是否生成成功即可
    std::ostringstream cmd;
    cmd << "python3 " << edge_tts_script_
        << " --text " << std::quoted(cleaned)
        << " --voice " << edge_tts_voice_
        << " --output " << output_path
        << " 2>/dev/null";

    int ret = system(cmd.str().c_str());
    // ret=-1 可能只是 SIGCHLD 干扰，不一定是真正的失败
    // 关键: 检查输出文件是否生成

    // 验证输出文件
    if (!file_exists(output_path)) {
        LOG_ERROR("[TTS] edge_tts 输出文件未生成: " + output_path);
        return false;
    }

    return true;
}
