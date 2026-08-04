#pragma once
/**
 * 语音合成引擎 (TTS) — 双后端
 *
 * - espeak: libespeak-ng 直接调用
 * - piper:  神经网络 TTS，常驻 Python 进程（模型只加载一次）
 *
 * Python 对应: src/tts.py → TTSEngine
 */

#include <string>
#include <cstdio>
#include "voice_emotion.h"

class ProsodyController;

class TTSEngine {
public:
    /// @param rate            espeak 语速
    /// @param voice           espeak 音色
    /// @param backend         "espeak" / "piper" / "edge_tts"
    /// @param piper_model     Piper 模型路径 (.onnx)
    /// @param edge_tts_voice  Edge TTS 音色 (如 zh-CN-XiaoyiNeural)
    explicit TTSEngine(int rate = 200,
                       const std::string& voice = "cmn+f3",
                       const std::string& backend = "espeak",
                       const std::string& piper_model = "",
                       const std::string& edge_tts_voice = "zh-CN-XiaoyiNeural");
    ~TTSEngine();

    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;

    /// 初始化（Piper 后端会预加载模型）
    bool initialize();

    /// 文字 → 音频播放
    /// @param text         待合成文本 (UTF-8)
    /// @param output_path  输出 WAV 路径（espeak 后端使用，Piper 忽略）
    /// @param user_context 用户原始输入（用于韵律分析，可选）
    /// @param voice_emo    声学情感分析结果（Layer 3.3，可选）
    bool synthesize(const std::string& text, const std::string& output_path,
                    const std::string& user_context = "",
                    const VoiceEmotionResult* voice_emo = nullptr,
                    bool is_ssml = false);

private:
    int rate_;
    std::string voice_;
    std::string backend_;
    std::string piper_model_;
    std::string piper_script_;
    std::string edge_tts_voice_;
    std::string edge_tts_script_;
    bool initialized_ = false;

    // Piper 常驻进程
    FILE* piper_in_  = nullptr;   // 写文本到 Python 进程
    FILE* piper_out_ = nullptr;   // 读 PCM 数据
    int piper_sample_rate_ = 22050;

    // espeak
    bool init_espeak();
    bool synthesize_espeak(const std::string& text, const std::string& output_path);

    // piper
    bool init_piper();
    bool synthesize_piper(const std::string& text, const std::string& output_path);
    void shutdown_piper();

    // edge_tts
    bool init_edge_tts();
    bool synthesize_edge_tts(const std::string& text, const std::string& output_path,
                              bool is_ssml = false);

    /// 从 piper_out_ 读取指定长度数据
    bool read_exact(void* buf, size_t len);

    // 韵律控制
    class ProsodyController* prosody_ = nullptr;
    bool prosody_enabled_ = true;
};
