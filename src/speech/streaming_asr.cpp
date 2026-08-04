/**
 * 流式语音识别引擎 (Streaming ASR) — 实现
 *
 * Layer 3.2: 语音交互深化 — 边说边识别
 *
 * 双后端实现:
 *   online  — sherpa-onnx OnlineRecognizer (Zipformer transducer)
 *             真正的逐帧增量解码，部分结果实时更新
 *   chunked — SenseVoice 离线模型 + 周期性增量重识别
 *             每 0.5s 对累积音频做一次完整识别，渐进式结果
 */

#include "streaming_asr.h"
#include "asr_engine.h"

#ifdef SHERPA_ONNX_AVAILABLE
  #include "sherpa-onnx/c-api/c-api.h"
#endif

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "logger.h"
#include "wav_utils.h"
#include "utf8_utils.h"

// ── StreamingASR 构造/析构 ────────────────────────────

StreamingASR::StreamingASR() = default;

StreamingASR::~StreamingASR()
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (online_rec_) {
        if (online_stream_) {
            SherpaOnnxDestroyOnlineStream(online_stream_);
        }
        SherpaOnnxDestroyOnlineRecognizer(online_rec_);
    }
#endif
    if (chunked_engine_) {
        delete static_cast<ASREngine*>(chunked_engine_);
    }
}

// ── 初始化 ────────────────────────────────────────────

bool StreamingASR::initialize(const StreamingASRConfig& cfg)
{
    cfg_ = cfg;

    std::cout << "[StreamASR] 初始化 后端=" << cfg_.backend
              << " 模型=" << cfg_.model_path << std::endl;

    if (cfg_.backend == "online") {
        if (init_online()) {
            initialized_ = true;
            return true;
        }
        // 降级
        LOG_WARN("   ⚠️ online 后端不可用，降级到 chunked");
        cfg_.backend = "chunked";
    }

    if (init_chunked()) {
        initialized_ = true;
        return true;
    }

    return false;
}

bool StreamingASR::init_online()
{
#ifdef SHERPA_ONNX_AVAILABLE
    // 检查模型文件是否存在
    std::string encoder = cfg_.model_path + "/encoder-epoch-99-avg-1.int8.onnx";
    std::string decoder = cfg_.model_path + "/decoder-epoch-99-avg-1.onnx";
    std::string joiner  = cfg_.model_path + "/joiner-epoch-99-avg-1.int8.onnx";
    std::string tokens  = cfg_.model_path + "/tokens.txt";

    auto fexists = [](const std::string& p) -> bool {
        FILE* f = fopen(p.c_str(), "rb");
        if (f) { fclose(f); return true; }
        return false;
    };

    if (!fexists(encoder) || !fexists(decoder) ||
        !fexists(joiner)  || !fexists(tokens)) {
        std::cerr << "   ❌ online 模型文件不全: " << cfg_.model_path << std::endl;
        LOG_ERROR("   需要: encoder/decoder/joiner .onnx + tokens.txt");
        LOG_ERROR("   下载: github.com/k2-fsa/sherpa-onnx/releases");
        return false;
    }

    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.feat_config.sample_rate = cfg_.sample_rate;
    config.feat_config.feature_dim = 80;

    // 将字符串存到 static 以保证 config 内部指针有效
    static std::string s_enc, s_dec, s_joi, s_tok, s_method;
    s_enc = encoder; s_dec = decoder; s_joi = joiner; s_tok = tokens;
    s_method = cfg_.decoding_method;

    config.model_config.transducer.encoder = s_enc.c_str();
    config.model_config.transducer.decoder = s_dec.c_str();
    config.model_config.transducer.joiner  = s_joi.c_str();
    config.model_config.tokens = s_tok.c_str();
    config.model_config.provider = "cpu";
    config.model_config.num_threads = cfg_.num_threads;
    config.decoding_method = s_method.c_str();

    online_rec_ = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!online_rec_) {
        LOG_ERROR("   ❌ 创建 online recognizer 失败");
        return false;
    }

    LOG_INFO("   ✅ online 流式后端就绪 (transducer)");
    return true;
#else
    LOG_ERROR("   ❌ sherpa-onnx 未编译，online 后端不可用");
    return false;
#endif
}

bool StreamingASR::init_chunked()
{
    AsrModelType mt = AsrModelType::SENSE_VOICE;
    if (cfg_.model_type == "zipformer_ctc") mt = AsrModelType::ZIPFORMER_CTC;

    auto* asr = new ASREngine(cfg_.model_path, mt);
    if (!asr->initialize()) {
        LOG_ERROR("   ❌ chunked 后端初始化失败");
        delete asr;
        return false;
    }

    chunked_engine_ = asr;
    // 流式部分识别不需要打印 [ASR]，最终结果才打印（由调用方控制）
    asr->set_quiet(true);
    audio_buf_.reserve(cfg_.sample_rate * 30);  // 预分配 30 秒
    std::cout << "   ✅ chunked 流式后端就绪 "
              << "(最小" << cfg_.min_chunk_seconds << "秒触发, "
              << "间隔" << cfg_.chunk_interval << "秒)" << std::endl;
    return true;
}

// ── 单次语音输入 ──────────────────────────────────────

void StreamingASR::start_utterance()
{
    // ── 重置状态 ──────────────────────────────────────
    partial_text_.clear();
    online_partial_.clear();
    last_chunk_time_ = 0.0f;
    chunk_count_     = 0;
    audio_buf_.clear();

#ifdef SHERPA_ONNX_AVAILABLE
    // 如果有旧的 stream，先销毁再重建
    if (online_stream_) {
        SherpaOnnxDestroyOnlineStream(online_stream_);
        online_stream_ = nullptr;
    }
    if (online_rec_) {
        online_stream_ = SherpaOnnxCreateOnlineStream(online_rec_);
        if (!online_stream_) {
            LOG_WARN("   ⚠️ 创建 online stream 失败");
        }
    }
#endif
}

void StreamingASR::feed(const float* samples, int n)
{
    if (!initialized_) return;

#ifdef SHERPA_ONNX_AVAILABLE
    if (online_stream_) {
        // online 后端：直接送入 sherpa-onnx 流式解码器
        SherpaOnnxOnlineStreamAcceptWaveform(
            online_stream_, cfg_.sample_rate, samples, n);

        // 尝试解码已就绪的帧
        decode_online();
        return;
    }
#endif

    // chunked 后端：累积音频
    audio_buf_.insert(audio_buf_.end(), samples, samples + n);

    float dur = audio_duration();

    // 达到最小长度 && 距上次识别超过间隔 → 触发部分识别
    if (dur >= cfg_.min_chunk_seconds &&
        (dur - last_chunk_time_) >= cfg_.chunk_interval) {
        run_chunked_partial();
        last_chunk_time_ = dur;
    }
}

const char* StreamingASR::partial()
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (online_stream_) {
        // 再尝试一次解码（可能有新帧就绪）
        decode_online();
        return online_partial_.empty() ? nullptr : online_partial_.c_str();
    }
#endif
    return partial_text_.empty() ? nullptr : partial_text_.c_str();
}

std::string StreamingASR::finalize()
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (online_stream_) {
        // 通知输入结束
        SherpaOnnxOnlineStreamInputFinished(online_stream_);

        // 解码剩余所有帧
        // 循环直到 stream 没有更多就绪帧
        for (int i = 0; i < 100; ++i) {  // 安全上限
            decode_online();
            // 检查是否还有更多帧（通过尝试获取结果看是否有变化）
            const SherpaOnnxOnlineRecognizerResult* r =
                SherpaOnnxGetOnlineStreamResult(online_rec_, online_stream_);
            if (r && r->text) {
                std::string new_text = r->text;
                SherpaOnnxDestroyOnlineRecognizerResult(r);
                if (new_text == online_partial_) {
                    break;  // 不再变化
                }
                online_partial_ = new_text;
            } else {
                break;
            }
        }

        std::string result = online_partial_;

        // 清理 stream（下次 start_utterance 时重建）
        SherpaOnnxDestroyOnlineStream(online_stream_);
        online_stream_ = nullptr;
        online_partial_.clear();

        return result;
    }
#endif

    // chunked 后端：最终完整识别
    return run_chunked_final();
}

void StreamingASR::cancel()
{
    partial_text_.clear();
    online_partial_.clear();
    last_chunk_time_ = 0.0f;
    chunk_count_     = 0;
    audio_buf_.clear();

#ifdef SHERPA_ONNX_AVAILABLE
    if (online_stream_) {
        SherpaOnnxDestroyOnlineStream(online_stream_);
        online_stream_ = nullptr;
    }
#endif
}

float StreamingASR::audio_duration() const
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (online_stream_) {
        // online 后端不跟踪总时长，返回 0
        return 0.0f;
    }
#endif
    return (float)audio_buf_.size() / (float)cfg_.sample_rate;
}

// ── online 后端内部方法 ───────────────────────────────

void StreamingASR::decode_online()
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (!online_rec_ || !online_stream_) return;

    // 解码所有就绪的帧
    while (SherpaOnnxIsOnlineStreamReady(online_rec_, online_stream_)) {
        SherpaOnnxDecodeOnlineStream(online_rec_, online_stream_);
    }

    // 获取最新部分结果
    const SherpaOnnxOnlineRecognizerResult* r =
        SherpaOnnxGetOnlineStreamResult(online_rec_, online_stream_);
    if (r && r->text) {
        std::string new_text = r->text;
        // 去掉 SenseVoice 样式标签 <|...|>（可能出现在某些模型中）
        auto pos = new_text.rfind('>');
        if (pos != std::string::npos && pos > 0 && new_text[0] == '<') {
            new_text = new_text.substr(pos + 1);
        }
        if (new_text != online_partial_) {
            // 过滤噪声幻觉（纯标点/短ASCII/日韩文）
            if (!utf8::is_garbage_text(new_text)) {
                online_partial_ = new_text;
                if (partial_cb_) {
                    partial_cb_(online_partial_.c_str());
                }
            }
        }
        SherpaOnnxDestroyOnlineRecognizerResult(r);
    }
#endif
}

// ── chunked 后端内部方法 ──────────────────────────────

void StreamingASR::run_chunked_partial()
{
    if (audio_buf_.empty()) return;

    auto* asr = static_cast<ASREngine*>(chunked_engine_);
    if (!asr) return;

    chunk_count_++;

    // 写出临时 WAV
    const std::string wav = "temp_streaming_chunk.wav";
    if (!wav_utils::write_wav_float(wav, audio_buf_, cfg_.sample_rate)) {
        return;
    }

    // 调用离线 ASR
    std::string text = asr->transcribe(wav);
    std::remove(wav.c_str());

    // 过滤噪声幻觉（纯标点/短ASCII/日韩文）
    bool garbage = utf8::is_garbage_text(text);

    // 保持最长的非垃圾部分识别结果
    if (!garbage && !text.empty() && text.size() > partial_text_.size()) {
        partial_text_ = text;
        if (partial_cb_) {
            partial_cb_(partial_text_.c_str());
        }
    }

    if (!garbage) {
        // 同一行更新，\r 回到行首覆盖上次输出
        std::cout << "\r   🎤 [" << audio_duration() << "s] " << text
                  << "                    " << std::flush;
    }
}

std::string StreamingASR::run_chunked_final()
{
    if (audio_buf_.empty()) return "";

    auto* asr = static_cast<ASREngine*>(chunked_engine_);
    if (!asr) return "";

    float dur = audio_duration();

    // 写出完整的临时 WAV
    const std::string wav = "temp_streaming_final.wav";
    if (!wav_utils::write_wav_float(wav, audio_buf_, cfg_.sample_rate)) {
        return "";
    }

    std::string text = asr->transcribe(wav);
    std::remove(wav.c_str());

    // 清理缓冲区
    audio_buf_.clear();
    partial_text_.clear();
    last_chunk_time_ = 0.0f;
    chunk_count_     = 0;

    // 清除同行的部分识别残留，换行输出最终结果
    std::cout << "\r\033[K   🎤 [最终 " << dur << "s] " << text << std::endl;
    return text;
}
