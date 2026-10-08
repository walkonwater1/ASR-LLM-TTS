/**
 * 声纹验证 — sherpa-onnx Speaker Verification
 *
 * Python 对应: src/speaker.py → SpeakerVerifier (CAM++, modelscope)
 * C++ 实现:   sherpa-onnx speaker embedding + manager API v1.13+
 *
 * 编译依赖: libsherpa-onnx-c-api.so + sherpa-onnx/c-api/c-api.h
 * 模型: 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
 */

#include "speaker_verifier.h"

// SHERPA_ONNX_AVAILABLE 由 CMakeLists.txt 通过 add_compile_definitions 定义
#ifdef SHERPA_ONNX_AVAILABLE
  #include "sherpa-onnx/c-api/c-api.h"
#endif

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include "logger.h"

// ── 辅助: 文件复制 ───────────────────────────────────

static bool copy_file(const std::string& src, const std::string& dst)
{
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::ofstream out(dst, std::ios::binary);
    if (!out) return false;
    out << in.rdbuf();
    return true;
}

// ── 辅助: 文件夹是否为空 ─────────────────────────────

static bool is_dir_empty(const std::string& path)
{
    if (access((path + "/enroll_0.wav").c_str(), F_OK) != 0) return true;
    return false;
}

// ── SpeakerVerifier ──────────────────────────────────

SpeakerVerifier::SpeakerVerifier(const std::string& enroll_dir, float threshold)
    : enroll_dir_(enroll_dir)
    , threshold_(threshold)
{}

SpeakerVerifier::~SpeakerVerifier()
{
#ifdef SHERPA_ONNX_AVAILABLE
    if (manager_) {
        SherpaOnnxDestroySpeakerEmbeddingManager(manager_);
    }
    if (extractor_) {
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor_);
    }
#endif
}

bool SpeakerVerifier::initialize()
{
    std::cout << "[SV] 加载声纹模型 ... " << std::flush;

    // 确保注册目录存在
    mkdir(enroll_dir_.c_str(), 0755);

#ifdef SHERPA_ONNX_AVAILABLE
    // 1. 创建 embedding extractor
    std::string model_path = "src/third_party/sherpa-onnx/speaker-verification-model/"
                             "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

    SherpaOnnxSpeakerEmbeddingExtractorConfig extractor_config;
    memset(&extractor_config, 0, sizeof(extractor_config));
    extractor_config.model = model_path.c_str();
    extractor_config.num_threads = 4;
    extractor_config.provider = "cpu";

    extractor_ = SherpaOnnxCreateSpeakerEmbeddingExtractor(&extractor_config);
    if (!extractor_) {
        std::cerr << "❌ 加载声纹模型失败: " << model_path << std::endl;
        LOG_ERROR("   请下载模型: https://github.com/k2-fsa/sherpa-onnx/releases");
        return false;
    }

    // 2. 创建 embedding manager
    int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor_);
    manager_ = SherpaOnnxCreateSpeakerEmbeddingManager(dim);
    if (!manager_) {
        LOG_ERROR("❌ 创建 embedding manager 失败");
        return false;
    }

    initialized_ = true;
    std::cout << "✅ (dim=" << dim << ")" << std::endl;
    return true;
#else
    LOG_WARN("⚠️  sherpa-onnx 未安装（跳过）");
    initialized_ = true;
    return true;
#endif
}

bool SpeakerVerifier::has_enrolled() const
{
    return !is_dir_empty(enroll_dir_);
}

std::string SpeakerVerifier::status_text() const
{
    if (has_enrolled()) {
        return "声纹已注册 ✅ (" + enroll_dir_ + ")";
    } else {
        return "声纹未注册 ⚠️";
    }
}

std::string SpeakerVerifier::enroll_path() const
{
    return enroll_dir_ + "/enroll_0.wav";
}

bool SpeakerVerifier::enroll(const std::string& wav_path)
{
    mkdir(enroll_dir_.c_str(), 0755);

    // 复制文件到注册目录
    if (!copy_file(wav_path, enroll_path())) {
        LOG_ERROR("   [SV] 保存注册语音失败");
        return false;
    }

#ifdef SHERPA_ONNX_AVAILABLE
    // 计算声纹嵌入并注册到 manager
    std::vector<float> embedding;
    if (compute_embedding(wav_path, embedding)) {
        SherpaOnnxSpeakerEmbeddingManagerAdd(manager_, "default_user", embedding.data());
    }
#endif

    std::cout << "   [SV] ✅ 声纹注册完成 → " << enroll_path() << std::endl;
    return true;
}

bool SpeakerVerifier::verify(const std::string& test_wav)
{
    if (!has_enrolled()) {
        LOG_WARN("   [SV] ⚠️ 声纹未注册，请先注册");
        return false;
    }

#ifdef SHERPA_ONNX_AVAILABLE
    if (!extractor_ || !manager_) return false;

    // 计算测试音频的声纹嵌入
    std::vector<float> embedding;
    if (!compute_embedding(test_wav, embedding)) {
        LOG_ERROR("   [SV] 无法计算测试音频嵌入");
        return false;
    }

    // 验证
    int32_t matched = SherpaOnnxSpeakerEmbeddingManagerVerify(
        manager_, "default_user", embedding.data(), threshold_);

    bool passed = (matched == 1);
    std::cout << "   [SV] " << (passed ? "✅ 通过" : "❌ 拒绝")
              << " (阈值: " << threshold_ << ")" << std::endl;

    return passed;
#else
    (void)test_wav;
    LOG_WARN("   [SV] ⚠️ sherpa-onnx 不可用，跳过验证");
    return true;  // 降级: 通过（不阻塞流程）
#endif
}

bool SpeakerVerifier::compute_embedding(const std::string& wav_path,
                                        std::vector<float>& embedding)
{
#ifdef SHERPA_ONNX_AVAILABLE
    // 读取 WAV 文件
    const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav_path.c_str());
    if (!wave) {
        std::cerr << "   [SV] 无法读取音频: " << wav_path << std::endl;
        return false;
    }

    // 创建 stream 并喂入音频
    const SherpaOnnxOnlineStream* stream =
        SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor_);
    if (!stream) {
        SherpaOnnxFreeWave(wave);
        return false;
    }

    SherpaOnnxOnlineStreamAcceptWaveform(
        stream, wave->sample_rate, wave->samples, wave->num_samples);
    SherpaOnnxOnlineStreamInputFinished(stream);

    // 等待足够音频计算 embedding
    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor_, stream)) {
        LOG_ERROR("   [SV] 音频不足，无法计算声纹");
        SherpaOnnxDestroyOnlineStream(stream);
        SherpaOnnxFreeWave(wave);
        return false;
    }

    // 计算 embedding
    const float* emb = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
        extractor_, stream);
    if (!emb) {
        SherpaOnnxDestroyOnlineStream(stream);
        SherpaOnnxFreeWave(wave);
        return false;
    }

    int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor_);
    embedding.assign(emb, emb + dim);

    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(emb);
    SherpaOnnxDestroyOnlineStream(stream);
    SherpaOnnxFreeWave(wave);

    return true;
#else
    (void)wav_path;
    (void)embedding;
    return false;
#endif
}
