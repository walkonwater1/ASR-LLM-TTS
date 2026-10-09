/**
 * 音频 I/O — ALSA (arecord / aplay)
 *
 * Python 对应: src/audio_io.py → AudioRecorder / AudioPlayer
 * C++ 实现:   系统命令 arecord / aplay
 *
 * 机器人集成时替换为硬件 SDK 的麦克风/扬声器接口。
 */

#include "audio_io.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <limits>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include "logger.h"

// ── AudioRecorder ────────────────────────────────────

AudioRecorder::AudioRecorder(int sample_rate, const std::string& device)
    : sample_rate_(sample_rate), device_(device)
{}

bool AudioRecorder::record(const std::string& output_path)
{
    // 交互式录音：按 Enter 开始/结束
    std::cout << "\n🔴 按下 Enter 开始录音..." << std::flush;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Fork: 子进程跑 arecord，父进程等 Enter 后 kill
    pid_t pid = fork();
    if (pid == 0) {
        // 子进程：执行 arecord
        std::string rate_str = std::to_string(sample_rate_);
        if (device_.empty()) {
            execlp("arecord", "arecord",
                   "-f", "S16_LE",
                   "-r", rate_str.c_str(),
                   "-c", "1",
                   output_path.c_str(),
                   nullptr);
        } else {
            execlp("arecord", "arecord",
                   "-D", device_.c_str(),
                   "-f", "S16_LE",
                   "-r", rate_str.c_str(),
                   "-c", "1",
                   output_path.c_str(),
                   nullptr);
        }
        _exit(1);   // execlp 失败
    }

    if (pid < 0) {
        LOG_ERROR("   ❌ fork 失败");
        return false;
    }

    LOG_INFO("   🎙️  录音中... 再次按下 Enter 结束");
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // 停止录音
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // 等文件写完

    // 检查文件
    std::ifstream f(output_path);
    if (!f.good()) {
        std::cerr << "   ❌ 录音文件未生成: " << output_path << std::endl;
        return false;
    }

    float duration = get_duration(output_path);
    std::cout << "   ✅ 录音完成 (" << duration << "s) → " << output_path << std::endl;

    return true;
}

float AudioRecorder::get_duration(const std::string& wav_path)
{
    std::ifstream file(wav_path, std::ios::binary);
    if (!file) return 0.0f;

    // 读 WAV 头，找 data chunk 大小
    file.seekg(4);   // skip "RIFF"
    uint32_t riff_size;
    file.read(reinterpret_cast<char*>(&riff_size), 4);
    file.seekg(12);  // skip "WAVE"
    file.seekg(8, std::ios::cur); // skip "fmt " + fmt_size
    file.seekg(2, std::ios::cur); // audio_format
    uint16_t channels;
    file.read(reinterpret_cast<char*>(&channels), 2);
    uint32_t sample_rate;
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    file.seekg(6, std::ios::cur); // byte_rate + block_align
    uint16_t bits_per_sample;
    file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

    // 找 "data" chunk
    char chunk_id[5] = {};
    uint32_t data_size = 0;
    while (file.read(chunk_id, 4)) {
        file.read(reinterpret_cast<char*>(&data_size), 4);
        if (std::strncmp(chunk_id, "data", 4) == 0) break;
        file.seekg(data_size, std::ios::cur);
    }

    uint32_t total_samples = data_size / (channels * bits_per_sample / 8);
    return (float)total_samples / sample_rate;
}

// ── AudioPlayer ──────────────────────────────────────

bool AudioPlayer::play(const std::string& file_path)
{
    // 用 env -u 清除 conda 的 LD_LIBRARY_PATH，避免 ALSA 库冲突
    std::string cmd = "env -u LD_LIBRARY_PATH aplay -q " + file_path + " 2>/dev/null";
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

pid_t AudioPlayer::play_async(const std::string& file_path)
{
    pid_t pid = fork();
    if (pid == 0) {
        // 清除 conda 环境变量，避免 ALSA 库冲突
        unsetenv("LD_LIBRARY_PATH");

        // 子进程: 重定向 stdin/stdout/stderr 到 /dev/null
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        execlp("aplay", "aplay", "-q", file_path.c_str(), nullptr);
        _exit(1);
    }
    if (pid < 0) {
        LOG_ERROR("   ❌ fork 播放进程失败");
    }
    return pid;
}

void AudioPlayer::stop_async(pid_t pid)
{
    if (pid <= 0) return;

    // 发送 SIGTERM 终止 aplay 进程
    kill(pid, SIGTERM);

    // 非阻塞回收，避免僵尸进程
    int status;
    waitpid(pid, &status, WNOHANG);
}
