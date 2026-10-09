#pragma once
/**
 * 音频 I/O
 *
 * Python 对应: src/audio_io.py → AudioRecorder / AudioPlayer
 * 依赖:      系统命令 arecord / aplay (ALSA)
 *
 * 机器人集成时替换为硬件 SDK 的麦克风/扬声器接口。
 */

#include <string>
#include <sys/types.h>

class AudioRecorder {
public:
    /// @param sample_rate  采样率 (Hz)，sensevoice 需要 16000
    /// @param device       arecord 采集设备，如 "plughw:1,0"（空 = ALSA 默认）
    explicit AudioRecorder(int sample_rate = 16000, const std::string& device = "");

    /// 交互式录音：提示按 Enter → 录音 → 再按 Enter 停止 → 保存 WAV
    /// @param output_path  输出 WAV 文件路径
    /// @return true = 成功
    bool record(const std::string& output_path = "temp_recording.wav");

    /// 获取 WAV 文件时长 (秒)
    static float get_duration(const std::string& wav_path);

private:
    int sample_rate_;
    std::string device_;
};


class AudioPlayer {
public:
    /// 播放 WAV 文件 (阻塞到播放结束，不可中断)
    static bool play(const std::string& file_path);

    /// 异步播放 WAV 文件，返回子进程 PID
    /// 持续监听模式下使用，可通过 stop_async() 打断
    static pid_t play_async(const std::string& file_path);

    /// 停止异步播放
    static void stop_async(pid_t pid);
};
