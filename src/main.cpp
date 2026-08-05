/**
 * 语音助手 — 主函数
 *
 * 用法:
 *   ./build/voice_pipeline
 *
 * 启动后直接进入持续语音监听模式：
 *   - 默认 SLEEP 状态，等待唤醒词（"你好爱秋"/"你好小希"/"小希小希"）
 *   - 唤醒后进入 ACTIVE 状态，正常对话
 *   - 90 秒无交互自动休眠（"我去休息啦"）
 *   - Ctrl+C 退出
 *
 * 编译: mkdir build && cd build && cmake .. && make
 */

#include "voice_pipeline.h"
#include "utils/config_watcher.h"
#include "logger.h"

#include <iostream>
#include <string>
#include <csignal>
#include <memory>

// 全局指针，用于 Ctrl+C 信号处理
static VoicePipeline* g_pipeline = nullptr;
static std::string    g_cli_config_path;

static void signal_handler(int /*sig*/)
{
    std::cout << std::endl;
    if (g_pipeline) {
        g_pipeline->stop_interactive();
    }
}

/// SIGHUP: 热重载配置 (Layer 4.4)
static void reload_handler(int /*sig*/)
{
    LOG_INFO("[SIGHUP] 📝 收到 SIGHUP，重新加载配置...");
    if (g_cli_config_path.empty() || !g_pipeline) {
        LOG_WARN("[SIGHUP] 无配置文件路径或 pipeline 未就绪");
        return;
    }
    PipelineConfig new_cfg;
    if (!new_cfg.load_from_file(g_cli_config_path)) {
        LOG_ERROR("[SIGHUP] ❌ 配置文件加载失败: {}", g_cli_config_path);
        return;
    }
    g_pipeline->reload_config(new_cfg);
}

int main(int /*argc*/, char** /*argv*/)
{
    // ── 日志系统 ────────────────────────────────────────
    voice::init_logger("voice_pipeline.log");

    // ── 配置 ──────────────────────────────────────────
    PipelineConfig cfg;

    // 1) 尝试自动加载 config.json（当前目录 → 环境变量）
    std::string config_path = PipelineConfig::auto_load_path();
    if (!config_path.empty() && cfg.load_from_file(config_path)) {
        std::cout << "📄 已加载配置: " << config_path << std::endl;
    } else {
        std::cout << "ℹ️  未找到 config.json，使用默认配置" << std::endl;
    }

    // ── 初始化 ────────────────────────────────────────
    VoicePipeline pipeline(cfg);
    if (!pipeline.initialize()) {
        std::cerr << "初始化失败，退出。" << std::endl;
        return 1;
    }

    pipeline.set_config_path(config_path);
    g_cli_config_path = config_path;
    g_pipeline = &pipeline;

    // Ctrl+C 处理
    signal(SIGINT, signal_handler);
    signal(SIGHUP, reload_handler);  // Layer 4.4: 热重载

    // ── 启动配置文件监听 (Layer 4.4) ──────────────────
    std::unique_ptr<ConfigWatcher> watcher;
    if (!config_path.empty()) {
        watcher = std::make_unique<ConfigWatcher>(config_path, [&pipeline, &config_path]() {
            PipelineConfig new_cfg;
            if (new_cfg.load_from_file(config_path)) {
                pipeline.reload_config(new_cfg);
            } else {
                LOG_ERROR("[ConfigWatcher] ❌ 配置文件加载失败: {}", config_path);
            }
        });
        watcher->start();
    }

    // ── 直接进入语音交互模式 ──────────────────────────
    pipeline.run_interactive();

    if (watcher) watcher->stop();
    g_pipeline = nullptr;
    return 0;
}
