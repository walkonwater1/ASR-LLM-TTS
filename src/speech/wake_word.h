#pragma once
/**
 * 唤醒词检测 (KWS) — 拼音匹配
 *
 * 实现方式:
 *   1. 汉字 → 拼音查表（静态 unordered_map）
 *   2. 子串匹配唤醒词拼音
 *
 * 支持单/多唤醒词。
 */

#include <string>
#include <vector>
#include <unordered_map>

class WakeWordDetector {
public:
    /// @param wake_word 单个唤醒词拼音，如 "zhan qi lai"，空字符串 = 关闭
    explicit WakeWordDetector(const std::string& wake_word = "");

    /// @param wake_words 多个唤醒词原文（中文），如 {"你好小希", "小希小希"}
    explicit WakeWordDetector(const std::vector<std::string>& wake_words);

    bool enabled() const { return !words_pinyin_.empty(); }

    /// 检查 ASR 文本是否包含唤醒词（单选模式）
    bool detect(const std::string& asr_text);

    /// 检查 ASR 文本匹配了哪个唤醒词
    /// @return 匹配到的唤醒词原文，未匹配返回空字符串
    std::string detect_any(const std::string& asr_text);

private:
    std::vector<std::string>         wake_words_;     // 唤醒词原文
    std::vector<std::string>         words_pinyin_;   // 唤醒词拼音（与 wake_words_ 对应）
    std::string                      wake_word_;      // 旧单选兼容

    /// 中文文本 → 拼音字符串（如 "你好" → "ni hao"）
    static std::string text_to_pinyin(const std::string& text);

    /// 单个汉字 → 拼音（查表）
    static const std::unordered_map<std::string, std::string>& pinyin_table();

    /// 初始化拼音列表
    void init_pinyin_list();
};
