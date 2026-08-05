/**
 * 唤醒词检测 — 拼音匹配
 *
 * Python 对应: src/kws.py → WakeWordDetector
 * C++ 实现:   纯 C++，静态汉字→拼音查找表 → 子串匹配
 *
 * 不需要 ML 模型。
 * 算法:
 *   1. 提取 ASR 文本中的汉字
 *   2. 查表转为拼音
 *   3. 检查唤醒词拼音是否为子串
 */

#include "wake_word.h"
#include "pinyin_table.h"

#include <iostream>
#include <algorithm>
#include "logger.h"

// ════════════════════════════════════════════════════════════════
// 汉字 → 拼音查找表（自动生成，覆盖 26703 个汉字）
// 来源: src/speech/pinyin_table.h（pypinyin 生成）
// ════════════════════════════════════════════════════════════════

const std::unordered_map<std::string, std::string>& WakeWordDetector::pinyin_table()
{
    return voice::full_pinyin_table();
}


// ════════════════════════════════════════════════════════════════
// WakeWordDetector
// ════════════════════════════════════════════════════════════════

WakeWordDetector::WakeWordDetector(const std::string& wake_word)
    : wake_word_(wake_word)
{
    if (!wake_word_.empty()) {
        wake_words_.push_back("");  // placeholder for detect()
        words_pinyin_.push_back(wake_word_);
    }
}

WakeWordDetector::WakeWordDetector(const std::vector<std::string>& wake_words)
    : wake_words_(wake_words)
{
    init_pinyin_list();
}

void WakeWordDetector::init_pinyin_list()
{
    words_pinyin_.clear();
    for (auto& w : wake_words_) {
        words_pinyin_.push_back(text_to_pinyin(w));
    }
    // 按拼音长度从长到短排序（防止短词误匹配长词的一部分）
    // 保持 words_pinyin_ 和 wake_words_ 同步排序
    std::vector<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < wake_words_.size(); ++i) {
        pairs.emplace_back(wake_words_[i], words_pinyin_[i]);
    }
    std::sort(pairs.begin(), pairs.end(),
        [](auto& a, auto& b) { return a.second.size() > b.second.size(); });
    wake_words_.clear();
    words_pinyin_.clear();
    for (auto& p : pairs) {
        wake_words_.push_back(p.first);
        words_pinyin_.push_back(p.second);
    }
}

bool WakeWordDetector::detect(const std::string& asr_text)
{
    if (!enabled()) {
        return true;  // 未启用唤醒词，直接通过
    }

    // 多唤醒词模式
    if (wake_word_.empty() && !words_pinyin_.empty()) {
        return !detect_any(asr_text).empty();
    }

    // 单选模式
    std::string pinyin_text = text_to_pinyin(asr_text);
    bool matched = pinyin_text.find(wake_word_) != std::string::npos;

    if (!matched) {
        std::cout << "   [KWS] 未检测到唤醒词 \"" << wake_word_
                  << "\" (识别: " << pinyin_text << ")" << std::endl;
    } else {
        LOG_INFO("   [KWS] ✅ 唤醒词检测成功");
    }

    return matched;
}

std::string WakeWordDetector::detect_any(const std::string& asr_text)
{
    if (!enabled()) return "";

    std::string pinyin_text = text_to_pinyin(asr_text);

    // 按拼音长度从长到短匹配（已在 init_pinyin_list 中排序）
    for (size_t i = 0; i < words_pinyin_.size(); ++i) {
        if (pinyin_text.find(words_pinyin_[i]) != std::string::npos) {
            LOG_INFO("   [KWS] ✅ 唤醒词: \"{}\"", wake_words_[i]);
            return wake_words_[i];
        }
    }

    std::cout << "   [KWS] 未检测到唤醒词 (识别: " << pinyin_text << ")" << std::endl;
    return "";
}

/// 返回 UTF-8 字符的字节长度（1-4），无效前导字节返回 1
static int utf8_char_len(unsigned char first_byte)
{
    if ((first_byte & 0x80) == 0) return 1;       // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;    // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;    // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;    // 11110xxx
    return 1;  // 非法前导字节，逐字节跳过
}

std::string WakeWordDetector::text_to_pinyin(const std::string& text)
{
    std::string result;

    const auto& table = pinyin_table();
    for (size_t i = 0; i < text.size(); ) {
        int len = utf8_char_len(static_cast<unsigned char>(text[i]));
        if (i + len > text.size()) break;  // 截断的 UTF-8，安全退出

        std::string ch = text.substr(i, len);
        auto it = table.find(ch);
        if (it != table.end()) {
            if (!result.empty()) result += " ";
            result += it->second;
        }
        // 不在表中的字符（ASCII/标点/罕见字）静默跳过，保持拼音串干净
        i += len;
    }

    return result;
}
