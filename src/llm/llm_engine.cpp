#include "logger.h"
/**
 * 大语言模型推理 — Ollama HTTP API
 *
 * Python 对应: src/llm.py → LLMEngine (ollama.chat)
 * C++ 实现:   libcurl + nlohmann/json → POST /api/chat
 *
 * Ollama 是本地服务(localhost:11434)，底层是 llama.cpp。
 * 如果要完全嵌入（无 Ollama 进程），替换为 llama.h C API。
 */

#include "llm_engine.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

using json = nlohmann::json;

// ── UTF-8 清洗 ────────────────────────────────────────

/// 移除字符串中的非法 UTF-8 字节序列，替换为 U+FFFD
static std::string sanitize_utf8(const std::string& input)
{
    std::string result;
    result.reserve(input.size());

    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = static_cast<unsigned char>(input[i]);

        // ASCII (0x00-0x7F)
        if (c <= 0x7F) {
            result.push_back(c);
            ++i;
        }
        // 2-byte sequence (0xC0-0xDF)
        else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < input.size() &&
                (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(input[i+1]);
                i += 2;
            } else {
                result += "\xEF\xBF\xBD"; // U+FFFD
                ++i;
            }
        }
        // 3-byte sequence (0xE0-0xEF)
        else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < input.size() &&
                (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(input[i+2]) & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(input[i+1]);
                result.push_back(input[i+2]);
                i += 3;
            } else {
                result += "\xEF\xBF\xBD"; // U+FFFD
                ++i;
            }
        }
        // 4-byte sequence (0xF0-0xF7)
        else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < input.size() &&
                (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(input[i+2]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(input[i+3]) & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(input[i+1]);
                result.push_back(input[i+2]);
                result.push_back(input[i+3]);
                i += 4;
            } else {
                result += "\xEF\xBF\xBD"; // U+FFFD
                ++i;
            }
        }
        // Invalid lead byte
        else {
            result += "\xEF\xBF\xBD"; // U+FFFD
            ++i;
        }
    }
    return result;
}

// ── libcurl 回调：将响应写入 string ──────────────────

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* str = static_cast<std::string*>(userdata);
    str->append(ptr, size * nmemb);
    return size * nmemb;
}

// ── LLMEngine ────────────────────────────────────────

LLMEngine::LLMEngine(const std::string& host,
                     const std::string& model,
                     const std::string& system_prompt)
    : host_(host)
    , model_(model)
{
    builder_.set_system(system_prompt);
}

std::string LLMEngine::chat(const std::string& user_message,
                            const std::string& history_context,
                            const std::string& extra_context)
{
    auto t0 = std::chrono::steady_clock::now();

    // 注入动态变量（每次对话前更新）
    {
        std::time_t now = std::time(nullptr);
        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M", std::localtime(&now));
        builder_.set("time", time_buf);
    }

    // 使用 PromptBuilder 组装 messages
    json messages = builder_.build_messages(user_message, history_context, extra_context);

    // 构造 JSON 请求体
    json body;
    body["model"] = model_;
    body["stream"] = false;
    body["messages"] = messages;

    std::string request_json = body.dump();

    // 发送 HTTP POST
    std::string reply;
    try {
        std::string response = http_post(request_json);
        response = sanitize_utf8(response);
        json resp_json = json::parse(response);
        reply = resp_json.value("message", json::object()).value("content", "");
    } catch (const std::exception& e) {
        reply = std::string("[Ollama 错误: ") + e.what() + "]";
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "   [LLM] " << elapsed << "ms"
              << " (" << reply.size() << " chars)" << std::endl;

    return reply;
}

std::string LLMEngine::http_post(const std::string& request_json)
{
    std::string url = host_ + "/api/chat";
    std::string response;

    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("curl_easy_init 失败");
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)request_json.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);  // LLM 推理可能较慢

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl 错误: ") + curl_easy_strerror(res));
    }

    return response;
}
