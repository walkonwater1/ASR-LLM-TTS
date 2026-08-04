#include "skill_sing.h"
#include "skill_utils.h"
#include <random>
#include <sstream>

bool SingSkill::match(const std::string& text)
{
    return contains_any(text, {
        "唱歌", "唱首歌", "唱一首", "来一首", "唱支歌",
        "唱一个", "来首歌", "给我唱", "唱首歌吧"
    });
}

std::string SingSkill::execute(const std::string& /*text*/)
{
    auto songs = build_songs();
    if (songs.empty()) return "抱歉，歌本丢了😅";

    // 随机选一首
    static std::mt19937 rng(std::random_device{}());
    int idx = std::uniform_int_distribution<int>(0, songs.size() - 1)(rng);
    auto& song = songs[idx];

    return to_ssml(song);
}

std::string SingSkill::execute(const std::string& text, const nlohmann::json& /*args*/)
{
    return execute(text);
}

// ── SSML 生成 ──────────────────────────────────────────

std::string SingSkill::to_ssml(const Song& song) const
{
    std::ostringstream ss;
    ss << "<speak version=\"1.0\" "
       << "xmlns=\"http://www.w3.org/2001/10/synthesis\" "
       << "xml:lang=\"zh-CN\">\n";

    // 歌曲标题 — 高音 + 慢速
    ss << "  <prosody pitch=\"+40%\" rate=\"x-slow\">"
       << song.title << "</prosody>\n";
    ss << "  <break time=\"600ms\"/>\n";

    // 逐句生成
    for (size_t i = 0; i < song.lines.size(); ++i) {
        auto& line = song.lines[i];
        int pitch = line.pitch_pct;

        ss << "  <prosody pitch=\"";
        if (pitch >= 0) ss << "+";
        ss << pitch << "%\" rate=\"slow\">"
           << line.text << "</prosody>\n";

        if (line.break_ms > 0) {
            ss << "  <break time=\"" << line.break_ms << "ms\"/>\n";
        }
    }

    // 结尾语气
    ss << "  <break time=\"400ms\"/>\n";
    ss << "  <prosody pitch=\"+10%\" rate=\"medium\">"
       << song.hint << "</prosody>\n";
    ss << "</speak>";

    return ss.str();
}

// ── 歌曲库 ─────────────────────────────────────────────

std::vector<SingSkill::Song> SingSkill::build_songs()
{
    using L = Song::Line;
    return {
        // ── 1. 小希之歌 ────────────────────────────
        {
            "小希之歌",
            "谢谢收听～还想听什么歌呢？",
            {
                L{"我是小希你的好朋友",       +30, 400},
                L{"每天陪在你身边不会走",       +25, 500},
                L{"问你天气问时间问所有",       +35, 400},
                L{"唱歌讲故事样样拿手",         +20, 600},
                L{"我是小希你最好的朋友",       +30, 400},
                L{"不管晴天还是雨后",           +25, 500},
                L{"只要你轻轻说一声",           +35, 400},
                L{"我就快快来到你左右",         +20, 800},
            }
        },
        // ── 2. 小星星 ──────────────────────────────
        {
            "小星星",
            "一闪一闪亮晶晶～",
            {
                L{"一闪一闪亮晶晶",             +40, 500},
                L{"满天都是小星星",             +35, 500},
                L{"挂在天上放光明",             +45, 500},
                L{"好像许多小眼睛",             +35, 600},
                L{"一闪一闪亮晶晶",             +40, 500},
                L{"满天都是小星星",             +35, 300},
            }
        },
        // ── 3. 生日快乐 ─────────────────────────────
        {
            "生日快乐歌",
            "祝你生日快乐呀～",
            {
                L{"祝你生日快乐",               +30, 400},
                L{"祝你生日快乐",               +35, 400},
                L{"祝你幸福祝你健康",           +40, 400},
                L{"祝你生日快乐",               +30, 600},
                L{"祝你生日快乐",               +30, 400},
                L{"祝你生日快乐",               +35, 400},
                L{"有个温暖家庭",               +40, 400},
                L{"祝你生日快乐",               +25, 300},
            }
        },
        // ── 4. 春天在哪里 ──────────────────────────
        {
            "春天在哪里",
            "春天就在你的眼睛里～",
            {
                L{"春天在哪里呀春天在哪里",     +35, 400},
                L{"春天在那青翠的山林里",       +30, 500},
                L{"这里有红花呀这里有绿草",     +40, 400},
                L{"还有那会唱歌的小黄鹂",       +30, 600},
                L{"嘀哩嘀哩嘀哩嘀哩",           +45, 300},
                L{"嘀哩嘀哩嘀哩嘀哩",           +40, 400},
                L{"春天在青翠的山林里",         +30, 400},
                L{"还有那会唱歌的小黄鹂",       +25, 300},
            }
        },
        // ── 5. 新年好 ──────────────────────────────
        {
            "新年好",
            "新年快乐，万事如意～",
            {
                L{"新年好呀新年好呀",           +35, 400},
                L{"祝贺大家新年好",             +30, 500},
                L{"我们唱歌我们跳舞",           +40, 400},
                L{"祝贺大家新年好",             +30, 600},
                L{"新年好呀新年好呀",           +35, 400},
                L{"祝贺大家新年好",             +30, 500},
                L{"我们唱歌我们跳舞",           +40, 400},
                L{"祝贺大家新年好",             +25, 300},
            }
        },
    };
}
