#pragma once
/**
 * 唱歌技能 — SSML 生成歌曲形式的语音
 *
 * 通过 edge_tts SSML <prosody pitch> 模拟简单旋律，
 * 支持多首内置歌曲，关键词触发。
 */

#include "skill_base.h"

class SingSkill : public Skill {
public:
    SingSkill() : Skill("sing") {}

    bool match(const std::string& text) override;
    std::string execute(const std::string& text) override;
    std::string execute(const std::string& text,
                        const nlohmann::json& args) override;
    bool is_direct_response() const override { return true; }

    std::string describe() const override {
        return "我可以唱歌给你听。";
    }

    FunctionDef get_function_def() const override {
        FunctionDef def;
        def.name = "sing";
        def.description = "唱一首歌";
        def.parameters = nlohmann::json::parse(R"({
            "type": "object",
            "properties": {}
        })");
        return def;
    }

private:
    struct Song {
        std::string title;       // 歌名
        std::string hint;        // 唱完后的提示语
        // 每句: {歌词, 音高偏移百分比, 语速}
        struct Line {
            std::string text;
            int pitch_pct;       // 音高偏移 -30 ~ +50
            int break_ms;        // 句尾停顿 ms
        };
        std::vector<Line> lines;
    };

    // 预加载歌曲库
    static std::vector<Song> build_songs();

    // 生成 SSML
    std::string to_ssml(const Song& song) const;
};
