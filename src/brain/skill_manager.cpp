#include "logger.h"
/**
 * 技能管理器 — 实现
 *
 * 注册所有可用技能。新增技能时：
 *   1. 在 brain/skills/ 下创建 skill_xxx.h + skill_xxx.cpp
 *   2. 在这里 #include 并 add_skill 注册
 *   3. 在 CMakeLists.txt 中添加源文件
 */

#include "skill_manager.h"

// ── 所有技能头文件 ────────────────────────────────────
#include "skills/skill_weather.h"
#include "skills/skill_time.h"
#include "skills/skill_search.h"
#include "skills/skill_rag.h"
#include "skills/skill_reminder.h"
#include "skills/skill_system.h"
#include "skills/skill_notes.h"
#include "skills/skill_calculator.h"
#include "skills/skill_entertainment.h"
#include "skills/skill_riddle.h"
#include "skills/skill_fortune.h"
#include "skills/skill_poetry.h"
#include "skills/skill_games.h"
#include "function_caller.h"
#include "embedding_engine.h"

#include <sstream>
#include <iostream>
#include <ctime>
#include <cstdio>

// ── 构造函数：注册所有技能 ────────────────────────────

SkillManager::SkillManager()
{
    add_skill(std::make_unique<WeatherSkill>());
    add_skill(std::make_unique<TimeSkill>());
    add_skill(std::make_unique<WebSearchSkill>());
    add_skill(std::make_unique<ReminderSkill>());
    add_skill(std::make_unique<SystemSkill>());
    add_skill(std::make_unique<NotesSkill>());
    add_skill(std::make_unique<CalculatorSkill>());
    add_skill(std::make_unique<EntertainmentSkill>());
    add_skill(std::make_unique<RiddleSkill>());
    add_skill(std::make_unique<FortuneSkill>());
    add_skill(std::make_unique<PoetrySkill>());
    add_skill(std::make_unique<GamesSkill>());
}

void SkillManager::add_skill(std::unique_ptr<Skill> skill)
{
    skills_.push_back(std::move(skill));
}

void SkillManager::register_rag(std::shared_ptr<EmbeddingEngine> embed,
                                const std::string& docs_dir)
{
    auto rag = std::make_unique<RAGSkill>(embed, docs_dir);
    // 默认启用状态由 config 决定，这里先添加，外部再 set_enabled
    add_skill(std::move(rag));
}

void SkillManager::set_enabled(const std::string& name, bool enabled)
{
    for (auto& s : skills_) {
        if (s->name() == name) {
            s->set_enabled(enabled);
            return;
        }
    }
}

// ── 辅助：收集函数定义 + 按名查找 ─────────────────────

std::vector<FunctionDef> SkillManager::collect_function_defs() const
{
    std::vector<FunctionDef> defs;
    for (auto& s : skills_) {
        if (!s->enabled()) continue;
        FunctionDef def = s->get_function_def();
        if (!def.name.empty()) {
            defs.push_back(std::move(def));
        }
    }
    return defs;
}

std::string SkillManager::execute_tool(const std::string& name,
                                         const nlohmann::json& args,
                                         const std::string& user_text)
{
    Skill* skill = find_skill(name);
    if (!skill || !skill->enabled()) {
        return "";
    }

    // 用结构化参数执行，技能可以选择性地使用 args
    return skill->execute(user_text, args);
}

Skill* SkillManager::find_skill(const std::string& name)
{
    for (auto& s : skills_) {
        if (s->name() == name) return s.get();
    }
    return nullptr;
}

// ── 核心：混合调度 ────────────────────────────────────

SkillResult SkillManager::detect_and_execute(const std::string& user_text)
{
    // ── 策略 1: Function Calling (LLM 驱动) ──────────
    if (function_caller_ && fc_enabled_) {
        auto func_defs = collect_function_defs();
        if (!func_defs.empty()) {
            ToolDecision td = function_caller_->decide(user_text, func_defs);

            if (td.use_tool) {
                Skill* skill = find_skill(td.tool_name);
                if (skill && skill->enabled()) {
                    std::string result = skill->execute(user_text);
                    if (!result.empty()) {
                        std::cout << "   [Skill-FC] \"" << user_text
                                  << "\" → LLM 选择了 " << td.tool_name << std::endl;
                        return {true, skill->is_direct_response(), td.tool_name, result};
                    }
                } else {
                    std::cout << "   [Skill-FC] LLM 选了未知工具: "
                              << td.tool_name << "，降级到关键字匹配" << std::endl;
                }
            }
        }
    }

    // ── 策略 2: Keyword Match (降级方案) ──────────────
    for (auto& s : skills_) {
        if (!s->enabled()) continue;

        if (s->match(user_text)) {
            std::string result = s->execute(user_text);
            if (!result.empty()) {
                std::cout << "   [Skill-KW] \"" << user_text
                          << "\" → 关键字匹配 " << s->name() << std::endl;
                return {true, s->is_direct_response(), s->name(), result};
            }
        }
    }

    return {false, "", ""};
}

std::string SkillManager::get_system_context()
{
    std::time_t now = std::time(nullptr);
    std::tm* local = std::localtime(&now);

    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "当前时间: %d-%02d-%02d %02d:%02d",
        local->tm_year + 1900,
        local->tm_mon + 1,
        local->tm_mday,
        local->tm_hour,
        local->tm_min);

    return std::string(buf);
}
