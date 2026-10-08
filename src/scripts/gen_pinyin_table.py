#!/usr/bin/env python3
"""把 pinyin_table.h 转成"POD 数组 + 惰性构建 map"的形式。

背景
----
原文件把 26703 条汉字→拼音直接写成 std::unordered_map 的花括号初始化列表。
GCC 处理超大型初始化列表极慢：在 x86 上单个 TU 的 -O2 编译要 16.5 秒、峰值
1.26GB 内存；到了 4 核 ARM 板子上，同一个文件实测超过 20 分钟，而且
make -j4 时多个此类 TU 并发会撑爆 3.8GB 内存。

数据一个字都没改，只改了存放形式：POD 数组（编译期几乎零成本）+ 首次调用时
构建 map（26k 次带 reserve 的 emplace，运行期几毫秒）。

用法
----
    python3 src/scripts/gen_pinyin_table.py            # 就地转换
    python3 src/scripts/gen_pinyin_table.py --check    # 只校验，不写文件

若手上是 pypinyin 直接生成的新数据，改法是同样的：让 pypinyin 输出
    {"字","yin"}, ...
形式的文本，替换本脚本 parse_pairs() 的输入即可。
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "speech" / "pinyin_table.h"

PAIR_RE = re.compile(r'\{\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\}')
PAIRS_PER_LINE = 20

PREAMBLE = '''// Auto-generated — 汉字 → 拼音表，覆盖 CJK 统一汉字 (U+4E00-U+9FFF) + 扩展A (U+3400-U+4DBF)
//
// 由 src/scripts/gen_pinyin_table.py 生成，请勿手工编辑。
//
// 数据没变，存放形式换成了"POD 数组 + 首次调用时构建 map"：
// 直接写成 26703 项的 std::unordered_map 初始化列表会让 GCC 编译期爆炸
// （x86 -O2 单文件 16.5s / 1.26GB；4 核 ARM 上超过 20 分钟，且 make -j4
// 时并发多个此类 TU 会 OOM）。POD 聚合初始化编译期几乎免费。

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

namespace voice {

inline const std::pair<const char*, const char*> kPinyinRaw[] = {
'''

ACCESSOR = '''};

// 首次调用时构建（C++11 magic static，线程安全），之后只读复用
inline const std::unordered_map<std::string, std::string>& full_pinyin_table() {
    static const std::unordered_map<std::string, std::string> table = [] {
        std::unordered_map<std::string, std::string> m;
        m.reserve(sizeof(kPinyinRaw) / sizeof(kPinyinRaw[0]));
        for (const auto& kv : kPinyinRaw) {
            m.emplace(kv.first, kv.second);
        }
        return m;
    }();
    return table;
}

}  // namespace voice
'''


def parse_pairs(text):
    return PAIR_RE.findall(text)


def render(pairs):
    lines = []
    for i in range(0, len(pairs), PAIRS_PER_LINE):
        chunk = pairs[i:i + PAIRS_PER_LINE]
        lines.append("    " + " ".join('{"%s", "%s"},' % (k, v) for k, v in chunk))
    body = "\n".join(lines)
    # 末行不该留尾逗号以外的怪异形态，保持统一即可
    return PREAMBLE + body + "\n" + ACCESSOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    if not HEADER.exists():
        sys.exit("找不到 %s" % HEADER)

    text = HEADER.read_text(encoding="utf-8")
    pairs = parse_pairs(text)
    if not pairs:
        sys.exit("没有解析出任何 {字, 拼音} 对，输入格式可能已变")

    # 旧实现是 unordered_map 的 initializer_list 构造，重复键首次生效；
    # 新实现用 emplace，同样首次生效 —— 这里确认语义一致、并暴露重复项。
    seen = {}
    dupes = []
    for k, v in pairs:
        if k in seen:
            if seen[k] != v:
                dupes.append((k, seen[k], v))
        else:
            seen[k] = v

    raw = pairs
    print("解析出 %d 条，去重后 %d 个不同汉字" % (len(pairs), len(seen)))
    if dupes:
        print("注意: %d 个汉字有冲突拼音（新旧实现都取首次出现的那个）:" % len(dupes))
        for k, first, later in dupes[:10]:
            print("  %s: 采用 %s（另有 %s）" % (k, first, later))

    out = render(raw)

    # 自检: 重新解析生成结果，逐条比对，确保没在渲染中丢数据或改数据
    back = parse_pairs(out)
    if back != pairs:
        sys.exit("自检失败: 渲染后重新解析的结果与输入不一致")
    print("自检通过: 渲染后重新解析与输入逐条一致")

    if args.check:
        return

    HEADER.write_text(out, encoding="utf-8")
    print("已写入 %s (%.0f KB)" % (HEADER, HEADER.stat().st_size / 1024))


if __name__ == "__main__":
    main()
