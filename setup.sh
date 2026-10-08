#!/usr/bin/env bash
#
# 一键安装 & 运行语音交互管线（x86_64 / aarch64 通用）
#
# 用法:
#   ./setup.sh                          # 安装 client profile + 编译
#   ./setup.sh --run                    # 安装 + 编译 + 运行
#   ./setup.sh --profile minimal        # 仅 core + sherpa（纯本地）
#   ./setup.sh --with piper             # 在 profile 基础上加组件
#   ./setup.sh --without edge           # 在 profile 基础上减组件
#   ./setup.sh --dry-run --profile client   # 只打印将要执行的动作
#   ./setup.sh --models                 # 仅下载运行库与模型（不编译）
#   ./setup.sh --build                  # 仅编译
#   ./setup.sh --clean                  # 清理编译产物
#
# 组件:
#   core       构建工具链 + espeak-ng + ALSA + .venv（总是安装）
#   sherpa     sherpa-onnx 运行时（按架构自动选 x64 / aarch64）+ ASR + 声纹模型
#   edge       edge-tts 云端 TTS（ffmpeg + pip edge-tts）
#   piper      Piper 本地 TTS（pip piper-tts + 音色模型）
#   embedding  RAG 向量模型（pip torch/modelscope + 导出 bge-small-zh ONNX）
#
# profile:
#   minimal    core sherpa
#   client     core sherpa edge        （默认；带麦克风的交互终端）
#   full       core sherpa edge piper embedding   （--all 同义）
#
# 硬件: CPU only, 无需 GPU（x86_64 / Jetson 等 aarch64）
# 系统: Ubuntu 20.04+

set -euo pipefail

# ── 路径 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
THIRD_PARTY_DIR="${SRC_DIR}/third_party"
SHERPA_DIR="${THIRD_PARTY_DIR}/sherpa-onnx"
BUILD_DIR="${SRC_DIR}/build"
VENV_DIR="${SCRIPT_DIR}/.venv"

# ── 颜色 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${BLUE}[~]${NC} $*"; }

# ── 架构检测 ──────────────────────────────────────────────
# 探测命令一律做存在性判断，避免 set -e 下提前退出
RAW_ARCH="$(uname -m)"
case "$RAW_ARCH" in
    x86_64|amd64)  TARGET_ARCH="x86_64"  ;;
    aarch64|arm64) TARGET_ARCH="aarch64" ;;
    *) err "不支持的架构: $RAW_ARCH (仅支持 x86_64 / aarch64)"; exit 1 ;;
esac

# Debian multiarch 目录三元组，用于 /usr/lib/<triplet> 与 /usr/include/<triplet>
MULTIARCH=""
if command -v dpkg-architecture >/dev/null 2>&1; then
    MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
fi
if [ -z "$MULTIARCH" ] && command -v gcc >/dev/null 2>&1; then
    MULTIARCH="$(gcc -print-multiarch 2>/dev/null || true)"
fi
[ -n "$MULTIARCH" ] || MULTIARCH="${TARGET_ARCH}-linux-gnu"

# ── 各上游 release 的架构标识（x64 / aarch64）─────────────
case "$TARGET_ARCH" in
    x86_64)  PKG_ARCH="x64";     SHERPA_VARIANT="shared"     ;;
    aarch64) PKG_ARCH="aarch64"; SHERPA_VARIANT="shared-cpu" ;;
esac

# ── sherpa-onnx 运行时（asset 名带 v 前缀 + 构建变体后缀）───
# 取值已实测存在（HTTP 206）:
#   x64     → sherpa-onnx-v1.13.2-linux-x64-shared.tar.bz2
#   aarch64 → sherpa-onnx-v1.13.2-linux-aarch64-shared-cpu.tar.bz2
SHERPA_VERSION="1.13.2"
SHERPA_ASSET="sherpa-onnx-v${SHERPA_VERSION}-linux-${PKG_ARCH}-${SHERPA_VARIANT}"
SHERPA_TAR="${SHERPA_ASSET}.tar.bz2"
SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/${SHERPA_TAR}"

# ── onnxruntime 头文件（仅头文件；.so 复用 sherpa 包内的那份）──
# src/llm/onnx_embedding.cpp 无条件 include <onnxruntime_c_api.h>，
# 但 sherpa 包只带 lib/libonnxruntime.so，因此头文件必须单独准备。
# ORT_VERSION 必须与 sherpa 内置的 libonnxruntime.so 版本一致（C API ABI）。
ORT_VERSION="1.24.4"
ORT_ASSET="onnxruntime-linux-${PKG_ARCH}-${ORT_VERSION}"
ORT_TAR="${ORT_ASSET}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TAR}"
ORT_DIR="${THIRD_PARTY_DIR}/onnxruntime"

# ── 模型下载地址（HF Mirror 国内更快）─────────────────────
HF_BASE="https://huggingface.co"
HF_MIRROR="https://hf-mirror.com"

# ASR 模型: SenseVoice Small int8 (~228MB)
ASR_MODEL_REPO="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
ASR_MODEL_FILES=(
    "model.int8.onnx"
    "tokens.txt"
)
ASR_MODEL_DIR="${SHERPA_DIR}/sense-voice-model"

# 声纹模型: CAM++ (~27MB)
SV_MODEL_REPO="csukuangfj/sherpa-onnx-speaker-verification"
SV_MODEL_FILES=(
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)
SV_MODEL_DIR="${SHERPA_DIR}/speaker-verification-model"

# Piper 音色（可选，由 piper 自带的 download_voices 拉取）
PIPER_VOICE="zh_CN-huayan-medium"

# ── 组件与 profile ────────────────────────────────────────
ALL_COMPONENTS=(core sherpa edge piper embedding)
declare -A COMPONENTS=()
DRY_RUN=0

component_enabled() {
    [ -n "${COMPONENTS[$1]:-}" ]
}

# 结果写入 PROFILE_LIST（不能用 $(...) 收集：命令替换里 exit 退不出主脚本）
profile_to_components() {
    case "$1" in
        minimal) PROFILE_LIST=(core sherpa) ;;
        client)  PROFILE_LIST=(core sherpa edge) ;;
        full)    PROFILE_LIST=(core sherpa edge piper embedding) ;;
        *) err "未知 profile: $1（可用: minimal / client / full）"; exit 1 ;;
    esac
}

# --dry-run 下只打印不执行
dry() {
    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] $*"
        return 0
    fi
    "$@"
}

# 检查所有需要的文件是否存在
dir_complete() {
    local dir="$1"; shift
    local f
    for f in "$@"; do
        [ -f "${dir}/${f}" ] || return 1
    done
    return 0
}

# ── 步骤1: 检查/安装系统依赖 ─────────────────────────────

install_deps() {
    log "检查系统依赖..."

    local missing=()
    local pkgs=()

    command -v cmake >/dev/null 2>&1 || { missing+=("cmake"); pkgs+=("cmake"); }
    command -v g++ >/dev/null 2>&1   || { missing+=("g++");   pkgs+=("build-essential"); }

    # espeak-ng（直接查库文件，避免 ldconfig 非 root 无权限）
    if [ ! -f "/usr/lib/${MULTIARCH}/libespeak-ng.so.1" ]; then
        missing+=("espeak-ng")
        pkgs+=("espeak-ng" "espeak-ng-data")
    fi

    # libcurl（pkg-config 优先，再查多架构头文件路径）
    if ! pkg-config --exists libcurl 2>/dev/null \
       && [ ! -f /usr/include/curl/curl.h ] \
       && [ ! -f "/usr/include/${MULTIARCH}/curl/curl.h" ]; then
        missing+=("libcurl4")
        pkgs+=("libcurl4-openssl-dev")
    fi

    # nlohmann-json
    if [ ! -f /usr/include/nlohmann/json.hpp ] && [ ! -f /usr/include/nlohmann/json_fwd.hpp ]; then
        missing+=("nlohmann-json3-dev")
        pkgs+=("nlohmann-json3-dev")
    fi

    # Boost::system + spdlog（CMakeLists 里都是 REQUIRED）
    if [ ! -f /usr/include/boost/version.hpp ] \
       || ! ls /usr/lib/"${MULTIARCH}"/libboost_system.so* >/dev/null 2>&1; then
        missing+=("libboost-system-dev")
        pkgs+=("libboost-system-dev")
    fi
    if [ ! -f /usr/include/spdlog/spdlog.h ]; then
        missing+=("libspdlog-dev")
        pkgs+=("libspdlog-dev")
    fi

    # ALSA 工具（amixer 调录音增益，交互终端需要）
    command -v amixer >/dev/null 2>&1 || { missing+=("alsa-utils"); pkgs+=("alsa-utils"); }

    # Python 运行时（.venv 需要 ensurepip）
    if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
        missing+=("python3-venv")
        pkgs+=("python3-venv" "python3-pip")
    fi

    # edge 组件需要 ffmpeg 做 mp3 → 16k wav
    if component_enabled edge; then
        command -v ffmpeg >/dev/null 2>&1 || { missing+=("ffmpeg"); pkgs+=("ffmpeg"); }
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        warn "缺少依赖: ${missing[*]}"
        if [ "$DRY_RUN" = "1" ]; then
            info "[dry-run] sudo apt-get update -qq"
            info "[dry-run] sudo apt-get install -y ${pkgs[*]}"
        else
            info "正在安装..."
            sudo apt-get update -qq
            sudo apt-get install -y "${pkgs[@]}"
        fi
        log "系统依赖已处理"
    else
        log "系统依赖已就绪"
    fi
}

# ── 步骤2: 下载函数（优先 HF 镜像） ────────────────────────

download_file() {
    local url="$1"
    local output="$2"
    local desc="$3"

    if [ -f "$output" ]; then
        log "已存在: ${desc}"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] 下载: ${desc} ← ${url}"
        return 0
    fi

    info "下载: ${desc} ..."
    mkdir -p "$(dirname "$output")"

    # 尝试多个下载源
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$output" "$url" 2>&1 || {
            warn "wget 失败，换镜像重试..."
            local mirror_url="${url/${HF_BASE}/${HF_MIRROR}}"
            wget -q --show-progress -O "$output" "$mirror_url" 2>&1
        }
    elif command -v curl &>/dev/null; then
        curl -L -# -o "$output" "$url" 2>&1 || {
            warn "curl 失败，换镜像重试..."
            local mirror_url="${url/${HF_BASE}/${HF_MIRROR}}"
            curl -L -# -o "$output" "$mirror_url" 2>&1
        }
    else
        err "需要 wget 或 curl，请先安装"
        exit 1
    fi

    if [ -f "$output" ]; then
        log "下载完成: ${desc} ($(du -h "$output" | cut -f1))"
    else
        err "下载失败: ${desc}"
        return 1
    fi
}

# ── 步骤3: 下载 sherpa-onnx 运行时库 ─────────────────────

install_sherpa_onnx() {
    log "检查 sherpa-onnx 运行时库 (${TARGET_ARCH})..."

    local lib_file="${SHERPA_DIR}/lib/libsherpa-onnx-c-api.so"
    local include_file="${SHERPA_DIR}/include/sherpa-onnx/c-api/c-api.h"

    if [ -f "$lib_file" ] && [ -f "$include_file" ]; then
        log "sherpa-onnx 已就绪"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] 下载并解压 sherpa-onnx v${SHERPA_VERSION} (${SHERPA_ASSET})"
        info "[dry-run]   ${SHERPA_URL}"
        info "[dry-run]   → ${SHERPA_DIR}/{lib,include}"
        return 0
    fi

    info "下载 sherpa-onnx v${SHERPA_VERSION} (${SHERPA_ASSET}) ..."

    # 先确认 asset 存在，避免把 404 页面当压缩包存下来
    if command -v curl &>/dev/null && ! curl -sfIL "$SHERPA_URL" >/dev/null 2>&1; then
        err "资源不存在或网络不可达: ${SHERPA_URL}"
        err "请确认 ${TARGET_ARCH} 对应的构建变体（表在 setup.sh 顶部 SHERPA_VARIANT）"
        return 1
    fi

    local tmp_dir="/tmp/sherpa-onnx-$$"
    mkdir -p "$tmp_dir"

    local tar_path="${tmp_dir}/${SHERPA_TAR}"

    # 下载
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$tar_path" "$SHERPA_URL" || true
    else
        curl -L -# -o "$tar_path" "$SHERPA_URL" || true
    fi

    if [ ! -f "$tar_path" ] || [ ! -s "$tar_path" ]; then
        err "下载 sherpa-onnx 失败"
        err "请手动下载: ${SHERPA_URL}"
        err "解压到: ${SHERPA_DIR}"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 解压
    info "解压 sherpa-onnx ..."
    tar -xjf "$tar_path" -C "$tmp_dir"

    # 内层目录名 = asset 名去掉 .tar.bz2，用 glob 定位而非拼接，
    # 这样切换构建变体（-shared / -shared-cpu / -static ...）时不会失效
    local extracted_dir
    extracted_dir="$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d -name 'sherpa-onnx-*' | head -n1)"
    if [ -z "$extracted_dir" ]; then
        err "解压目录未找到（tar 内容与预期不符）"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 复制需要的文件
    mkdir -p "${SHERPA_DIR}/lib"
    mkdir -p "${SHERPA_DIR}/include"
    cp -r "${extracted_dir}/lib/"* "${SHERPA_DIR}/lib/"
    cp -r "${extracted_dir}/include/"* "${SHERPA_DIR}/include/"

    # 清理
    rm -rf "$tmp_dir"

    # 校验内置 onnxruntime 版本与将要下载的头文件是否一致（ABI 不匹配会静默崩溃）
    if command -v strings >/dev/null 2>&1; then
        local bundled=""
        bundled="$(strings "${SHERPA_DIR}/lib/libonnxruntime.so" 2>/dev/null \
                   | grep -oE '^1\.[0-9]+\.[0-9]+$' | head -n1 || true)"
        if [ -n "$bundled" ] && [ "$bundled" != "$ORT_VERSION" ]; then
            warn "sherpa 内置 onnxruntime 版本为 ${bundled}，与 ORT_VERSION=${ORT_VERSION} 不一致"
            warn "如 ONNX embedding 运行异常，请把 setup.sh 里的 ORT_VERSION 改成 ${bundled} 后重跑"
        fi
    fi

    log "sherpa-onnx 安装完成"
}

# ── 步骤3b: onnxruntime 头文件 ───────────────────────────

install_onnxruntime_headers() {
    log "检查 onnxruntime 头文件..."

    local header="${ORT_DIR}/include/onnxruntime_c_api.h"
    if [ -f "$header" ]; then
        log "onnxruntime 头文件已就绪"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] 下载并解压 ${ORT_ASSET}"
        info "[dry-run]   ${ORT_URL}"
        info "[dry-run]   → ${ORT_DIR}/include"
        return 0
    fi

    info "下载 onnxruntime ${ORT_VERSION} 头文件 (${PKG_ARCH}) ..."

    if command -v curl &>/dev/null && ! curl -sfIL "$ORT_URL" >/dev/null 2>&1; then
        err "资源不存在或网络不可达: ${ORT_URL}"
        err "请确认 ORT_VERSION（当前 ${ORT_VERSION}）是否与 sherpa 内置 libonnxruntime.so 匹配"
        return 1
    fi

    local tmp_dir="/tmp/onnxruntime-$$"
    mkdir -p "$tmp_dir"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${tmp_dir}/${ORT_TAR}" "$ORT_URL" || true
    else
        curl -L -# -o "${tmp_dir}/${ORT_TAR}" "$ORT_URL" || true
    fi

    if [ ! -s "${tmp_dir}/${ORT_TAR}" ]; then
        err "下载 onnxruntime 失败"
        err "请手动下载: ${ORT_URL}"
        err "把 include/ 解压到: ${ORT_DIR}/include"
        rm -rf "$tmp_dir"
        return 1
    fi

    info "解压 onnxruntime ..."
    tar -xzf "${tmp_dir}/${ORT_TAR}" -C "$tmp_dir"

    local extracted_dir
    extracted_dir="$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d -name 'onnxruntime-linux-*' | head -n1)"
    if [ -z "$extracted_dir" ] || [ ! -d "${extracted_dir}/include" ]; then
        err "解压目录未找到（tar 内容与预期不符）"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 只取 include/ —— .so 用 sherpa 包内那份，避免两个不同版本的 libonnxruntime.so
    mkdir -p "${ORT_DIR}/include"
    cp -r "${extracted_dir}/include/"* "${ORT_DIR}/include/"

    rm -rf "$tmp_dir"
    log "onnxruntime 头文件安装完成"
}

# ── 步骤4: 下载模型（sherpa 组件） ──────────────────────

download_hf_model() {
    local repo="$1"
    local filename="$2"
    local dest_dir="$3"
    local desc="$4"

    local hf_url="${HF_BASE}/${repo}/resolve/main/${filename}"
    local mirror_url="${HF_MIRROR}/${repo}/resolve/main/${filename}"

    download_file "$hf_url" "${dest_dir}/${filename}" "$desc" || \
        download_file "$mirror_url" "${dest_dir}/${filename}" "$desc (mirror)"
}

install_models() {
    log "检查模型文件..."

    # ── ASR 模型: SenseVoice ──────────────────────────
    if dir_complete "$ASR_MODEL_DIR" "${ASR_MODEL_FILES[@]}"; then
        log "ASR 模型已就绪 (SenseVoice)"
    else
        info "下载 ASR 模型 (SenseVoice Small int8, ~228MB)..."
        if [ "$DRY_RUN" != "1" ]; then mkdir -p "$ASR_MODEL_DIR"; fi
        local f
        for f in "${ASR_MODEL_FILES[@]}"; do
            download_hf_model "$ASR_MODEL_REPO" "$f" "$ASR_MODEL_DIR" "ASR: $f"
        done
        log "ASR 模型下载完成"
    fi

    # ── 声纹模型: CAM++ ────────────────────────────────
    if dir_complete "$SV_MODEL_DIR" "${SV_MODEL_FILES[@]}"; then
        log "声纹模型已就绪 (CAM++)"
    else
        info "下载声纹模型 (CAM++, ~27MB)..."
        if [ "$DRY_RUN" != "1" ]; then mkdir -p "$SV_MODEL_DIR"; fi
        local f
        for f in "${SV_MODEL_FILES[@]}"; do
            download_hf_model "$SV_MODEL_REPO" "$f" "$SV_MODEL_DIR" "SV: $f"
        done
        log "声纹模型下载完成"
    fi
}

# ── 步骤5: Python 运行时 (.venv) ─────────────────────────

setup_venv() {
    log "准备 Python 运行时 (.venv)..."

    if [ ! -x "${VENV_DIR}/bin/python3" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            info "[dry-run] python3 -m venv ${VENV_DIR}"
        else
            info "创建虚拟环境: ${VENV_DIR}"
            if ! python3 -m venv "$VENV_DIR"; then
                err "创建 .venv 失败。请确认已安装 python3-venv（apt install python3-venv）"
                return 1
            fi
        fi
    else
        log "已存在: ${VENV_DIR}"
    fi

    # pip 原生支持 PIP_EXTRA_INDEX_URL，无需额外传参；
    # aarch64 上 torch/onnxruntime 通常需要厂商 wheel 源，例如:
    #   export PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126
    if [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
        info "使用额外 pip 源: ${PIP_EXTRA_INDEX_URL}"
    fi

    dry "${VENV_DIR}/bin/pip" install -q -U pip

    local comp
    for comp in "$@"; do
        local req="${SCRIPT_DIR}/requirements/${comp}.txt"
        if [ ! -f "$req" ]; then
            warn "缺少依赖清单: ${req}（跳过）"
            continue
        fi
        info "安装 ${comp} 依赖: requirements/${comp}.txt"
        if [ "$DRY_RUN" = "1" ]; then
            info "[dry-run] ${VENV_DIR}/bin/pip install -r ${req}"
            continue
        fi
        if ! "${VENV_DIR}/bin/pip" install -r "$req"; then
            if [ "$comp" = "embedding" ] && [ "$TARGET_ARCH" = "aarch64" ]; then
                # aarch64 上 torch/onnxruntime 的 wheel 常需厂商源，不应中断整体安装
                warn "embedding 依赖安装失败（aarch64 上 torch/onnxruntime 通常需要厂商 wheel 源）"
                warn "可设置 PIP_EXTRA_INDEX_URL 后重试；或改在 x86 上执行"
                warn "  python scripts/export_embedding_model.py"
                warn "再把 models/embedding/ 拷到本机（RAG 技能只依赖 onnxruntime）"
            else
                err "${comp} 依赖安装失败"
                return 1
            fi
        fi
    done
}

# ── 步骤6: Piper 音色模型（piper 组件，尽力而为）─────────

install_piper_voice() {
    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] .venv/bin/python -m piper.download_voices ${PIPER_VOICE}"
        return 0
    fi

    log "检查 Piper 音色 (${PIPER_VOICE})..."
    if ! "${VENV_DIR}/bin/python3" -m piper.download_voices "$PIPER_VOICE" 2>/dev/null; then
        warn "Piper 音色下载失败（不影响其他组件）"
        warn "可稍后手动执行: .venv/bin/python -m piper.download_voices ${PIPER_VOICE}"
        warn "或把 .onnx / .onnx.json 放到 ~/pretrained_models/piper/zh_CN/"
    else
        log "Piper 音色已就绪"
    fi
}

# ── 步骤7: RAG 向量模型（embedding 组件，尽力而为）──────

install_embedding_model() {
    log "导出 RAG 向量模型 (bge-small-zh-v1.5 → ONNX)..."

    if [ -f "${SCRIPT_DIR}/models/embedding/model.onnx" ]; then
        log "向量模型已就绪"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] ${VENV_DIR}/bin/python3 scripts/export_embedding_model.py"
        return 0
    fi

    cd "$SCRIPT_DIR"
    if "${VENV_DIR}/bin/python3" scripts/export_embedding_model.py; then
        log "向量模型导出完成"
    else
        warn "向量模型导出失败 — RAG 技能将不可用"
        warn "可稍后手动执行: .venv/bin/python3 scripts/export_embedding_model.py"
    fi
}

# ── 步骤8: 编译 ─────────────────────────────────────────

build() {
    log "编译 voice_pipeline (${TARGET_ARCH})..."

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] cmake .. -DCMAKE_BUILD_TYPE=Release  (在 ${BUILD_DIR})"
        info "[dry-run] make -j$(nproc 2>/dev/null || echo 4)"
        return 0
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | while IFS= read -r line; do
        echo "       $line"
    done

    local nproc
    nproc=$(nproc 2>/dev/null || echo 4)
    make -j"$nproc" 2>&1 | while IFS= read -r line; do
        echo "       $line"
    done

    cd "$SCRIPT_DIR"

    if [ -f "${BUILD_DIR}/voice_pipeline" ]; then
        log "编译成功 → ${BUILD_DIR}/voice_pipeline"
        log "二进制大小: $(du -h "${BUILD_DIR}/voice_pipeline" | cut -f1)"
    else
        err "编译失败，请检查错误信息"
        exit 1
    fi
}

# ── 步骤9: 运行 ─────────────────────────────────────────

# 从 config.json 读 llm.host（LLM 是远程/云端时不该探测本机 Ollama）
read_llm_host() {
    local cfg="${VOICE_PIPELINE_CONFIG:-${SCRIPT_DIR}/config.json}"
    [ -f "$cfg" ] || return 0
    command -v python3 >/dev/null 2>&1 || return 0

    python3 - "$cfg" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("llm", {}).get("host", "") or "")
except Exception:
    pass
PY
}

run() {
    log "启动语音交互管线..."
    echo ""

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] exec ${BUILD_DIR}/voice_pipeline"
        return 0
    fi

    # 确保在项目根目录运行（模型路径使用相对路径）
    cd "$SCRIPT_DIR"

    # 设置 sherpa-onnx 库路径
    export LD_LIBRARY_PATH="${SHERPA_DIR}/lib:${LD_LIBRARY_PATH:-}"

    # LLM 端点：仅当指向本机时才检查 Ollama（远程/云端 API 直接跳过）
    local llm_host
    llm_host="$(read_llm_host)"
    if [ -z "$llm_host" ]; then
        info "未读到 llm.host（跳过 LLM 端点检查）"
    elif [[ "$llm_host" == *127.0.0.1* || "$llm_host" == *localhost* ]]; then
        curl -s "${llm_host%/}/api/tags" &>/dev/null || \
            warn "Ollama 未运行，请先启动: ollama serve"
    else
        info "LLM 远程端点: ${llm_host}（跳过本地 Ollama 检查）"
    fi

    # Python 运行时：优先仓库内 .venv，其次回退 conda
    if [ -x "${VENV_DIR}/bin/python3" ]; then
        export VOICE_PYTHON="${VENV_DIR}/bin/python3"
        info "Python 运行时: .venv"
    elif [ -z "${CONDA_PREFIX:-}" ]; then
        local conda_base
        for conda_base in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
            if [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
                info "自动激活 conda 环境: chatAudio"
                source "${conda_base}/etc/profile.d/conda.sh"
                conda activate chatAudio 2>/dev/null || {
                    warn "conda 环境 chatAudio 不存在，Piper TTS 可能无法使用"
                    warn "建议改用: ./setup.sh --with piper（创建仓库内 .venv）"
                }
                break
            fi
        done
    fi

    # 修复 conda 环境覆盖 ALSA 插件路径导致录音失败的问题（按架构取路径）
    local alsa_dir="/usr/lib/${MULTIARCH}/alsa-lib"
    if [ -d "$alsa_dir" ]; then
        export ALSA_PLUGIN_DIR="$alsa_dir"
    fi

    # 修复虚拟声卡 Capture 音量过低导致 VAD 检测不到语音
    if amixer -c 0 sget Capture &>/dev/null; then
        amixer -c 0 set Capture 100% &>/dev/null || true
        amixer -c 0 set "Mic Boost (+20dB)" on &>/dev/null || true
        amixer -c 0 set Mic 50% on &>/dev/null || true
    fi

    exec "${BUILD_DIR}/voice_pipeline"
}

# ── 清理 ─────────────────────────────────────────────────

clean() {
    log "清理编译产物..."
    rm -rf "$BUILD_DIR"
    log "清理完成"
}

# ── 组件解析 ─────────────────────────────────────────────

resolve_components() {
    local profile="$1"
    local with_list="$2"
    local without_list="$3"

    COMPONENTS=()
    local -a PROFILE_LIST=()
    profile_to_components "$profile"

    local c
    for c in "${PROFILE_LIST[@]}"; do
        COMPONENTS["$c"]=1
    done

    # 逗号 / 空格分隔均可
    local item
    for item in ${with_list//,/ }; do
        component_valid "$item" || { err "未知组件: $item（可用: ${ALL_COMPONENTS[*]}）"; exit 1; }
        COMPONENTS["$item"]=1
    done
    for item in ${without_list//,/ }; do
        component_valid "$item" || { err "未知组件: $item（可用: ${ALL_COMPONENTS[*]}）"; exit 1; }
        unset "COMPONENTS[$item]"
    done

    # core 无法移除（构建工具链 + espeak + ALSA + .venv）
    COMPONENTS["core"]=1

    local selected=()
    for c in "${ALL_COMPONENTS[@]}"; do
        if component_enabled "$c"; then
            selected+=("$c")
        fi
    done

    info "架构: ${TARGET_ARCH} (multiarch: ${MULTIARCH})"
    info "sherpa-onnx: ${SHERPA_ASSET}"
    info "已选组件: ${selected[*]}"
}

component_valid() {
    local name="$1" c
    for c in "${ALL_COMPONENTS[@]}"; do
        [ "$c" = "$name" ] && return 0
    done
    return 1
}

# 按已选组件执行安装（不含 build）
install_selected() {
    install_deps

    # 编译期硬依赖：onnx_embedding.cpp 无条件 include onnxruntime 头文件，
    # 因此与是否选 embedding 组件（那个只管模型导出）无关
    install_onnxruntime_headers

    if component_enabled sherpa; then
        install_sherpa_onnx
        install_models
    else
        warn "未选 sherpa 组件 — 跳过 sherpa-onnx 与 ASR/声纹模型"
    fi

    # 需要 pip 依赖的组件 → 建 .venv
    local pip_comps=()
    local c
    for c in edge piper embedding; do
        if component_enabled "$c"; then
            pip_comps+=("$c")
        fi
    done

    if [ ${#pip_comps[@]} -gt 0 ]; then
        setup_venv "${pip_comps[@]}"
        if component_enabled piper; then
            install_piper_voice
        fi
        if component_enabled embedding; then
            install_embedding_model
        fi
    else
        info "未选 edge/piper/embedding 组件 — 跳过 Python 运行时"
    fi
}

# ── 帮助 / 主流程 ────────────────────────────────────────

print_banner() {
    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║    ASR-LLM-TTS 语音交互管线 一键安装     ║"
    echo "  ║    ASR → 唤醒词 → 声纹 → LLM → TTS       ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo ""
}

print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "安装范围:"
    echo "  --profile <name>   组件集合: minimal | client | full (默认 client)"
    echo "  --with <a,b>       额外启用组件（可重复）"
    echo "  --without <a,b>    禁用组件（可重复）"
    echo "  --all              等价于 --profile full"
    echo "  --dry-run          只打印将要执行的动作，不落地"
    echo ""
    echo "动作:"
    echo "  (无参数)           安装所选组件 + 编译"
    echo "  --run              安装所选组件 + 编译 + 运行"
    echo "  --models           仅下载运行库与模型（core + sherpa + onnxruntime）"
    echo "  --build            仅编译"
    echo "  --clean            清理编译产物"
    echo "  --help             显示此帮助"
    echo ""
    echo "组件:"
    echo "  core               构建工具链 + espeak-ng + ALSA + .venv（总是安装）"
    echo "  sherpa             sherpa-onnx 运行时 + ASR + 声纹模型（按架构自动选择）"
    echo "  edge               edge-tts 云端 TTS（ffmpeg + edge-tts）"
    echo "  piper              Piper 本地 TTS（piper-tts + 音色模型）"
    echo "  embedding          RAG 向量模型（torch/modelscope + ONNX 导出）"
    echo ""
    echo "profile:"
    echo "  minimal            core sherpa"
    echo "  client             core sherpa edge                 （默认）"
    echo "  full               core sherpa edge piper embedding"
    echo ""
    echo "环境变量:"
    echo "  PIP_EXTRA_INDEX_URL  额外 pip 源（aarch64 上 torch/onnxruntime 需要）"
    echo ""
    echo "运行时依赖:"
    echo "  - LLM 服务（config.json 里 llm.host 指向 Ollama 或云端 API）"
    echo "  - espeak-ng 已安装"
    echo "  - 麦克风/音箱可用 (ALSA)"
}

main() {
    local profile="client"
    local mode="install"
    local with_list=""
    local without_list=""

    print_banner

    while [ $# -gt 0 ]; do
        case "$1" in
            --help|-h)      print_help; exit 0 ;;
            --clean)        clean; exit 0 ;;
            --profile)      profile="${2:?--profile 需要参数}"; shift 2 ;;
            --profile=*)    profile="${1#*=}"; shift ;;
            --with)         with_list+="${with_list:+,}${2:?--with 需要参数}"; shift 2 ;;
            --with=*)       with_list+="${with_list:+,}${1#*=}"; shift ;;
            --without)      without_list+="${without_list:+,}${2:?--without 需要参数}"; shift 2 ;;
            --without=*)    without_list+="${without_list:+,}${1#*=}"; shift ;;
            --all)          profile="full"; shift ;;
            --dry-run)      DRY_RUN=1; shift ;;
            --models)       mode="models"; shift ;;
            --build)        mode="build"; shift ;;
            --run)          mode="run"; shift ;;
            *) err "未知参数: $1"; print_help; exit 1 ;;
        esac
    done

    case "$mode" in
        build)
            build
            exit 0
            ;;
        models)
            # 旧语义 + onnxruntime 头文件: 备齐所有"下载类"产物，但不编译
            resolve_components minimal "$with_list" "$without_list"
            install_deps
            install_sherpa_onnx
            install_onnxruntime_headers
            install_models
            log "运行库与模型下载完成！"
            log "下一步: $0 --build && $0 --run"
            exit 0
            ;;
        run)
            resolve_components "$profile" "$with_list" "$without_list"
            install_selected
            build
            run
            ;;
        install)
            resolve_components "$profile" "$with_list" "$without_list"
            install_selected
            build
            log "全部完成！"
            echo ""
            log "运行: $0 --run"
            log "或者: cd ${SCRIPT_DIR} && ./src/build/voice_pipeline"
            ;;
    esac
}

main "$@"
