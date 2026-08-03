#!/usr/bin/env python3
"""
Edge-TTS CLI wrapper for RTSP voice pipeline.
One-shot mode:  python3 edge_tts_cli.py --text "你好" --output out.wav
Server mode:    python3 edge_tts_cli.py --server  (stdin protocol, persistent)

Server protocol (lines on stdin):
  INPUT:  <output_path>|<text>
  OUTPUT: OK|<output_path>|<samples>|<duration_s>|<time_ms>
          ERR|<message>

Chinese voices: zh-CN-XiaoxiaoNeural (warm), zh-CN-YunxiNeural (male),
                zh-CN-XiaoyiNeural (bright)
"""

import argparse
import asyncio
import sys
import os
import time
import subprocess
import tempfile


VOICE = "zh-CN-XiaoxiaoNeural"


async def synthesize(text, output_path, voice=VOICE):
    """Synthesize text to WAV using Edge-TTS."""
    import edge_tts

    # Generate mp3 temp file
    tmp_mp3 = output_path + ".mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_mp3)

    # Convert mp3 → WAV (16kHz mono)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_mp3,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-loglevel", "error",
        output_path
    ], check=True, capture_output=True)

    # Cleanup
    try:
        os.remove(tmp_mp3)
    except OSError:
        pass


def run_oneshot(text, output_path, voice=VOICE):
    t_start = time.time()
    asyncio.run(synthesize(text, output_path, voice))
    t_ms = (time.time() - t_start) * 1000

    # Get audio stats
    import wave
    with wave.open(output_path, 'rb') as wf:
        n_samples = wf.getnframes()
        dur = n_samples / wf.getframerate()

    print(f"[EdgeTTS] {n_samples} samples ({dur:.1f}s) in {t_ms:.0f}ms → {output_path}",
          file=sys.stderr)


def run_server():
    """Persistent mode: read text lines from stdin, output WAV files."""
    print(f"[EdgeTTS] ready (voice={VOICE})", file=sys.stderr, flush=True)
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "EXIT":
            break

        # Format: output_path|text
        parts = line.split("|", 1)
        if len(parts) != 2:
            print(f"ERR|bad format: {line}", flush=True)
            continue

        output_path, text = parts
        text = text.strip()
        if not text:
            print(f"ERR|empty text", flush=True)
            continue

        try:
            t_start = time.time()
            asyncio.run(synthesize(text, output_path))

            import wave
            with wave.open(output_path, 'rb') as wf:
                n_samples = wf.getnframes()
                dur = n_samples / wf.getframerate()

            t_ms = (time.time() - t_start) * 1000
            print(f"OK|{output_path}|{n_samples}|{dur:.1f}s|{t_ms:.0f}ms", flush=True)
        except Exception as e:
            print(f"ERR|{e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Edge-TTS CLI for RTSP server")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output", help="Output WAV file path")
    parser.add_argument("--voice", default=VOICE,
                        help=f"Voice name (default: {VOICE})")
    parser.add_argument("--server", action="store_true",
                        help="Run in persistent server mode (stdin protocol)")
    args = parser.parse_args()

    if args.server:
        run_server()
    elif args.text and args.output:
        run_oneshot(args.text, args.output, args.voice)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
