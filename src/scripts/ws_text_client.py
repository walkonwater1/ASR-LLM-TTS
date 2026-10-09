#!/usr/bin/env python3
"""
WebSocket 文字客户端 — 给 ws_voice_server 发一条 text 消息并打印回复。

用于**不接麦克风**验证 LLM 链路：ws_voice_server 只初始化管线、不启动
采集线程，所以整条路径完全不碰录音设备。走的是真实的
VoicePipeline::process_text，比裸 curl 更能证明链路通。

零依赖，只用 Python 标准库。

用法:
    ./build/ws_voice_server config.json 9002      # 先起服务端
    python3 src/scripts/ws_text_client.py "你好"   # 另开一个终端发文字

    指定地址: python3 src/scripts/ws_text_client.py "你好" 192.168.10.2:9002

协议:
    发送  {"type":"text","content":"你好"}
    回复  {"type":"reply","text":"...","emotion":""}
"""
import base64
import json
import os
import socket
import struct
import sys


def handshake(sock, host, port):
    """完成 WebSocket 握手（RFC 6455），返回响应头之后可能已经到达的字节。"""
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("连接被关闭（握手阶段）")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    status = head.split(b"\r\n")[0].decode(errors="replace")
    if "101" not in status:
        raise RuntimeError(f"握手失败: {status}")
    return rest


def send_text(sock, payload: bytes):
    """发一个带掩码的 text 帧（客户端发出的帧必须掩码）。"""
    n = len(payload)
    header = bytearray([0x81])  # FIN + opcode=text
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header += struct.pack(">H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack(">Q", n)
    mask = os.urandom(4)
    header += mask
    sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


def send_control(sock, opcode, payload=b""):
    """发控制帧（用于回 pong）。"""
    mask = os.urandom(4)
    header = bytes([0x80 | opcode, 0x80 | len(payload)]) + mask
    sock.sendall(header + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


class Reader:
    """按帧读取；处理 7/16/64 位长度与掩码（服务端帧不掩码）。"""

    def __init__(self, sock, initial=b""):
        self.sock = sock
        self.buf = bytearray(initial)

    def _need(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("连接被服务端关闭")
            self.buf += chunk

    def frame(self):
        self._need(2)
        opcode = self.buf[0] & 0x0F
        masked = self.buf[1] & 0x80
        ln = self.buf[1] & 0x7F
        off = 2
        if ln == 126:
            self._need(4)
            ln = struct.unpack(">H", self.buf[2:4])[0]
            off = 4
        elif ln == 127:
            self._need(10)
            ln = struct.unpack(">Q", self.buf[2:10])[0]
            off = 10
        if masked:
            self._need(off + 4)
            mask = bytes(self.buf[off:off + 4])
            off += 4
            self._need(off + ln)
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(self.buf[off:off + ln]))
        else:
            self._need(off + ln)
            data = bytes(self.buf[off:off + ln])
        del self.buf[:off + ln]
        return opcode, data


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好"
    target = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:9002"
    host, _, port = target.partition(":")
    port = int(port or 9002)

    try:
        sock = socket.create_connection((host, port), timeout=180)
    except OSError as e:
        print(f"❌ 连不上 {host}:{port} —— {e}")
        print("   服务端起了吗？ ./build/ws_voice_server config.json 9002")
        return 3

    try:
        rest = handshake(sock, host, port)
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1

    reader = Reader(sock, rest)

    body = json.dumps({"type": "text", "content": text}, ensure_ascii=False)
    print(f"→ 发送: {body}")
    send_text(sock, body.encode())

    for _ in range(30):
        try:
            opcode, data = reader.frame()
        except RuntimeError as e:
            print(f"❌ {e}")
            return 1

        if opcode == 0x8:                       # close
            print("服务端关闭了连接")
            return 1
        if opcode == 0x9:                       # ping → pong
            send_control(sock, 0xA, data)
            continue
        if opcode not in (0x1, 0x2):            # 非数据帧
            continue

        raw = data.decode("utf-8", "replace")
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            print(f"← 非 JSON 帧: {raw[:400]}")
            continue

        kind = msg.get("type")
        print(f"← [{kind}] {json.dumps(msg, ensure_ascii=False)}")
        if kind == "reply":
            print(f"\n✅ LLM 链路通：{msg.get('text', '')}")
            return 0
        if kind == "error":
            print(f"\n❌ 服务端报错：{msg.get('message', '')}")
            return 2

    print("⚠️ 没等到 reply")
    return 1


if __name__ == "__main__":
    sys.exit(main())
