#!/usr/bin/env python3
"""claude-cli-shim — makes `claude -p` look like an OpenAI-compatible HTTP endpoint.

Rebuilt 2026-06-17 from architecture doc (cos-interactive skill reference),
gateway/run.py launcher, LaunchAgent plist, observed log/stderr behaviour,
live-probe response contracts, and traceback line numbers (408/476/483).

Architecture (see cos-interactive/references/claude-cli-shim-architecture.md):
  - Listens on 127.0.0.1:8788 (loopback only, no auth surface)
  - Speaks OpenAI /v1/chat/completions (streaming SSE + buffered)
  - Translates Claude stream-json → OpenAI chat.completion.chunk deltas
  - Admission gate: CoS keywords OR [CLAUDE_BUILDER] OR [SEVERE]/[CoS_DIRECT]
  - Rate limiting: sliding window 100 req/hr
  - Token tracking: JSONL telemetry (chars/4 estimates)
  - Health check at GET /health
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("CLAUDE_SHIM_PORT", "8788"))

CLAUDE_BIN = os.environ.get(
    "CLAUDE_CLI_BIN",
    os.path.expanduser("~/.local/bin/claude"),
)

# Rate limiting — sliding window
RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = 3600  # seconds (1 hour)

# Claude CLI invocation
CLAUDE_MAX_TURNS = 5
CLAUDE_MODEL = "opus"

# Telemetry
TELEMETRY_DIR = os.path.expanduser("~/.hermes/telemetry")
TELEMETRY_FILE = os.path.join(TELEMETRY_DIR, "claude-shim-usage.jsonl")

# Logging
LOG_DIR = os.path.expanduser("~/.hermes/logs")
LOG_FILE = os.path.join(LOG_DIR, "claude-shim.log")

# ---------------------------------------------------------------------------
# Admission gate — CoS-lane-only
# ---------------------------------------------------------------------------
# CoS strategy keywords (from architecture doc §Guards table)
COS_KEYWORD_RE = re.compile(
    r"\b(cos|chief of staff|strateg(?:y|ic)|infra(?:structure)?)\b",
    re.IGNORECASE,
)

# Explicit markers that bypass the keyword check
MARKER_RE = re.compile(
    r"\[(CLAUDE_BUILDER|SEVERE|CoS_DIRECT)\]"
)

# Reframe-tell detection (anti-gaming: flags attempts to shoehorn CoS
# keywords into non-CoS content — observed in stdout log 2026-06-15)
REFRAME_TELL_PATTERNS = [
    re.compile(r"\breframe\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you|this)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_rate_timestamps: deque = deque()


def _rate_check() -> bool:
    """Return True if request is within rate limit."""
    now = time.time()
    with _rate_lock:
        # Purge old entries outside the window
        while _rate_timestamps and _rate_timestamps[0] < now - RATE_LIMIT_WINDOW:
            _rate_timestamps.popleft()
        if len(_rate_timestamps) >= RATE_LIMIT_MAX:
            return False
        _rate_timestamps.append(now)
        return True


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
_telemetry_lock = threading.Lock()


def _log_telemetry(
    request_id: str,
    prompt_chars: int,
    response_chars: int,
    severe: bool,
    stream: bool,
):
    """Append one telemetry row to the JSONL file."""
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    row = {
        "ts": int(time.time()),
        "request_id": request_id,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "prompt_tokens_est": prompt_chars // 4,
        "completion_tokens_est": response_chars // 4,
        "severe": severe,
        "stream": stream,
    }
    with _telemetry_lock:
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("claude-cli-shim")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_fh = logging.FileHandler(LOG_FILE)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)


# ---------------------------------------------------------------------------
# Admission check
# ---------------------------------------------------------------------------
def _check_admission(system_message: str) -> tuple:
    """Return (admitted: bool, severe: bool, reframe_tells: int, reason: str)."""
    if not system_message:
        return False, False, 0, "no system message"

    # Check explicit markers first
    if MARKER_RE.search(system_message):
        severe = "[SEVERE]" in system_message
        return True, severe, 0, "marker"

    # Check CoS keywords
    if COS_KEYWORD_RE.search(system_message):
        # Count reframe tells (anti-gaming detection)
        reframe_tells = sum(
            1 for pat in REFRAME_TELL_PATTERNS if pat.search(system_message)
        )
        return True, False, reframe_tells, "cos_keyword"

    return False, False, 0, "no CoS-lane signal, not severe"


# ---------------------------------------------------------------------------
# Prompt extraction from OpenAI messages array
# ---------------------------------------------------------------------------
def _build_prompt(messages: list) -> tuple:
    """Extract system message and build prompt for claude -p.

    Returns (system_message: str, full_prompt: str, prompt_chars: int).
    """
    system_parts = []
    user_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[Assistant]: {content}")

    system_message = "\n".join(system_parts)
    # claude -p takes the prompt via stdin or --prompt flag
    # We build a combined prompt that includes system context
    full_prompt = ""
    if system_message:
        full_prompt += system_message + "\n\n"
    full_prompt += "\n".join(user_parts)

    return system_message, full_prompt, len(full_prompt)


# ---------------------------------------------------------------------------
# Claude CLI invocation
# ---------------------------------------------------------------------------
def _invoke_claude_stream(prompt):
    """Run `claude -p` in streaming mode. Yields (chunk_text, is_done) tuples."""
    cmd = [
        CLAUDE_BIN,
        "-p",
        "--model", CLAUDE_MODEL,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(CLAUDE_MAX_TURNS),
    ]

    logger.info(
        "invoking: %s (prompt %d chars)", " ".join(cmd), len(prompt)
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write(prompt.encode("utf-8"))
    proc.stdin.close()

    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                msg = obj.get("message", {})
                content_blocks = msg.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        yield block.get("text", ""), False
            elif obj.get("type") == "result":
                result_text = obj.get("result", "")
                if result_text:
                    yield result_text, False
                yield "", True
                break
        except json.JSONDecodeError:
            continue

    proc.wait()
    rc = proc.returncode
    if rc != 0:
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
        logger.error("claude failed rc=%d: %s", rc, stderr_out or f"claude exited {rc}")
    else:
        logger.info("claude completed rc=0 emitted=True")


def _invoke_claude_buffered(prompt):
    """Run `claude -p` in buffered mode. Returns (full_text, rc)."""
    cmd = [
        CLAUDE_BIN,
        "-p",
        "--model", CLAUDE_MODEL,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(CLAUDE_MAX_TURNS),
    ]

    logger.info(
        "invoking: %s (prompt %d chars)", " ".join(cmd), len(prompt)
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write(prompt.encode("utf-8"))
    proc.stdin.close()

    stdout_data = proc.stdout.read().decode("utf-8", errors="replace")
    proc.wait()
    rc = proc.returncode

    full_text = ""
    for line in stdout_data.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                msg = obj.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "text":
                        full_text += block.get("text", "")
            elif obj.get("type") == "result":
                result_text = obj.get("result", "")
                if result_text and not full_text:
                    full_text = result_text
        except json.JSONDecodeError:
            continue

    emitted = bool(full_text)
    if rc != 0:
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
        logger.error("claude failed rc=%d: %s", rc, stderr_out or f"claude exited {rc}")
    else:
        logger.info("claude completed rc=0 emitted=%s", emitted)

    return full_text, rc


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class ShimHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/chat/completions handler."""

    server_version = "claude-cli-shim/1.0"

    def log_message(self, format, *args):
        """Override to route access logs through our logger."""
        logger.debug("HTTP %s", format % args)

    # ---- GET /health ----
    def do_GET(self):
        if self.path == "/health" or self.path == "/health?":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    # ---- POST /v1/chat/completions ----
    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/chat/completions?"):
            self._send_json(404, {"error": "not found"})
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": {"message": "empty request body", "type": "invalid_request"}})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request"}})
            return

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        request_id = uuid.uuid4().hex[:12]

        # Rate limit check
        if not _rate_check():
            self._send_json(
                429,
                {"error": {"message": "rate limit exceeded (100 req/hr)", "type": "rate_limit"}},
            )
            return

        # Build prompt and extract system message
        system_message, full_prompt, prompt_chars = _build_prompt(messages)

        # Admission gate
        admitted, severe, reframe_tells, reason = _check_admission(system_message)

        if not admitted:
            logger.info("rejected non-CoS request (%s)", reason)
            self._send_json(
                403,
                {
                    "error": {
                        "message": "Claude is CoS only. This task should be routed to DeepSeek, Kimi, or Gemini.",
                        "type": "cos_lane_only",
                    }
                },
            )
            return

        # Log acceptance
        # Check for [CLAUDE_BUILDER] marker specifically
        if "[CLAUDE_BUILDER]" in (system_message or ""):
            logger.info("accepted Opus builder-lane request ([CLAUDE_BUILDER])")
        elif severe:
            logger.info("accepted SEVERE escalation request")
        elif reframe_tells > 0:
            logger.warning(
                "accepted CoS request with %d reframe tells — flagged",
                reframe_tells,
            )

        logger.info(
            "accepted request_id=%s severe=%s stream=%s",
            request_id, severe, stream,
        )

        # Invoke Claude
        if stream:
            self._stream_response(full_prompt, request_id, severe, prompt_chars)
        else:
            self._buffered_response(full_prompt, request_id, severe, prompt_chars)

    # ---- Streaming SSE response ----
    def _stream_response(self, prompt, request_id, severe, prompt_chars):
        """Stream OpenAI-compatible SSE chunks."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        response_chars = 0
        chat_id = f"chatcmpl-claude-{int(time.time())}"
        created = int(time.time())
        chunk_index = 0

        try:
            # Initial chunk carries the assistant role (matches live shim wire format)
            role_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "claude-cli-opus",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                ],
            }
            self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

            for text, is_done in _invoke_claude_stream(prompt):
                if is_done:
                    # Send final chunk with finish_reason
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "claude-cli-opus",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    )
                    break

                if text:
                    response_chars += len(text)
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "claude-cli-opus",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    chunk_index += 1

            # Usage summary chunk — emitted BEFORE [DONE], matching live shim order
            usage_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "claude-cli-opus",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_chars // 4,
                    "completion_tokens": response_chars // 4,
                    "total_tokens": (prompt_chars + response_chars) // 4,
                },
            }
            self.wfile.write(
                f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
            )
            self.wfile.flush()

            # [DONE] sentinel — final line
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            logger.warning("client disconnected mid-stream request_id=%s", request_id)

        # Log telemetry
        _log_telemetry(request_id, prompt_chars, response_chars, severe, True)

    # ---- Buffered (non-streaming) response ----
    def _buffered_response(self, prompt, request_id, severe, prompt_chars):
        """Return a complete OpenAI chat.completion object."""
        try:
            full_text, rc = _invoke_claude_buffered(prompt)
            response_chars = len(full_text)

            chat_id = f"chatcmpl-claude-{int(time.time())}"
            payload = {
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "claude-cli-opus",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text,
                        },
                        "finish_reason": "stop" if rc == 0 else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_chars // 4,
                    "completion_tokens": response_chars // 4,
                    "total_tokens": (prompt_chars + response_chars) // 4,
                },
            }
            self._send_json(200, payload)
        except Exception as e:
            logger.error("_buffered_response failed: %s", e, exc_info=True)
            try:
                self._send_json(500, {"error": {"message": str(e), "type": "server_error"}})
            except Exception:
                pass

        # Log telemetry
        _log_telemetry(request_id, prompt_chars, response_chars, severe, False)

    # ---- JSON response helper ----
    def _send_json(self, status_code: int, payload: dict):
        """Send a JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Threaded HTTP Server
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread."""
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), ShimHandler)
    logger.info(
        "claude-cli-shim listening on http://%s:%d (bin=%s)",
        LISTEN_HOST, LISTEN_PORT, CLAUDE_BIN,
    )
    print(
        f"claude-cli-shim listening on http://{LISTEN_HOST}:{LISTEN_PORT}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down (KeyboardInterrupt)")
        server.shutdown()


if __name__ == "__main__":
    main()
