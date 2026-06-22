#!/usr/bin/env bash
# Restart ONLY the fak `serve` gateway on :8080 with a given capability-floor
# --policy, leaving the SGLang upstream (:30000) running. Used to add a third
# SWE-bench arm: the fak gateway with the coding agent's `bash` tool ALLOWED (vs
# the built-in DefaultPolicy that DEFAULT_DENYs it). All shell metachars the Slack
# bridge escapes (&, >, <) live in this shipped file. Type only `bash <thisfile>`.
set -u
POLICY="${1:-/tmp/swebench-coding-agent-policy.json}"
LOG="${2:-/tmp/fak-gateway-allow.log}"
FAK="/srv/fleet/tools/.bin/fak"
pkill -f "fak serve" || true
sleep 2
setsid "$FAK" serve --addr 127.0.0.1:8080 --provider openai \
  --base-url http://127.0.0.1:30000/v1 --model qwen36-27b \
  --api-key-env NONE_LOCAL --policy "$POLICY" </dev/null >"$LOG" 2>&1 &
echo "restarted fak gateway pid $! policy=$POLICY log=$LOG"
