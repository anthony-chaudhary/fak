#!/usr/bin/env bash
# One-command bring-up + smoke check for the governed-local-model stack.
#
# Brings the compose stack up, waits (bounded) for the governed front door to answer
# /healthz, then shows the gate working: an unauthenticated /v1 call is refused, an
# authenticated one goes through. The stack is left running; tear it down with
# `docker compose down -v`.
set -euo pipefail
cd "$(dirname "$0")"

# A key is mandatory (compose refuses to start without one). Mint an ephemeral one
# for this run if the caller did not export their own.
: "${FAK_GATEWAY_KEY:=$(openssl rand -hex 32)}"
export FAK_GATEWAY_KEY

echo "== bringing the stack up (first run pulls the images + qwen2.5:1.5b, ~2 GB) =="
docker compose up -d

echo "== waiting for the governed front door on :8080 (bounded: 150 tries, 2s apart) =="
for _ in $(seq 1 150); do
  curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 && break
  sleep 2
done

echo "== /healthz (the one always-unauthenticated route) =="
curl -s http://localhost:8080/healthz
echo

echo "== an unauthenticated /v1 call is refused by the gate =="
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://localhost:8080/v1/models

echo "== an authenticated chat call crosses the gate =="
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Say OK."}]}'
echo

echo "== up. tear down with: docker compose down -v =="
