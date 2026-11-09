#!/bin/bash
# scratch-only remote runner: one serve arm + pinned h2h client (cold 128/512/2048) on raider 5090.
# usage: raider-arm.sh <label> <binary> [ENV=VAL ...]
label=$1; bin=$2; shift 2
W=~/cuda-prefill-sm120-20261106; out=$W/runs; mkdir -p $out
PORT=18181
export CUDA_VISIBLE_DEVICES=GPU-2a2a837f-c82d-0f22-4538-e3a063903380
export LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64
export FAK_WORKSPACE_ADMISSION_PERMISSIVE=1 FAK_GPU_BUDGET_MB=auto FAK_STREAM_Q4K=1 FAK_INKERNEL_RADIX=off
unset FAK_CUDA_Q4K_PREFILL FAK_CUDA_QWEN38_PREFILL_ATTN FAK_CUDA_TIMESTAMP_PROFILE
for kv in "$@"; do export "$kv"; done
M=/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf; MID=Qwen3.8-27B-Q4_K_M
CTX=${CTX:-4096}; SPEC=${SPEC-}; COLD=${COLD:-128,512,2048}; REPS=${REPS:-3}
cd $W
{ echo "$(date -u +%FT%TZ) $label env: $* | $bin serve --config $W/config.toml --engine inkernel --backend cuda --gguf $M --model $MID --kv-precision fp32 --native-context-tokens $CTX --addr 127.0.0.1:$PORT --speculative '$SPEC'"; sha256sum $bin; } >> $out/argv.txt
$bin serve --config $W/config.toml --engine inkernel --backend cuda --gguf $M --model $MID --kv-precision fp32 \
  --native-context-tokens $CTX --addr 127.0.0.1:$PORT --speculative "$SPEC" > $out/$label.serve.log 2>&1 &
spid=$!
t0=$(date +%s); for i in $(seq 1 300); do curl -sf -o /dev/null http://127.0.0.1:$PORT/readyz && break; kill -0 $spid 2>/dev/null || { echo SERVE_DIED; tail -15 $out/$label.serve.log; echo "ARM_DONE $label"; exit 3; }; sleep 2; done
echo "ready_after_s=$(( $(date +%s)-t0 ))"
nvidia-smi --query-gpu=timestamp,memory.used,power.draw,power.limit,clocks.sm --format=csv,noheader -lms 1000 > $out/$label.smi.csv 2>&1 &
smipid=$!
~/oss-engines/fak-h2h-c9870f151d bench h2h run --arm fak=http://127.0.0.1:$PORT/v1 --model $MID --cold $COLD --decode ${DECODE:-0} --turns 0 --answer ${ANSWER:-8} --reps $REPS --run-id $label --ledger $out/$label.h2h.jsonl --host raider-fak > $out/$label.client.stdout 2> $out/$label.client.stderr
cat $out/$label.client.stderr | grep -E "^(cold|decode)"
# greedy parity probe: fixed prompt, temperature 0, 64 tokens
curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"$(head -c 1600 $W/parity-prompt.txt)\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > $out/$label.parity.json
python3 -c "import json,hashlib,sys; d=json.load(open('$out/$label.parity.json')); t=d['choices'][0]['message'].get('content') or ''; print('parity sha256', hashlib.sha256(t.encode()).hexdigest()[:16], 'usage', d.get('usage',{}).get('prompt_tokens'), d.get('usage',{}).get('completion_tokens'))"
kill $smipid 2>/dev/null; kill $spid; wait $spid 2>/dev/null; sleep 3
echo "ARM_DONE $label"
