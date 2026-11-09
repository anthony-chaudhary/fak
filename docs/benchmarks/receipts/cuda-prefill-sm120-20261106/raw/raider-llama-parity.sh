#!/bin/bash
W=~/cuda-prefill-sm120-20261106; out=$W/runs; PORT=18182
export CUDA_VISIBLE_DEVICES=GPU-2a2a837f-c82d-0f22-4538-e3a063903380
L=~/oss-engines/llama.cpp-b11870/build-cuda/bin/llama-server
M=/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf; MID=Qwen3.8-27B-Q4_K_M
$L -m $M -a $MID -ngl 99 -c 4096 -np 1 --port $PORT > $out/Lp.serve.log 2>&1 &
spid=$!
for i in $(seq 1 200); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 2; done
curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"$(head -c 1600 $W/parity-prompt.txt)\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > $out/Lp.parity.json
python3 -c "import json; d=json.load(open('$out/Lp.parity.json')); print(repr(d['choices'][0]['message'].get('content')), d.get('usage'))"
kill $spid; wait $spid 2>/dev/null
