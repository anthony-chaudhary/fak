#!/bin/bash
# scratch-only: llama.cpp b11870 denominator, same h2h client, same model, same GPU.
label=$1
W=~/cuda-prefill-sm120-20261106; out=$W/runs; PORT=18182
export CUDA_VISIBLE_DEVICES=GPU-2a2a837f-c82d-0f22-4538-e3a063903380
L=~/oss-engines/llama.cpp-b11870/build-cuda/bin/llama-server
M=/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf; MID=Qwen3.8-27B-Q4_K_M
{ echo "$(date -u +%FT%TZ) $label $L -m $M -a $MID -ngl 99 -c 4096 -np 1 --port $PORT"; sha256sum $L; } >> $out/argv.txt
$L -m $M -a $MID -ngl 99 -c 4096 -np 1 --port $PORT > $out/$label.serve.log 2>&1 &
spid=$!
for i in $(seq 1 200); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 2; done
~/oss-engines/fak-h2h-c9870f151d bench h2h run --arm llama=http://127.0.0.1:$PORT/v1 --model $MID --cold ${COLD:-128,512,2048} --decode 0 --turns 0 --answer 8 --reps ${REPS:-3} --run-id $label --ledger $out/$label.h2h.jsonl --host raider-fak --slots-url http://127.0.0.1:$PORT/slots > $out/$label.client.stdout 2> $out/$label.client.stderr
grep -E "^(cold|decode)" $out/$label.client.stderr
kill $spid; wait $spid 2>/dev/null; sleep 3; echo "ARM_DONE $label"
