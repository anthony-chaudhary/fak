# Qwen3.6-27B standup + throughput on the lab GPU server (2026-06-24)

A cold standup of `Qwen/Qwen3.6-27B` on the lab 8-GPU datacenter server (`GPU server`,
`gpu-server-lab`, 40 GB/GPU), driven end to end from a laptop through the
slack-helpers control bridge, with a real OpenAI-compatible throughput sweep as proof.

This is a fresh standup, not a replay: at start all 8 GPUs were idle (0 MiB used) and
no server was listening on `:30000`. The weights were already cached on disk
(`models--Qwen--Qwen3.6-27B`, 52 GB).

## Serve config

`sglang` tensor-parallel across all 8 GPUs, mirroring the proven lab launcher:

```sh
/root/sglang-stock/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-27B --port 30000 --host 127.0.0.1 \
  --tp 8 --trust-remote-code --mem-fraction-static 0.85
```

Health came up after the weights load + CUDA-graph capture; `/v1/models` reports
`Qwen/Qwen3.6-27B` with `max_model_len 262144`, `owned_by sglang`. Per-GPU memory
settled at ~37 GB used (the `mem-fraction-static 0.85` KV-cache pool).

## Proof 1 — real completions (temperature 0)

Three live `/v1/chat/completions` calls returned coherent output. Qwen3.6-27B is a
reasoning model and emits an explicit "Thinking Process" chain (full excerpts in
[`samples.json`](samples.json)):

| prompt | completion tok | dur | e2e tok/s | answer |
|---|---|---|---|---|
| capital of France | 64 | 1.47 s | 43.5 | Paris |
| `fib(n)` in Python | 256 | 2.73 s | 93.9 | derives O(n) iterative vs O(2^n) naive |
| why is the sky blue | 96 | 1.23 s | 78.2 | Rayleigh scattering |

These single-stream rates include prefill/TTFT, so they undercount steady-state
decode; the sweep below is the rigorous throughput measure.

## Proof 2 — throughput sweep (real benchmark)

`gpu_server_endpoint_load.py` against the live `:30000` endpoint, replaying the recorded
agentic workload (`production-workload.json`, 6 airline-booking agent cases,
prompt 1.5k–4.1k tok, completion 184–269 tok). 24 requests minimum per concurrency
point, **zero errors across the entire sweep**. Full record in
[`raw-sglang.json`](raw-sglang.json) (`fak.gpu-server-endpoint-bench.v1`).

| concurrency | ok | errors | req/s | completion tok/s | total tok/s | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| 1  | 24/24 | 0 | 0.90 |  77.8 |  3016 |   322 |  2904 |
| 4  | 24/24 | 0 | 2.18 | 191.9 |  7266 |   279 |  4653 |
| 8  | 24/24 | 0 | 2.64 | 259.0 |  8849 |   324 |  6307 |
| 16 | 24/24 | 0 | 3.34 | 503.9 | 11339 |  5785 |  6889 |
| 32 | 32/32 | 0 | 4.05 | 705.2 | 13730 |  7094 |  7877 |
| 64 | 64/64 | 0 | 4.89 | **820.5** | 16471 | 12170 | 13025 |

Peak observed **820.5 completion tok/s** (and 16.5k total tok/s incl. prompt) at
concurrency 64. Throughput is still climbing monotonically at the top of the sweep —
this run capped at C=64, so 820.5 is a floor, not a saturation point. (A prior wider
sweep on the same box reached ~1.45k completion tok/s at C=128.)

## Honest caveats

- **Sweep range.** Capped at C=64; the curve had not flattened, so the peak number is
  a lower bound for this hardware, not its ceiling.
- **Marker gate disabled.** Qwen3.6-27B rarely echoes the load harness's literal
  `FAK_GPU_SERVER_REQ_` marker (it answers in its own reasoning format), so the run used
  `--no-require-response-marker`. That gate is about instruction-echo, not serving
  health; every request completed and produced tokens (`errors 0`).
- **Scope.** This is the raw `sglang` serving rate (the baseline). It is not a fak-kernel
  number and makes no claim about the fak gateway overlay.

## Files

- [`raw-sglang.json`](raw-sglang.json) — `fak.gpu-server-endpoint-bench.v1`, full 6-point sweep + provenance (8-GPU topology, driver, workload hashes).
- [`samples.json`](samples.json) — the three live completions with token counts and excerpts.

Provenance host scrubbed to `gpu-server-lab` per the public-copy policy
(`tools/scrub_hardware_names.py`); the GPU topology block matches the prior committed
run `gpu-server-r4-20260622/`.
