---
loop: goal
goal_slug: mac-oss-performance-number-one
witness: "go test -v ./internal/metalgemm/... && fak macbench all --model qwen38:27b --json"
budget: { max_iters: 20 }
lane: metalgemm
---
# Objective
Deliver undisputed #1 open-source performance for Apple Silicon (macOS / Metal / ARM64) in public `fak`, pairing world-class single-stream and multi-agent inference throughput with a frictionless "one-touch" developer experience (`fak up`) that acts as the primary physical proof wedge for commercial server-side inference.

# Non-Goals
- Do not build proprietary multi-tenant cloud features or commercial metering in public `fak` (preserved strictly in `fak-private` per `BOUNDARY.md`).
- Do not introduce non-Go external runtime dependencies, Python frameworks, or loose scripts.
- Do not compromise on-device safety floors (`fak guard`) or context MMU correctness for speculative benchmark wins.
- Do not claim hardware performance from simulated in-memory test harnesses (`net.Pipe()`, Go channels).

# Plan
- [ ] 1. Amortize and fuse Metal command buffers via indirect command buffers (ICB) and pre-recorded execution graphs to eliminate the 39s host synchronization bottleneck (#9430 M4 / #8324).
- [ ] 2. Implement register-blocked ARM64 NEON and Metal GEMM tiles for Q8/int8 prefill to reach parity and lead over llama.cpp/MLX (#9430 M2 / #9230).
- [ ] 3. Enable zero-copy streamed weight residency from mmap directly into shared Metal buffers (`MTLResourceStorageModeShared`), capping peak startup memory under 22 GiB on 36GB laptops (#9073).
- [ ] 4. Deliver turnkey "one-touch" developer experience (`fak up`) with automated memory-fitting (`macfit`), zero manual flags, and instant local REPL / OpenAI-compatible endpoint.
- [ ] 5. Run full comparative benchmark sweep against llama.cpp, MLX, and Ollama on Apple Silicon M3/M4 hardware and publish immutable receipts.

# Scoreboard & Target Metrics
- Single-stream decode (Qwen3.8-27B Q4_K_M): > 7.5 tok/s (exceeding llama.cpp Metal).
- Single-stream prefill (Qwen3.8-27B, P=2048): > 40 tok/s.
- Multi-agent serving time (5-agent × 50-turn): > 5.0× less work via Radix KV reuse.
- Time to first token from clean shell: < 60s via `curl -fsSL https://fak.dev/install.sh | bash && fak up`.
