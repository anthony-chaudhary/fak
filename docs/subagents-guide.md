---
title: "Subagents Guide: Parallel Multi-Agent Workflows with fak up"
description: "Guide to running parallel subagents with zero-cold-start prefix reuse using the fak up local model backend."
---

# Subagents Guide: Parallel Multi-Agent Workflows with `fak up`

`fak up` is a local inference engine that shares prompt caches across concurrent coding subagents.

> **In short:** Run parallel coding subagents locally without cold-start prefill lag. `fak up` serves an OpenAI-compatible endpoint where concurrent workers share warm prefix KV blocks for instant token generation.

```bash
# Quick start: launch the local backend in headless mock mode
go build -o fak ./cmd/fak
./fak up --mock --headless
```

---

## Why Subagents?

Sequential execution in a single context window degrades attention and inflates turn latency. Decomposing objectives into specialized workers (`worker`, `researcher`, `tester`) keeps context bounded.

Each subagent loads extensive project context, including instructions, rules, and tool schemas. Spawning four workers sequentially forces repeated prefill passes over 100,000 prompt tokens.

### In-Kernel Prefix Cache Reuse (RadixKV)

`fak` eliminates redundant prefill through in-kernel RadixKV tree reuse:
- Shared prefixes (rules, tool schemas, system instructions) process once into unified memory.
- Concurrent subagents bind directly to existing KV blocks with sub-millisecond TTFT.
- Co-batched decode delivers up to 4.1× speedup over tuned per-agent baselines (`BENCHMARK-AUTHORITY.md`).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Coordinator                             │
│       Decomposes task into parallel cohorts across disjoint files       │
└─────────────────────────────────────────────────────────────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Subagent 1: Dev │        │ Subagent 2: Test│        │ Subagent 3: QA  │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   fak up Backend (http://127.0.0.1:8080)                │
│   • Apple Silicon Metal GPU acceleration (metalgemm)                    │
│   • In-kernel RadixAttention prefix caching (zero cold-start prefill)   │
│   • Default-deny capability floor (protects against destructive ops)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install `fak`

`fak` is distributed as a single static Go binary with zero external runtime dependencies:

```bash
# Build from source:
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak

# Or install globally to your Go bin:
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Verify the installation:
```bash
./fak --help
```

---

## Step 2: Start the Engine with `fak up`

The `fak up` command probes unified memory, selects the optimal model tier, and starts serving:

```bash
./fak up
```

### Automatic Setup Steps
1. Unified Memory Probing (`macfit`): Probes physical hardware and reserves headroom to prevent memory swap.
2. Auto-Selects Model Tier: Selects the highest quality model fitting comfortably into RAM (e.g. Qwen 3.8 27B Q4_K_M).
3. Starts Serving: Boots an OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1` accelerated by Metal GPU compute kernels.
4. Interactive REPL: Provides an interactive chat terminal on stdin/stdout.

Terminal output on startup:
```text
[READY] fak up v1.0.0 running on http://127.0.0.1:8080
  • Model:                      27B (qwen3.8-27b-q4_k_m, quant: Q4_K_M) | Context: 65536 tokens | Headroom: 33.3%
  • OpenAI-compatible endpoint: http://127.0.0.1:8080/v1/chat/completions
  • Health check endpoint:      http://127.0.0.1:8080/healthz
you> Explain context caching in one line
fak> The Context MMU shares paged KV blocks across runs for sub-millisecond reuse.
     [telemetry: 76.1 tok/s | 64 tokens | 840ms | context: 128/65536]
```

### Headless Mode for Agent Sessions
Run `fak up` as a background service while using your terminal for agent harnesses:

```bash
./fak up --headless
```

### Offline Verification Mode
Verify the entire execution path using scripted test completions:

```bash
./fak up --mock --headless
```

---

## Step 3: Connect Your Agent Harness

`fak up` exposes an OpenAI-compatible `/v1` endpoint. You can connect any agent harness:

### Option A: OpenCode (`fak opencode`)
Launch OpenCode with automated environment wiring:
```bash
./fak opencode
```
`fak opencode` detects the local server at `127.0.0.1:8080`, sets `OPENAI_BASE_URL`, and starts live split-pane telemetry.

### Option B: Claude Code (`fak claude`)
Run Claude Code directly against the local server:
```bash
./fak claude
```

### Option C: OpenAI Codex (`fak codex --raw`)
Run Codex with automated provider configuration:
```bash
./fak codex --raw
```

### Option D: Pi (`fak pi`)
Launch the Pi terminal coding agent:
```bash
./fak pi
```

---

## Step 4: Run Your First Parallel Subagent Workflow

Once your harness is connected to `fak up`, run a multi-agent prompt that triggers parallel subagent fanout:

```text
"Using parallel subagents across disjoint packages:
 1. Subagent A (researcher): Inspect internal/gateway and summarize error recovery handling.
 2. Subagent B (worker): Add test coverage for Pi configuration in internal/projectassets.
 3. Subagent C (tester): Run package tests under internal/projectassets and cmd/fak.
Collect receipts and report final status."
```

### Execution Flow
1. Decomposition: The top-level coordinator creates three independent task packets.
2. Concurrent Dispatch: All three subagents launch simultaneously.
3. Prefix Hit: Shared system prompts, tool schemas, and repository rules hit warm in-kernel KV cache blocks.
4. Immediate Generation: Tokens generate immediately across all sessions via warm prefix cache hits.

### Live Telemetry Monitoring
While subagents run, `fak` displays throughput and cache efficiency:
```text
fak-turn ok prov=24.8k tok (88% of prompt) fak=0 tok cache=healthy_cache
[fak info] 4 active · 4 subagents · 4 in-flight · 88% x-agent reuse
```
- `4 subagents`: Number of concurrent subagents running.
- `88% x-agent reuse`: Fraction of prompt tokens resolved directly from shared prefix KV blocks.

---

## Step 5: Verify Calibrated Fan-Out Architecture [SIMULATED]

Validate calibrated multi-agent fanout behavior via architecture simulation:

```bash
./fak bench subagent --concurrency=4
```

*(Note: Runs calibrated AMD Strix Halo UMA simulation with logit parity validation; for physical Mac Metal execution, run `fak macbench many-agent`.)*

Sample output receipt:
```text
================================================================================
Strix Halo Subagent Fan-Out Benchmark Receipt (fak.benchmark.subagent_fanout/v1)
================================================================================
Engine:              fak-native
Architecture:        RDNA 3.5 / gfx1151 (UMA)
Target Device:       gfx1151 (AMD Radeon 8050S / Strix Halo)
MALL Cache Size:     32 MB
Sustainable DRAM BW: 210.00 GB/s (Peak: 273.06 GB/s)
Sustainable MALL BW: 800.00 GB/s

Benchmark Parameters:
  Scenario:          shared_prefix_forked
  Concurrency (B):   4 subagents
  Repetitions:       5 runs
  Prefix Tokens:     30000 tokens
  Gen Tokens/Sub:    64 tokens
  Execution Mode:    Calibrated Architecture Simulation
  Provenance:        simulation

Statistical Performance Summary:
  Mean Throughput:     18720.23 tokens/sec
  Mean DRAM Traffic:       3.54 GB/s
  Mean MALL Hit Rate:    96.79%
  Logit Cosine:        1.000000 (Threshold: >= 0.999900) [PASS]

Canonical Phase Timing Breakdown (Mean):
  host_dispatch:            0.185 ms (  1.36%)
  prefix_tree_lookup:       0.045 ms (  0.33%)
  kv_allocation:            0.065 ms (  0.48%)
  gpu_kernel:              12.994 ms ( 95.01%)
  token_sampling:           0.386 ms (  2.82%)

Verification Digest: sha256:9d843c068282c6da2fe39ce51202c484ceb326e3680247a74eb9a8d4f07d3c4a
```

---

## Core Rules for Multi-Agent Workflows

Follow these proven architectural invariants when running parallel subagents:

1. Tree-Disjoint Ownership:
   Assign each subagent a non-overlapping set of files (e.g. Worker 1 touches `internal/gateway/**`, Worker 2 touches `docs/**`) to prevent git merge conflicts.

2. Depth-0 Coordinator vs. Depth-1 Leaf Worker:
   - Coordinator (Depth 0): Plans work, assigns disjoint lanes, and integrates results.
   - Leaf Worker (Depth 1): Executes atomic tasks directly within its package boundary. Leaf workers must not spawn nested subagents, preventing recursion depth exhaustion (`#12028`).

3. Deterministic Witnesses ("Model proposes, test disposes"):
   Require every worker to execute a concrete witness command (`go test -v ./...` or `fak validate --mine`) before accepting changes; self-reports are never facts.

4. Default-Deny Capability Floor:
   `fak` automatically enforces capability rules across subagent calls, blocking destructive commands while permitting standard development tools.

---

## Troubleshooting & FAQ

| Symptom | Cause | Remedy |
|---|---|---|
| `port 8080 already in use` | Another server is bound to port 8080. | Bind `fak up` to another port: `./fak up --addr 127.0.0.1:9000 --headless`. |
| Out of memory or slow swap | Model tier exceeds available RAM. | Override model tier to a smaller variant: `./fak up --model 7B` or `./fak up --context 16384`. |
| Agent connects but hangs | Model weights still downloading or preheating. | Wait for `[READY]` banner or check liveness with `curl http://127.0.0.1:8080/healthz`. |
| Subagents editing same file | Overlapping task prompts. | Structure prompts with explicit disjoint file ownership for each worker. |

---

## Next Steps

- [Mac Local Models Guide](fak/mac-local-models.md): Review Apple Silicon unified memory sizing tables and hardware profiles.
- [Harness Integration Index](integrations/README.md): Connect your preferred agent harness.
- [Capability Floor & Policy Guide](fak/policy-guide.md): Customize security rules and tool allowlists.
