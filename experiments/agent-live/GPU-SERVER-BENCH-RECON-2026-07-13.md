# GPU-Server Benchmark Recon — 2026-07-13

**Objective:** pick up a top open benchmarking need and gather scrub-safe readiness data
for the 80 GB GPU-server class through the private control plane.

**Operator:** autonomous agent loop. **Machine class reached:** 8× 80 GB datacenter GPUs.

## Verified findings (read back through the bridge — not fabricated)

| Fact | Value | How verified |
|---|---|---|
| Machine class | 8× **80** GB datacenter GPUs | private readback, folded to generic hardware class |
| GPUs | **all idle** | scrubbed accelerator telemetry, confirmed twice |
| Inference serve | **none resident** | `curl :8080/v1/models` and `:30000/v1/models` → no listener |
| Collision risk | **none on GPUs right now** | GPUs at 0% — no other agent is mid-benchmark on the accelerators |

**Bottom line:** the 80 GB GPU server is powered up and completely idle — available for a benchmark —
but no model serve is currently running, so any serve-dependent benchmark (#3079
concurrency sweep, #3081 prefix-cache, #3060 livecodebench) needs a serve stood up first.

## Private-control boundary

Raw routing identifiers, channel defaults, session commands, transcripts, and recovery
details remain in `fak-private`. The only public control-plane datum retained here is the
folded status class: `WAIT_PRIVATE_RECOVERY` after the initial scrubbed readiness readback.

## Top open benchmarking needs (candidates picked up)

- **#4380** guard ablation
- **#3079** concurrency sweep (needs resident serve)
- **#3081** prefix-cache benchmark (needs resident serve)
- **#3060** livecodebench (needs resident serve)

## Ready-to-run plan (for a healthy control plane, or an on-box operator shell)

Since GPUs are idle, the highest-value non-colliding run is **#3079 concurrency sweep**:

1. Confirm servable model + framework inventory (`ls <model-dir>`, `command -v vllm
   llama-server`, `pip show vllm sglang`).
2. Stand up the serve as a **detached** job (`bg`) writing a durable log under
   `<artifact-dir>/`, so results survive transport flakiness.
3. Run a **bounded** concurrency sweep (e.g. concurrency ∈ {1,2,4,8,16,32}, fixed prompt
   set, short duration) → record tokens/s and TTFT per level to a JSON artifact on-box,
   then `pull` when the channel is quiet.
4. Tear the serve down (GPUs back to idle) to avoid leaving load for other agents.

## Honest limitation

A full serve+sweep was **not** completed this session: the private control plane folded to
`WAIT_PRIVATE_RECOVERY`, making the multi-step job unavailable for a trustworthy public
witness. No benchmark numbers are reported because none could be measured and read back
with confidence. The verified state above (idle GPU-server class, no serve) is the gathered
data; the plan is ready once the private readiness fold returns `READY_FOR_DEV_WORK`.
