# `lab-dgx3` Benchmark Recon — 2026-07-13

**Objective:** pick up a top open benchmarking need and gather data on `lab-dgx3` via the
lab-connect (Slack control-hub) bridge.

**Operator:** autonomous agent loop. **Host reached:** `lab-dgx3`.

## Verified findings (read back through the bridge — not fabricated)

| Fact | Value | How verified |
|---|---|---|
| Host identity | `lab-dgx3` | `hostname` via `run`, read back OK (x2) |
| GPUs | 8× **80** GB GPUs, **all idle** | `nvidia-smi --query-gpu=memory.used,utilization.gpu` → `MEM=[0;0;0;0;0;0;0;0] UTIL=[0;0;0;0;0;0;0;0]` (x2) |
| Inference serve | **none resident** | `curl :8080/v1/models` and `:30000/v1/models` → no listener |
| Collision risk | **none on GPUs right now** | GPUs at 0% — no other agent is mid-benchmark on the accelerators |

**Bottom line:** `lab-dgx3` is powered up and completely idle — available for a benchmark —
but no model serve is currently running, so any serve-dependent benchmark (#3079
concurrency sweep, #3081 prefix-cache, #3060 livecodebench) needs a serve stood up first.

## Operational fixes / knowledge (reusable)

1. **Channel bug:** the default `SLACK_CHANNEL` env points at **`lab-dgx2`** (`<lab-dgx2-channel-id>`).
   `lab-dgx3` is a different channel — `FAK_DGX3_CHANNEL` (`<lab-dgx3-channel-id>`). Data-gathering on
   `lab-dgx3` must target `-channel <lab-dgx3-channel-id>` explicitly. (Earlier attempts silently ran
   against `lab-dgx2` because of this default.)

2. **Reliable-readback recipe** (the control channel is shared by 10+ agents and is
   congested):
   - Spawn a **fresh private session** (`!new default`) and drive **that** session id —
     shared/warm sessions are PTY-contended (another agent's command occupies the shell,
     so yours never echoes → `sentinel_missing`).
   - Use **`-transcript`** readback, not the default file/channel-history readback. The
     default reads `conversations.history` (channel, limit 60); on a busy channel your
     sentinel is flooded out of the window. `-transcript` reads your **quiet private
     thread** instead.
   - Keep commands **tiny and fast**, one short line of stdout. Heavy commands (`find`,
     `pip list`) spend enough on-box time that the readback window rots.
   - **Back off** — aggressive retry loops burn the bot token's Slack read rate-limit
     (Tier 3 ≈ 50/min); once throttled, reads return `context deadline exceeded`. Sparse
     deliberate calls beat tight loops.
   - New **persistent** (`tmux bash -li`) sessions were coming up **wedged** (login shell
     never reached a prompt) under load — prefer the `default` pty profile, which is what
     answered.

## Top open benchmarking needs (candidates picked up)

- **#4380** guard ablation
- **#3079** concurrency sweep (needs resident serve)
- **#3081** prefix-cache benchmark (needs resident serve)
- **#3060** livecodebench (needs resident serve)

## Ready-to-run plan (for a healthy control plane, or an on-box operator shell)

Since GPUs are idle, the highest-value non-colliding run is **#3079 concurrency sweep**:

1. Confirm servable model + framework inventory (`ls /work/models`, `command -v vllm
   llama-server`, `pip show vllm sglang`).
2. Stand up the serve as a **detached** job (`bg`) writing a durable log under
   `/work/agent-probes/`, so results survive transport flakiness.
3. Run a **bounded** concurrency sweep (e.g. concurrency ∈ {1,2,4,8,16,32}, fixed prompt
   set, short duration) → record tokens/s and TTFT per level to a JSON artifact on-box,
   then `pull` when the channel is quiet.
4. Tear the serve down (GPUs back to idle) to avoid leaving load for other agents.

## Honest limitation

A full serve+sweep was **not** completed this session: the Slack-relayed control-plane
readback degraded (self-inflicted rate-limiting + channel congestion), making the
multi-step long-running job unreliable to drive and verify. No benchmark numbers are
reported because none could be measured and read back with confidence. The verified
state above (idle `lab-dgx3`, no serve) is the gathered data; the plan above is ready to
execute once the control plane is quiet or from a direct on-box shell.
