# DGX bridge / benchmark harness — handoff for the next agent

> **2026-06-23 update — discovery is FIXED and the live path is verified.** Three stacked
> discovery bugs are resolved: (1) the channel now resolves from `.env.slack.local`
> (`SLACK_CHANNEL=` / `FAK_SLACK_CHANNEL`) so no `-channel` flag is needed; (2) the hub's
> new compact `*Sessions:*` reply header is recognized; (3) the new
> `id status profile/mode | age | thread=...` line grammar parses. Verified end-to-end:
> `go run ./cmd/dgxbridge exec '...'` (no flags) auto-picks the newest running session and
> reads back real `nvidia-smi` from the lab DGX (8× A100-40GB). The readback / `!dump`
> concern below was DOWNSTREAM of discovery and did not reproduce on the live pipe-mode
> sessions. The real channel id + host live only in gitignored `.env.slack.local`.

Status at leave-off (2026-06-20 ~16:10Z): the **pure-Go harness is built, tested green,
and committed**; the **live run is blocked on the Slack control bridge's read path**,
which is the thing to fix next for 10× clarity/observability.

## What's done (all committed, `(fak dgx)` stamped)

| Component | Path | State |
|---|---|---|
| Slack control-bridge RPC client | `fak/internal/dgxbridge/` | green, table-tested |
| Bridge CLI (status/exec/readfile/pull/ship, `-probe`) | `fak/cmd/dgxbridge/` | builds, runs |
| OpenAI endpoint load generator | `fak/cmd/loadgen/` | green, smoke-verified end-to-end |
| Dual-track orchestrator (ours / sglang) | `fak/cmd/dgxbench/` | green, script-gen tested |
| Slack channel GC / lifecycle | `fak/cmd/slackgc/` | green; **applied** a 3h spam sweep (7 msgs, archived) |

Token lives in gitignored `.env.slack.local` (`source` it, then `export SLACK_BOT_TOKEN`).
GH follow-ups filed: #533 (sglang track live), #534 (gateway-overhead), #535 (model ladder).

## The blocker, root-caused (this is the 10× observability target)

The bridge runs in **PTY mode**. Two independent failures make `Exec` unreliable:

1. **Live stdout mirror is wedged** on Slack `msg_too_long` — the bridge edits one
   rolling message via `update_message`; PTY full-screen redraw overflows 4000 chars
   so ~93% of updates fail (seen in `!metrics`: `update_message_failed 223/240`).
   *Mitigation already in place:* read via `!dump` transcript file, not the mirror.

2. **`!dump` does not always upload a fresh transcript.** CONFIRMED by experiment on
   2026-06-20: on the dead persistent session it worked; on the newer
   `default: login shell` sessions, posting `!dump` produced **no new
   `transcript.jsonl`** in `files.list` (created-time unchanged across a dump), even
   though the session answered `!status` (probed LIVE). So `Exec` re-dumps forever and
   times out with `transcript present but completion sentinel not found yet`. The
   transcript `files.list` returns is then a *stale other-session* file
   (per-session path `/var/lib/slack-control/sessions/persistent-N/transcript.jsonl`).

**Net:** a session can be LIVE for control verbs yet have a broken/again result-readback.
`dgxbridge -probe` correctly distinguishes live-vs-stale *banners*, but does NOT yet
verify the **readback path** works.

## The 10× clarity/observability fixes (recommended, in priority order)

1. **Verify the readback path, not just liveness.** Add a `Bridge.SelfTest(ctx)` that
   runs `echo <nonce>` and confirms it round-trips through whatever read path is used,
   returning a typed reason on failure (`mirror_wedged`, `dump_no_upload`,
   `wrong_session_transcript`). `dgxbench`/`dgxbridge` should self-test before a run and
   print exactly which rung failed — instead of a generic timeout.

2. **Bind `!dump` output to the requesting session.** Today `newestTranscript` picks the
   newest channel transcript by `created`; it should match the transcript whose
   `bridge_start.thread_ts` equals the target thread (the JSONL carries it). Download a
   few candidates and pick the one for *this* thread. This kills the "stale other-session
   transcript" failure even when multiple sessions dump.

3. **Prefer a non-PTY (pipe) bridge.** `slack-helpers control.py` defaults `pty=False`.
   A bridge started `--no-pty`/pipe mode line-buffers output and would not wedge on
   `msg_too_long` at all — most of this complexity disappears. The cleanest fix is an
   **operator action**: restart the DGX control session in pipe mode. Document the exact
   bring-up command in `GLM-5.2-DGX-FAST-LOOP-2026-06-20.md` and have the orchestrator
   detect `mode: pty` (from `!status`) and warn loudly.

4. **A request-scoped result file, not a transcript scrape.** Strongest fix: have the
   remote command write results to `/tmp/<nonce>.json` and **upload that file directly**
   via the bridge's own upload (or a remote `slack`/`bench` upload), then download by
   exact name. No transcript, no PTY noise, no msg_too_long, no session-matching. The
   `Ship`/`ReadFile` base64 path is a stopgap; a real file upload is cleaner for the
   MATRIX.json/GATE.json artifacts the bench produces anyway.

5. **Structured run log.** `dgxbench` should emit a `RUN_LOG.jsonl` (each step: build,
   serve-ready, loadgen, with timestamps + the bridge round-trip latency) so a run is
   debuggable from one artifact instead of scraping stderr. This is what would have made
   today's debugging 10 minutes instead of an hour.

## How to actually get the smoke running (once readback is fixed)

```sh
source .env.slack.local; export SLACK_BOT_TOKEN
go -C fak run ./cmd/dgxbridge -dgx-host dgx-a100.example.lab -probe status   # find a LIVE session
go -C fak run ./cmd/dgxbench -track ours -run-id smollm2-fak-dgx-<date>      # priority: fak's own engine
# pull artifacts:
go -C fak run ./cmd/dgxbridge -thread-ts <ts> pull /srv/fleet/fak/experiments/dgx/<run>/MATRIX.json ./MATRIX.json
```

Known DGX facts: `dgx-a100.example.lab`, 8× A100-40GB **all idle**; base `python3` has
no torch/sglang/vllm (serving env is elsewhere — the `bench` command is installed for the
sglang track); in-repo model `fak/experiments/model-baseline/gguf/SmolLM2-135M-Instruct-Q8_0.gguf`
exists at `/srv/fleet` but has **no tokenizer.json beside it** (the `ours` track fetches/ships one);
fak CUDA build needs **arch `sm_80`** for A100 (build_cuda.sh defaults to `sm_89`/Ada).

A peer session is independently building a **Python** `tools/dgx_ladder_runner.py` for the
same goal — coordinate to avoid duplicate/conflicting DGX runs (and PTY-thread collisions).

See memory `dgx-pure-go-bench-harness` for the condensed version.
