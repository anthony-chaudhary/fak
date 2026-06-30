# Terminal-Bench 2.1 on the GPU server fleet — execution runbook

- Generated: `2026-06-27` (session-authored)
- Status: `PRECREDENTIAL` — `result_claim_allowed=false`. No Terminal-Bench number.
- Wire prerequisite: **SHIPPED** — the client-facing `/v1/responses` inbound route landed
  on origin/main as `945ea50e` (#925), so the fak arm can route a Responses agent through the
  gateway. This is no longer a blocker.
- Sibling artifacts: [`terminalbench-official-run-contract-20260626.json`](terminalbench-official-run-contract-20260626.json),
  [`../../docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md`](../../docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md).

This runbook records the WITNESSED GPU server fleet readiness and the exact path to the credentialed
raw-vs-fak run. It makes no result claim; it makes the run turnkey.

## Witnessed node readiness — a GPU server node ("GPU server", 2026-06-27 16:42Z)

Probed read-only via the private control bridge (`fak-private/tools/dgxsh.py`; the node's
control channel and hostname live in the gitignored private tooling, never the public repo):

| Gate | Status |
|---|---|
| Docker engine | ✅ `29.2.0` — **runs containers** (`docker run hello-world` OK) |
| `uv` / Python | ✅ `uv 0.11.21`, `Python 3.12.3` |
| GPUs / RAM | ✅ 8-GPU datacenter server, 886 GiB host RAM free |
| Harbor / `tb` CLI | ⚠️ install via `uv tool install terminal-bench` (no auth) |
| `go` | ❌ absent — build fak elsewhere + stage a linux binary (NOT needed at run time) |
| **`OPENAI_API_KEY`** | ❌ **absent — the operator gate** |

So GPU server is **infrastructure-ready**. The only run-time blocker is the credential.

## The win, and why it is operator-gated

The **win** is a single checked-in, hashed **raw-vs-fak compare artifact** over the *same*
official `terminal-bench-core==0.1.1` task ids, same model, same images/budget/concurrency/retry,
reporting benchmark-native solve **separately** from safe-resolve / blocked-dangerous-actions /
unnecessary-blocks / runtime / cost — **plus a gateway-traffic witness** (≥1 real structured model
HTTP request + ≥1 gateway inference-turn event on the fak arm). The gateway witness is producible
**only** by the live credentialed run, which is why the packet stays `BLOCKED_PRECREDENTIAL` until
it exists. Both arms spend paid tokens (raw → OpenAI directly; fak → OpenAI *through* the gateway),
so the run needs a real `OPENAI_API_KEY` **and explicit paid-spend authority** — the operator's to
grant, never the agent's.

## Do now, zero spend (each a real artifact, none needs a key or GPU)

Confirmed by a source-grounded investigation: the Terminal-Bench **oracle** agent runs the task's
checked-in reference solution and calls **no LLM** (`total_input_tokens=0`), so it needs no key.

1. **Stage the fak linux binary on GPU server + prove the route.** A clean `linux/amd64` fak binary
   built at `f918250d` (the shipped `/v1/responses` code; the origin tip `e9b7238f` fails to
   compile on an unrelated `internal/callavoid` peer break, so build at the wire tip) — 25.9 MB,
   `/v1/responses` string verified present.
   - Staging note: the 26 MB binary **cannot** cross the Slack bridge (≤4 KB PTY lines); it needs
     a real file channel (scp / a shared mount / `git pull` + on-node `go build` once go is
     installed). This is the one mechanical step the current Slack-only bridge cannot do for me.
   ```bash
   /tmp/fak serve --addr 127.0.0.1:8137 --policy examples/dev-agent-policy.json &
   curl -sS 127.0.0.1:8137/healthz     # expect {"ok":true,...,"planner":"mock"}
   ```
2. **Keyless `/v1/responses` wire smoke** (proves inbound wire + kernel adjudication on the run
   host, deterministically, free):
   ```bash
   curl -sS 127.0.0.1:8137/v1/responses -H 'Content-Type: application/json' -d '{
     "model":"mock","input":"Look up customer mia_li_3668.",
     "tools":[{"type":"function","name":"get_user_details","description":"Look up a customer.",
       "parameters":{"type":"object","properties":{"user_id":{"type":"string"}},"required":["user_id"]}}]}'
   ```
   PASS: `object=="response"`; an `output[]` `function_call` named `get_user_details`;
   `fak.adjudications[0].verdict.kind=="ALLOW"` (`admitted:true`). (`get_` is on the dev-agent
   policy allow-prefix; the mock planner's first move is that call.) Do NOT send `stream:true` (400).
   - **WITNESSED 2026-06-27** against the real binary built at `f918250d` (the shipped wire):
     `fak serve` came up `planner=mock`; this exact POST returned `object=response`,
     `status=completed`, a `function_call` named `get_user_details`, and
     `fak.adjudications[0] = get_user_details / ALLOW / admitted=true`. The wire + kernel
     adjudication work end to end on the real binary; only the upstream-model hop is left for
     the credentialed run. (Witnessed on the dev box, where the route logic is OS-identical to
     the staged linux binary; re-run on GPU server once the binary is staged to pin the host.)
3. **Oracle smoke** (no key; validates the Harbor harness end to end — build → run reference
   solution → tests → score):
   ```bash
   uv tool install terminal-bench && tb run --agent oracle --dataset terminal-bench-core --task-id hello-world --log-level info
   ```
   Check the single-task id resolves in the packaged dataset; on the newer Harbor CLI use
   `harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 1`.

Then regenerate the host preflight to record the now-green gates:
`go run ./cmd/terminalbench --preflight --probe-gateway --out <...>.json --md <...>.md`.

## Operator handoff — the one gate, then turnkey

Export a real paid `OPENAI_API_KEY` on GPU server and authorize spend. Then:

**Gateway up (fak arm's upstream):**
```powershell
$env:OPENAI_API_KEY='<paid key>'
./fak serve --addr 127.0.0.1:8080 --provider openai-responses `
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY `
  --policy examples/dev-agent-policy.json
```

**Raw arm** (hits OpenAI directly) and **fak arm** (identical but `--agent terminus-through-fak`
plus the two base-url exports that force the agent's traffic onto the gateway) — the literal
commands are frozen in [`terminalbench-official-run-contract-20260626.md`](terminalbench-official-run-contract-20260626.md).
The arms differ in exactly three tokens (`terminus`→`terminus-through-fak` + the two
`$env:OPENAI_*` exports); that sameness IS the comparability proof.

Substitute the real benchmark-native `terminal-bench-core==0.1.1` task ids for the
`<space-separated official task ids>` placeholder (the first `required_before_claim` item; the two
local fixture ids only pin subset shape). Fold both arms' outputs into the raw-vs-fak compare +
gateway witness, then — and only then — add the `BENCHMARK-AUTHORITY.md` row.
