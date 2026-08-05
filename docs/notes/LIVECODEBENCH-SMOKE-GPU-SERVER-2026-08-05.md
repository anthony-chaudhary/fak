# LiveCodeBench smoke against an already-running GLM-5.2 serve, over the control bridge (2026-08-05)

**What this note is.** The durable recipe for exercising the LiveCodeBench generation arm
against a model that is *already serving* on a remote GPU server we cannot SSH into, driving
the whole thing over the Slack control bridge and bringing the artifacts back **as files**.
It records one witnessed run plus the three transport traps that cost the most time, so the
next run is a script invocation rather than a rediscovery.

The shape is deliberate: **a named verb, launched by a script, answered by files.** Not a
transcript scrape. `cmd/livecodebench` is the verb; `tools/livecodebench_smoke_gpu_server.sh`
is the script; `RESULT.txt` + `raw-report.json` + a `.done` sentinel holding the rc are the
files.

## The verb and the script

`tools/livecodebench_smoke_gpu_server.sh` assumes the serve is **already healthy and never
starts one** — that is its whole contract, and the counterpart to
`tools/glm52_load_witness.sh`, which stands a model up and measures the load. It probes
`$LCB_ENDPOINT/models`, resolves the model id from the first advertised `id` if
`LCB_MODEL` is unset, builds `cmd/livecodebench` **from the checkout under test**, runs the
`raw` arm over the committed offline suite, summarizes the generations, and writes the rc to
`$LCB_OUT.done`.

`raw` is the unadjudicated arm — straight to the serving endpoint, no gateway in the path.
That is the correct arm for smoking a bare model serve; the `fak` arm exists to measure what
the adjudicated gateway adds and would confound this check.

Distinct rc codes, so a detached run is diagnosable from the sentinel alone: `0` pass,
`90` no repo root, `91` build failed, `92` endpoint unreachable, `93` no model id,
`94` runner failed, `95` every generation empty.

The default suite is the committed `internal/livecodebench/testdata/suite_release_v2_sample.json`,
so a smoke never depends on the HuggingFace datasets-server being reachable from the box.

**Honesty fence.** `livecodebench raw` **grades nothing**. It emits generations plus a shape
check, and the report it writes carries `result_claim_allowed=false`. A pass@1 number still
requires the official `lcb_runner` evaluator. This note reports a *generation* smoke and
nothing more.

## The witnessed run (2026-08-05)

Remote GPU server, 8 accelerators, Go 1.26.4 at `/usr/local/go` (not on the default `PATH` —
the script exports it), python3 3.12.3. The serve was a `llama-server` GGUF host of GLM-5.2
listening on `:8000`, `n_ctx` 8192, aliased `glm-5.2`.

```
LCB_SMOKE_START 2026-08-05T17:04:23Z
ENDPOINT_OK models={"models":[{"name":"glm-5.2",...
LCB_MODEL=glm-5.2
BUILD_OK 9s
livecodebench raw: 3 problem(s) x n=1 (model glm-5.2), 29 cached prompt tokens
RAW_OK gen_s=67
PROBLEMS=3 SAMPLES=3 NONEMPTY=2 WITH_CODE=2
TOKENS prompt=75 completion=1525 cached=29 retries=0
  lcb-sample-001   chars=1158
  lcb-sample-002   chars=0
  lcb-sample-003   chars=739
SMOKE_PASS nonempty=2 gen_s=67
CLAIM result_claim_allowed=false
LCB_SMOKE_DONE rc=0
```

Both artifacts came back byte-verified: `raw-report.json` 2849 bytes
`md5=0a32a93e8de9b1d82e8528e816ec477f`, `RESULT.txt` 1412 bytes
`md5=bed03a81a25e6a46f05446f2f136d7b1`.

**2/3, and the honest reading of the third.** Sample-002 came back with **zero characters**.
That is not a dead endpoint and not a runner bug. A direct probe of the same prompt returned
`finish_reason=length`, `completion_tokens=512`, `content_len=0`, `reasoning_len=1681` — a
reasoning model spends its budget on the private chain **first**, and a budget that runs out
mid-think returns `length` with an empty `content`. The default is now `LCB_MAX_TOKENS=2048`
and the summarizer prints an `EMPTY_HINT` naming the current budget whenever any sample is
empty, so the next reader is not sent hunting a phantom serve failure.

**The 2048 default is witnessed, not assumed.** Re-run against the same live serve from the
committed script (`HEAD=176f948`), with only the default supplying the budget:

```
LCB_MAX_TOKENS=2048
PROBLEMS=3 SAMPLES=3 NONEMPTY=3 WITH_CODE=3
TOKENS prompt=75 completion=2142 cached=30 retries=0
  lcb-sample-001   chars=1548
  lcb-sample-002   chars=1067
  lcb-sample-003   chars=739
SMOKE_PASS nonempty=3 gen_s=94
```

`RESULT.txt` 1500 bytes `md5=9fae7c6d087473ff884b0d988d0025f8`. The 512-budget empty is gone;
the cost is 67s → 94s for three problems. Note `completion=2142` across three samples — the
budget is per-request headroom for the think chain, not a per-request spend.

Sample-003 answered *"It looks like you didn't include the array in your message"* — correct
behavior. The committed sample suite carries one-sentence **stub** prompts, not real
LiveCodeBench statements. It is a transport-and-shape fixture, not a difficulty fixture. Do
not read `WITH_CODE` from this suite as a capability signal.

## Three transport traps (the expensive part)

These are properties of the bridge, not of this benchmark. They will bite any remote-drive
task in this context.

**1. The hub ignores `!send-file` uploads.** The bridge's default command transport uploads
the framed wrapper as a text file with a `!send-file <session>` comment. The hub does not act
on it, so every command silently never reaches a shell and every run dies `sentinel_missing`.
The symptom reads exactly like a wedged shell, which is the trap — `!tail <session> <n>`
proved the shell was alive and echoing. **Fix: pass `-message-commands`,** which posts the
wrapper as a plain message instead.

**2. Slack linkifies bare URLs, and bash reads the result as a redirection.** With
`-message-commands` on, a command containing `http://127.0.0.1:8000/v1/models` arrives at the
shell as `<http://127.0.0.1:8000/v1/models>` and dies on `syntax error near unexpected token
'>>'`. Plain `>` and `&` survive fine; it is specifically the autolinker. **Fix: never send
command text. Send a base64 carrier** — `echo <b64> | base64 -d | bash` — which is
alphanumeric and immune to every content rewrite the transport applies. Cap it around 3500
b64 characters per message and use the bridge's own `push` for anything larger.

**3. A single-readback `pull` truncates silently past roughly 1.4 KB.** The pull path
base64s the whole file into one transcript readback; past that size the readback window drops
the **head** of the payload, and the decoder strips non-base64 bytes and re-pads, so it
returns the surviving tail **without an error**. Witnessed: a 2849-byte JSON came back as
1349 bytes of valid-looking JSON with its opening brace gone, and a 1412-byte text file came
back with line 1 reduced to `ab`. **Fix: split remotely (`split -b 900 -d -a 3`), pull each
part, reassemble locally, and compare against the remote `md5sum`.** A pull that does not
verify a digest is not a pull.

A corollary worth stating plainly: the hub's control verbs (`!tail`, `!interrupt`, `!reset`,
`!sessions`, `!profiles`, `!send`) have **no CLI binding**. When a session looks wedged, the
diagnosis lives in those verbs, posted directly to the channel — that is what separates "the
hub is down", "the shell is wedged", and "the command never arrived", which otherwise present
identically.

## Running it

Ship the script to the box, launch it detached with the environment set, poll for the
sentinel, then pull the two artifacts with digest verification. The knobs that actually
matter for a first run:

```
LCB_ENDPOINT=http://127.0.0.1:8000/v1
LCB_MODEL=glm-5.2
LCB_N=1
LCB_TEMPERATURE=0
LCB_MAX_TOKENS=2048     # a reasoning model needs headroom for the think chain (trap above)
LCB_CONCURRENCY=1       # a host-offloaded 753B decode is serial-slow
LCB_OUTDIR=/tmp/fakgpu/lcb-smoke
```

Prefer `poll` over a blocking `wait` for the detached job: a long `wait` holds a
`conversations.history` read open and dies on the workspace rate limit
(`HUB_UNRESPONSIVE: hub_timeout`) even though the job itself is fine.

## What is `not yet`

- **No pass@1.** Generation only, per the fence above. Grading needs `lcb_runner`.
- **No real problem statements.** The committed suite is a 3-problem stub sample; a
  difficulty-meaningful run needs a fetched `release_v2` suite.
- **The chunked-pull and base64-carrier wrappers live in scratch, not in the repo.** They are
  described here precisely enough to rebuild; folding them into the bridge client is the
  obvious follow-on.
