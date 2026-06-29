// usage.go holds the `fak` top-level help text, extracted from main.go so the
// verb-dispatch monolith stays under the steerability god-file line (#984).
// Pure code motion: usage() is unchanged.
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// usage prints the full `fak` help banner. The verb list is long enough to be a
// god-function on its own, so the body is split into three contiguous raw-string
// sections (core verbs / ops verbs / scorecards + aliases) printed back-to-back â€”
// the output is byte-identical to the single block it replaced.
func usage() {
	fmt.Fprintf(os.Stderr, "fak - the Fused Agent Kernel (v%s)\n\n", appversion.Current())
	usageCoreVerbs()
	usageOpsVerbs()
	usageScorecardVerbs()
}

func usageCoreVerbs() {
	fmt.Fprint(os.Stderr, `usage:
  fak run       --trace FILE [--engine inkernel] [--vdso=true] [--policy FILE]
  fak commit    --path P [--path P ...] (-m STR | -F FILE/-) [--push] [--trunk B] [--no-signoff] [--review-model M] [--json]
                (the SAFE SHARED-TRUNK COMMIT: commit by EXPLICIT pathspec on a
                 multi-session trunk and refuse to report success unless ONLY those
                 paths landed. Lock-guards the commit, writes the message to a file
                 (never -m, so an em-dash/multiline subject can't misparse as a
                 pathspec), runs the real hooks (never --no-verify), then asserts the
                 committed file set == the requested paths. If a peer raced extra files
                 in -> PATHSPEC_RACE, commit left intact, never pushed/force-pushed.
                 With --review-model, a scout model reviews the diff before commit:
                 refute -> REVIEW_REFUTED; unreachable reviewer -> logged fail-open.
                 Refuses OFF_TRUNK / MERGE_IN_PROGRESS / NOTHING_STAGED up front.
                 Exit 0 ok, 2 usage, 3 a pre-commit refusal, 1 a raced/refused commit)
  fak sweep     [--dir DIR] [--json] | --apply --lane L -m "SUBJECT" [--path P ...] [--push]
                (DRIVE A DIRTY MULTI-SESSION TREE TOWARD ZERO: the layer above fak commit.
                 Default mode REPORTS the working tree grouped by lane â€” every stampable
                 change under the (fak <leaf>) trailer its paths imply (the SAME dos.toml
                 path->lane engine the pre-commit lint binds to), plus the residual a sweep
                 must NOT silently commit: stray scratch/log junk, and root-level files with
                 no inferable lane. It never invents a subject â€” with --apply --lane L -m S it
                 commits exactly lane L's dirty paths (narrow with --path) through the safe
                 commit path (appends the (fak L) stamp, pre-lints, refuses OFF_TRUNK / a
                 pathspec race / an off-lane stamp). --json feeds a drive-to-zero loop.
                 Exit 0 ok, 2 usage, 3 a pre-commit refusal, 1 a raced/failed commit)
  fak affected  [--base REF] [--file P] [--budget DUR] [--report FILE]
                [--list] [--json] [--short] [--run RE] [--] [go test args]
                (the FAST INNER LOOP: run go test for only the packages your
                 working-tree change can affect  -  the changed packages plus every
                 package that (transitively, test imports included) imports one  -  not
                 the whole go test ./... suite. Seconds, not minutes, for a one-leaf edit,
                 so you verify the REAL oracle (not -short) on every change. --list prints
                 the selected packages; --base REF selects everything changed since a ref;
                 --budget reds on a latency regression after tests pass; --report writes the
                 measured verify-loop JSON; --file supplies a representative changed path.
                 make ci still runs the full suite as the authoritative gate.)
  fak hooks     pre-commit [--root DIR] [--json] | commit-msg <msgfile> [--root DIR]
                (the COMMIT-BOUNDARY GATES in ONE process â€” the Go port of the
                 tools/check_*.py git-hook checkers. pre-commit runs all 7 staged-diff
                 gates (PUBLIC_LEAK/SECRET_SHAPE/DOC_PLACEMENT/BROKEN_LINK/FILE_ADMISSION/
                 INDEX_SYNC/PROVENANCE_LABEL) over ONE staged-diff read instead of spawning
                 7 Python interpreters (~10.7s -> ~0.3s measured). Honors each gate
                 FLEET_<NAME>_GUARD block|warn|off + one-shot escape env. Exit 0 clean,
                 1 a block gate fired, 2 could-not-run (the shell hook then falls back to
                 the Python checkers â€” fail-open). The shell hooks prefer this.)
  fak hygiene   [--root DIR] [--json] [--gates A,B,...]
                (the WHOLE-TREE hygiene gates in ONE process â€” the --audit-tree twin of
                 fak hooks. Runs the ported make-hygiene checkers (DOC_PLACEMENT/BROKEN_LINK/
                 FILE_ADMISSION/SECRET_SHAPE/PROVENANCE_LABEL/INDEX_SYNC) over ONE
                 git-ls-files read instead of spawning a Python interpreter per checker.
                 Exit 0 clean, 1 a gate fired, 2 could-not-run (make/CI then falls back to
                 the Python path â€” fail-open). make hygiene / make index-sync prefer this.)
  fak preflight --tool NAME --args JSON [--policy FILE]
  fak egress    check (--url URL | --command CMD | --host HOST | --tool T --args JSON)
                (prove the NETWORK-EGRESS floor on one destination: the cloud-metadata /
                 link-local SSRF class (169.254.169.254, metadata.google.internal, ...) the
                 kernel refuses so fak guard blocks instance-credential theft on a random
                 VM with no human in the loop. Runs the real adjudication fold; exit 0
                 allowed, 1 blocked, 2 usage. See examples/remote-vm-guard/)
  fak attest    --policy FILE [--probes FILE] [--out FILE] [--json] [--quiet]
                 (the COMPLIANCE ATTESTATION GENERATOR: prove the capability floor
                  from preflight. Runs the real adjudication fold over a probe set
                  and emits a re-checkable attestation. Default probes are DERIVED
                  from the manifest  -  each deny must be DENIED with its cited reason,
                  each allow/allow_prefix ALLOWED, and an unnamed tool DENIED
                  DEFAULT_DENY. --probes FILE attests arg-value cases. Exit 0 if the
                  floor is PROVEN, 1 if any probe drifts, 2 on usage error)
  fak model     load <hf://owner/repo[@rev]/file>
                (resolve an hf:// URI to a locally cached file path: Hub download with
                 HF_TOKEN auth and SHA256 verification against the Hub LFS oid. The
                 cached path is printed on stdout; --gguf and the loaders accept it)
  fak bench     --suite NAME [--out report.json] [--baseline-n 30]
                (transport A/B: in-process adjudication p50 vs spawned-hook p50)
  fak benchmarks list [--offline] | describe <name> | run <name>
                (THE INDEX of every benchmark fak ships -- start with
                 'fak benchmarks list --offline' for the zero-asset set)
  fak ablate    --sweep vdso[,...] [--suite NAME] [--baseline all-off] [--out FILE] [--json]
                (self-ablation: replay one frozen trace under N feature configs;
                 one row per arm, deltas off the kernel counters, same-trace guard)
  fak turntax   --suite NAME [--out turntax-report.json]
                [--prompt-tokens N --completion-tokens N --turn-latency-ms F]
                (TURN-TAX A/B: the extra error-code MODEL TURN a SOTA loop fires  - 
                 malformed args, duplicate read, poison  -  vs fak's 1-shot. Replays a
                 class-labeled trace through the real kernel, prices the turns it
                 deletes per lever, and keeps the safety floor on its own axis)
  fak agent     [--task STR] [--provider openai|anthropic|gemini|xai]
                [--base-url URL --model M --api-key-env VAR | --offline]
                [--max-turns N] [--out agent-report.json] [--policy FILE]   (LIVE turn-count A/B)
  fak policy    --dump | --check FILE
                (--dump writes the built-in DefaultPolicy as a manifest you edit;
                 --check validates a manifest and prints the floor it admits. The
                 capability floor  -  WHICH tools may be called  -  is a deployable
                 file, not a Go edit: dump -> edit -> --check -> --policy)
  fak route     [--manifest FILE] [--aspect request|tool_call|query|state|step|scout]
                [--tool NAME --prompt-tokens N --latency interactive|batch --complexity low|medium|high --labels k=v,...]
                [--simulate "<out>[@score],..."] [--json] | --dump | --check FILE
                (the MODEL-ROUTING oracle  -  first-class per-aspect + ensemble model
                 routing. For one classified SUBJECT (an aspect of a request: the
                 whole request, a tool call, a sub-query, a state, a reasoning step)
                 print which MODEL  -  or which ENSEMBLE of models + reduction
                 (first|vote|best_of|all_reduce|concat)  -  the policy selects. The
                 routing policy is a deployable JSON manifest: --dump -> edit ->
                 --check -> --manifest, mirroring 'fak policy'. --simulate folds
                 stand-in member outputs through the plan's reduction so the
                 ensemble half runs end to end with no model in the loop)
  fak routebench [--corpus FILE] [--routed FILE] [--single FILE] [--frontier MODEL]
                 [--prices ...] [--latencies ...] [--json] | --dump-corpus
                 (the OFFLINE ROUTING BENCHMARK: run a corpus of recorded cases
                  through TWO manifests  -  a per-aspect + ensemble policy vs a
                  single-model baseline (the SOTA shape)  -  and print the delta on
                  COST / LATENCY / QUALITY. Each case carries the stand-in OUTPUT
                  every candidate model produces (like 'fak route --simulate'),
                  so it reuses the pure Route + Combine halves and is deterministic
                  end to end  -  no key, no GPU, no network. Default: the built-in
                  8-case demo corpus + DefaultManifest vs a one-frontier-model
                  baseline. Every figure is a ROUGH lens, never a bill or SLA)
  fak accounts  <list|resolve|discover|validate> [--registry FILE] [--home DIR] [--json]
                (the CONFIG-HOME REGISTRY: every CLAUDE_CONFIG_DIR seat with its
                 DISK-TRUE identity (a dir named for one account but logged into
                 another is FLAGGED), plus tombstone -> auto-rehome so anything
                 pinned to a retired seat resolves to a live one. resolve <name>
                 prints the config dir to use, following the rehome chain)
  fak lint      [--json] [--strict] [--kernel-only]
                (the STATIC TOOL LINTER: the definition-time dual of the kernel's
                 call-time re-checks. Reports a dead cache hint, an unreachable pure
                 registration, a canned answer for a write-shaped tool, or a schema
                 the model is shown but the kernel never enforces  -  once, instead of
                 the runtime silently papering over it every call. Exit 1 on an
                 error finding, or on any finding with --strict)
  fak codelint  [--json] [--errors-only] [--list] PATH...
                (the LANGUAGE-SERVER-PACK code linter: route each file to the pack
                 that owns its extension and report parse/compile errors  -  the
                 write-time check the kernel runs over CODE the agent produces
                 (Go/JSON in-process, Python/CUDA via their toolchains, degrading
                 to no-opinion where a checker is absent). The same Lint the
                 SWE-bench fleet runs on every agent file write. Exit 1 on an error)
  fak answer-shape [--text - | --file PATH | --text STR] [--max-repeat 0.5] [--max-chars N] [--ngram 3] [--json]
                (the DEGENERATION/VERBOSITY WITNESS: judge the SHAPE of a candidate
                 answer or tool result  -  how repetitive (looping) and how long
                 (runaway) it is  -  against your thresholds. The graded consumer dual
                 of the context-MMU's write-time repeat-admit rung. Reads stdin on
                 "-" or no source. Exit 1 when degenerate, so it gates a pipeline)
  fak doctor    [--text - | --file PATH | --text STR] [--max-repeat 0.5] [--max-chars N] [--ngram 3] [--json]
                (the OPERATOR DIAGNOSTIC: run the answer-shape witness over a text and
                 cross-check the real kernel admit verdict (would the context-MMU
                 quarantine it?), then RECOMMEND what to do about each finding. Exit 1
                 on any finding. The fak analogue of 'dos doctor')
  fak index     lane <path>... | leaf [<query>] | docs <query>   [--json] [--limit N] [--root DIR]
                (the QUERYABLE SELF-INDEX: query fak's own dev facts instead of
                 re-surveying prose. 'lane' resolves which lane/leaf owns a path
                 (+ the (fak <leaf>) commit stamp it implies); 'leaf' searches the
                 dos.toml lane taxonomy by name/tree/description; 'docs' ranks the
                 curated INDEX.md doc map by relevance. A VIEW over dos.toml + INDEX.md,
                 never a competing source of truth. --json for tooling/MCP)
`)
}

func usageOpsVerbs() {
	fmt.Fprint(os.Stderr, `  fak recall    [--dir DIR] [--out recall-report.json] [--query STR]
                (persist a finished session as a core dump, reload it in a FRESH
                 store, and demonstrate the quarantine surviving the boundary)
  fak snapshot  kinds | demo | info | dump-fleet | restore-fleet
                (DUMP/RESTORE any primitive on the loops ladder  -  a turn, a tool, a
                 session, a fleet, an RSI loop  -  to a portable, sha256-integrity bundle.
                 'kinds' lists the ladder; 'demo' is the offline witness: a SESSION
                 image dumped on laptop/model-A, packed to one .faksession, resumed on
                 model-B (drive re-attached, content byte-identical, the recall
                 quarantine SEALED across the offload, migration logged, integrity
                 fail-closed) + a FLEET of drive states dumped and restored verbatim;
                 'info --file F' verifies + prints a .snap envelope or a session image;
                 'dump-fleet --addr URL --out F' offloads a LIVE fleet's drive state from
                 a running gateway and 'restore-fleet --addr URL --file F' re-establishes
                 it on another. The session image is model-agnostic  -  logical content
                 only, no KV cache or token ids  -  so a resume re-prefills on any model)
  fak traj      similar | cluster | score | gc | export  --corpus <turns.jsonl> [...]
                (the TRAJECTORY-CORPUS toolkit over the JSONL a trajectory.Recorder
                 exports: 'similar' finds the k past queries most like a --query by
                 simhash cosine; 'cluster' groups near-duplicate queries; 'score' runs
                 the registered scorers worst-first; 'gc' PROPOSES prune candidates
                 (never deletes); 'export' re-emits normalized JSONL. fak ships the data
                 plane + similarity primitive + scorer seam; a gardening skill builds the
                 prune policy on top  -  see docs/observability/trajectory.md)
  fak dream     [--dir DIR] [--out-dir DIR] [--out dream-report.json]
                (offline "sleep" pass over a core image: re-screen, pre-seal
                 refuted witnesses, repair descriptors, surface duplicate aliases,
                 and write a pruned cleaned image)
  fak memory    drivers | explain | run  [--driver NAME] [--query-file PLAN.json]
                [--intent STR] [--k N] [--budget BYTES] [--dir IMAGE] [--apply]
                (the MEMORY-OPERATION ALGEBRA  -  build SQL, not a specific query: an
                 agent authors its OWN render/clean/compact/dream strategy as a
                 composable Op pipeline (scan|filter|rank|limit|budget |
                 render|tombstone|consolidate|reclassify|prune) instead of the kernel
                 hardcoding one. 'drivers' lists the built-in strategies; 'explain'
                 shows a plan without running it; 'run' executes it (mutations PROPOSED
                 unless --apply). Default backend is the in-memory demo corpus; --dir
                 runs over a recall core image)
  fak debug     [--session JSONL] [--dir DIR] [--cmd report|html|info|bt|x|ws|grep|tombstone|context-query|context-diff]
                [--query STR] [--step N] [--grep PAT] [--k N] [--reason STR]
                [--requested-by STR] [--out cdb-report.json|cdb-report.html]
                (the CONTEXT DEBUGGER: attach to a finished session as a core dump and
                 demand-page only the working set a question touches. --session ingests
                 a REAL Claude Code transcript; default is the committed fixture.
                 --cmd html emits a self-contained static HTML inspection report  -  the
                 shareable artifact a teammate opens in a browser)
  fak session   ls | status <id> | stop <id> | pause <id> | resume <id> | throttle <id> |
                run <id> <state> | budget <id> [--turns N] [--tokens N] [--context-tokens N] |
                pace <id> [--max-tokens N] |
                priority <id> <N>   [--addr URL] [--key K] [--if-rev N] [--json]
                (the OPERATOR control surface: read a served session's live DRIVE state
                 and CANCEL or UPDATE it in flight, over the /v1/fak/session(s) routes)
  fak resume    plan [--resident-tokens N] [--idle-seconds S] [--ttl 5m|1h] [--horizon N]
                [--shed-budget N] [--seed-tokens N] [--input-price F] [--output-price F]
                [--image DIR] [--json]
                (the DETERMINISTIC RESUME-CACHE decision: "I am resuming a 250k-token
                 session  -  what happens to the prompt cache, and what should I do?"
                 Projects the cache POSTURE (cold if the session was idle past the cache
                 TTL, warm if not), prices RESUME_FULL / CUT / RESET, and recommends a
                 cut-by-default re-entry. Pure: same facts in, same priced verdict out.
                 --image grounds it on a real portable session image)
  fak ps        [--json] [--watch] [--interval D] [--frames N] [--addr URL] [--key K]
  fak top       (= fak ps --watch)
                (the READ-ONLY PROCESS TABLE: one aligned row per live session folded
                 from GET /v1/fak/sessions; --watch is the top mode. Issues no control
                 verb - control a session with fak session)
  fak signal    <id> pause | resume | stop [--reason R] | steer --text "..."
                (JOB CONTROL for a running session - the OS process-model names over the
                 control plane; steer sends INPUT to a running agent, taken at its next
                 turn boundary. Answers Claude Code #21419, the SIGCONT+stdin gap)
  fak task      sample [--json] [--done N --total N --unit UNIT]
                (the PROCESS-LOCAL TASK MANAGER snapshot: current hardware/runtime
                 sample plus task/step/concept progress and ETA when progress is known)
  fak c         [--dry-run] [--account ACCT] [--model MODEL] [--policy FILE] ...
                (shorthand for 'fak console agent': the ACCOUNT LAUNCHER that
                 starts a fak-guard-wrapped interactive Claude Code session.
                 --debug-stats is ON by default: one compact per-turn line leading
                 with a verdict (ok/warming/degraded/cold) + the NET write-premium-
                 aware token saving, then cache health + compaction. Token-saving
                 defaults â€” compact-history-
                 budget and elide-result-bytes â€” are passed explicitly so they
                 appear in --dry-run output. 'fak c' is the canonical shortcut)
  fak console   issues [--epic N] [--issues-json FILE] [--json] |
                loops [--ledger FILE] [--json] | sessions [--sessions-json FILE] [--json] |
                garden [--garden-json FILE] [--json] [--check] |
                guard --guard-json FILE [--json] | overview [--json]
                (the NATIVE TERMINAL CONTROL PANE spine: ranked GitHub issue
                 lanes, durable loop-ledger lanes, and live session DRIVE lanes,
                 plus garden health, guard proof packets, and a composed overview,
                 with fixture-friendly JSON models for deterministic use. fak tui
                 is the compatibility alias)
  fak claude-mac-fak [--dry-run] [--probe] [--prompt STR]
                (one-command Mac gateway dogfood: defaults to the always-on
                 node-macos-a fak serve gateway, fetches the bearer over ssh when
                 FAK_GATEWAY_KEY is empty, and opens interactive Claude Code
                 through the existing fak console agent launcher. --probe runs
                 a one-shot JSON check)
  fak info      [--gateway-url URL] [--interval DUR] [--once] [--json]
                (the live fak-info overlay: poll a fak guard/serve gateway's
                 /debug/vars and print ONE compact line per tick — the OBSERVED
                 cache economy (saved-token-equiv, multiplier, hit, PROVEN/REFUTED),
                 the floor SAFETY counters (blocked/repaired/quarantined), and
                 liveness. The 20% pane 'fak guard --split' opens beside the agent;
                 also runnable by hand in a second pane. Read-only; loopback needs
                 no bearer)
  fak loop      append | run -- CMD | status | admit
                (the DURABLE LONG-RUNNING-LOOP ledger: hash-chained fire/admit/start/
                 end/witness events, an OS-scheduler wrapper, a read fold, and the
                 tunable admission governor that gates the always-on loop by policy)
  fak cron      emit --target launchd|systemd|taskscheduler [--loop ID | <job>]
                [--interval DUR] [--fak-bin PATH] [--label NAME] [--ledger FILE] [-- CMD ARG...]
                (project the in-kernel loop schedule DOWN to a real OS scheduler unit:
                 render a launchd .plist / systemd timer+service / Windows
                 Register-ScheduledTask snippet whose command is 'fak loop run --loop
                 <id> ...'. The OS scheduler owns wall-clock firing; fak owns the
                 semantics (overlap-lock, missed-run). The operator installs the unit)
  fak serve     [--addr 127.0.0.1:8080 | --stdio]
                [--provider openai|anthropic|gemini|xai --base-url URL [--replica-base-url URL ...] --model M --api-key-env VAR]
                [--engine inkernel] [--gguf FILE] [--policy FILE] [--policy-check] [--require-key-env VAR] [--vdso=true]
                [--session-id ID --context-budget-tokens N [--reset-on-budget]]
                [--invalidation global|namespace|resource]
                [--engine-cache-engine sglang|vllm --engine-cache-base-url URL --engine-cache-admin-key-env VAR]
                [--engine-cache-require-exact-span]
                (the GATEWAY: front the kernel over an OpenAI-compatible HTTP surface
                 + MCP so a NON-Go agent can route tool calls through the kernel.
                 HTTP routes: POST /v1/chat/completions (adjudication proxy),
                 POST /v1/fak/{syscall,adjudicate,admit}, GET|POST /v1/fak/changes
                 (the cross-agent "what changed" feed), POST /v1/fak/revoke
                 (refute a poisoned witness), POST /v1/fak/context/change
                 (request a durable recall tombstone),
                 GET /v1/models, POST /mcp, GET /healthz, GET /metrics, GET /debug/vars. --invalidation scopes the live
                 fleet's tier-2 cache eraser. --engine-cache-engine resets a
                 self-hosted SGLang/vLLM prefix cache after a quarantined proxy
                 tool result, before the upstream turn. --engine-cache-require-exact-span
                 fails closed instead of using a whole-cache reset fallback. --stdio serves MCP (fak_adjudicate /
                 fak_syscall / fak_admit / fak_changes / fak_revoke /
                 fak_session_reset / fak_context_change) over stdin/stdout)
  fak serve-wiring [--md|--check]
                (audit fak serve flag -> gateway.Config -> runtime-read wiring)
`)
}

func usageScorecardVerbs() {
	fmt.Fprint(os.Stderr, `  fak cluster   selftest | coordinator --listen ADDR --size N --vec a,b,c |
                worker --coord ADDR --rank R --size N --vec a,b,c   [--op allreduce|allgather]
                (MULTI-NODE COMPUTE: run a real cross-node collective over fak's DistComm
                 process group (host float32). Launch 'coordinator' on one box and 'worker'
                 on each other, pointing at the coordinator's address; every rank holds only
                 its own --vec and they reduce/gather it across the wire. 'selftest' proves
                 the path bit-exact vs the in-process reference over loopback, no second box.
                 Host-layer, CPU-runnable today; the NCCL/RCCL device collective is the
                 separate GPU rung  -  see docs/serving/multi-node-compute.md)
  fak leaseref  live [--dir DIR] | list [--json] [--dir DIR] | reap [--dir DIR]
                (CROSS-MACHINE LEASE VISIBILITY (#825): read the refs/fak/locks/* ref
                 namespace internal/leaseref persists leases under, so a peer's lease
                 rides ordinary git fetch/push between clones. 'live' emits the
                 non-expired records as the dos_arbitrate live_leases JSON
                 [{lane,lane_kind,tree}] so an arbiter on another box SEES the lease;
                 'reap' deletes the expired (reapable) records. VISIBILITY, not atomic
                 acquisition  -  see docs/cli-reference.md)
  fak guard     [--provider anthropic|openai|gemini|xai] [--base-url URL] [--policy FILE]
                [--session-id ID --context-budget-tokens N [--reset-on-budget|--restart-on-budget]]
                [--restart-limit N] [--restart-seed-dir DIR]
                [--api-key-env VAR] [--env VAR] [--audit FILE|off] [--no-audit] [--dump-policy] [--quiet]
                [--split auto|on|off] [--split-where bottom|right] -- <agent command...>
                (RUN YOUR REAL AGENT THROUGH THE KERNEL: the one-command front door.
                 Starts the gateway in-process on a private loopback port, injects its
                 URL into the CHILD only (never your shell), execs the agent, and on
                 exit prints what the kernel allowed vs blocked. Default upstream is the
                 real Anthropic API in passthrough mode, so 'fak guard -- claude' wraps
                 your normal Claude Code  -  your key + prompt cache flow through, every
                 proposed tool call crosses the capability floor first. Every verdict is
                 appended to a durable, tamper-evident DECISION JOURNAL by default
                 (--audit FILE to relocate, --no-audit to turn off; replay with
                 'fak audit verify'). --dump-policy prints the built-in floor to edit;
                 --policy FILE enforces your own. --split (auto) opens a 20% 'fak info'
                 pane beside the agent in a multiplexer terminal so the live cache
                 economy + floor safety stay visible during the session)
  fak guard-verdict-rsi fold|run|--check
                (the GUARD VERDICT RSI loop: folds the real guard decision journal,
                 scores verdict-quality, and keeps only on rows + strict gain + witness)
  fak guard-rsi-scorecard [--json] [--markdown] [--compare FILE]
                (native control-pane payload for guard RSI loop maturity and realized value)
  fak dogfood-score [--json] [--markdown] [--compare FILE] [--window-hours N]
                (scores the launched-session dogfooding loop: is it WIRED to run honestly,
                 and does the model report itself truthfully  -  the keystone defect is a turn
                 that claims success over an OBSERVED Stop-hook error, read from real transcripts)
  fak token-defaults-scorecard [--json] [--markdown]
                (native token-saving-defaults control-pane payload)
  fak skill-effectiveness-scorecard [--json] [--markdown]
                (native skill-pack effectiveness control-pane payload)
  fak conflation-scorecard [--json] [--markdown] [--compare FILE]
                (native provenance-honesty control-pane payload: every reported number/status
                 labels its provenance -- WITNESSED vs OBSERVED -- folded into conflation_debt)
  fak claim-check --self-test | --file claim.json | --statement S --baseline real|strawman|none
                  [--net] --scope S --provenance WITNESSED|OBSERVED|MODELED|SIMULATED --witness S
                  [--realized=false --gate-reason R] [--json]
                (the named NET-TRUE follow-on (#1171): grade an efficiency/perf claim against
                 the six-question net-true-value rubric -> net-true (0) / strawman (3) / not-yet
                 (3). --self-test grades the built-in honest+strawman corpus. docs/standards/net-true-value.md)
  fak support-maturity-scorecard [--json] [--markdown] [--compare FILE]
                [--matrix-md] [--write-doc] [--check-doc] [--workspace DIR]
                (native support-maturity payload: fold the generated model x backend coverage
                 matrix into support_maturity_debt, coverage percentage, and an A-F grade.
                 --matrix-md emits the generated matrix block; --write-doc regenerates it in
                 docs/HARDWARE-MATRIX.md; --check-doc reds when a committed cell is stale)
  fak learning-debt-dispatch --scorecard FILE [--cap N] [--cache FILE] [--live]
                [--fetch-existing] [--existing-json FILE] [--repo owner/repo] [--json]
                (learning-scorecard -> backlog: file at most --cap triage issues for HARD
                 learning-debt defects, deduped by the gitignored seen-cache plus issue-body
                 markers. Dry-run by default; --live is required to call gh and update cache)
  fak steering  status | report | alert [--index-delta N] [--pin] | pin
                [--channel ID] [--scorecard-json FILE] [--dry-run]
                (the STEERABILITY Slack surface for #steering-guard: status posts the
                 current index card, report posts the full per-KPI + per-group snapshot,
                 alert posts ONLY on a regression vs the pinned floor (tools/steering_baseline.json)
                 -- hard debt > 0, index drop, or a NEW drift signal -- with action buttons
                 pointing each drift at the skill that retires it; pin re-baselines the floor.
                 Posts via FAK_SCOREBOARD_TOKEN, never the lab SLACK_BOT_TOKEN)
  fak product   post [--status | --persona | --from FILE | --title T --notes BODY]
                [--notes-file FILE] [--channel ID] [--source WHO] [--dry-run]
                (the PRODUCT-direction Slack surface for #product: --status folds the
                 product scorecard, --persona folds persona-readiness, --title/--notes
                 posts free-form product prose (persona items, direction calls). Same
                 workspace as #scoreboard but resolves FAK_PRODUCT_CHANNEL -- never falls
                 back to #scoreboard. Posts via FAK_SCOREBOARD_TOKEN, not the lab token)
  fak nodeusage post [--fleet snap.json | --kpi NAME --value V --grade G --verdict OK|ACTION |
                --from FILE] [--detail D] [--title T] [--channel ID] [--source WHO] [--dry-run]
                (the COMPUTE-NODE-USAGE Slack surface for #node-usage: --fleet folds a
                 'fak lab status --json' snapshot (the headline node-usage signal:
                 per-state/per-class node counts + readiness) into a card; --kpi posts
                 an ad-hoc node-usage number (active workers, open-issue inbound load).
                 Same workspace as #scoreboard but resolves FAK_NODE_USAGE_CHANNEL --
                 never falls back to #scoreboard. Posts via FAK_SCOREBOARD_TOKEN
                 (node-usage token fallback), not the lab token)
  fak slack     check [--auth] [--json] | send --channel ID --text MSG [--token T] [--dry-run]
                (DEBUG + USE the whole Slack surface from one place. 'check' reports, for
                 every surface (scoreboard/blockers/bench/dispatch/dojo/marketing/
                 node-usage/product/steering/chatrelay), the bot token + channel it would
                 use and WHERE each resolved from (env / .env.slack.local / fallback /
                 default); --auth runs Slack auth.test per distinct token to prove it
                 actually works (exit 1 on any failure, so it gates CI); --json for tooling.
                 'send' posts an ad-hoc message to ANY channel (token defaults to
                 FAK_SCOREBOARD_TOKEN), --text - reads the body from stdin, --dry-run previews)
  fak cadence   [--json] [--check] [--append-history] [--window N] [--ledger FILE]
                (the CONSOLIDATED regular-cadence report: folds the three dimensions
                 an operator tracks  -  SCORES (scorecard control pane), WORK-DONE
                 (git commits + '(fak ' ships over a trailing window), RELEASES
                 (release-status)  -  into one control-pane envelope. --append-history
                 records a dated row in docs/cadence/history.jsonl so the trend accrues
                 across weeks; --check is advisory (non-zero only if a dimension could
                 not be measured; the scorecard ratchet owns debt regressions))
  fak release-staleness [--json] [--check] [--stale-commits N] [--stale-days N]
                (the PUBLISH-freshness signal: how far the latest published vX.Y.Z tag
                 -  what 'go install ...@latest' resolves to  -  lags HEAD, in commits AND
                 days. The dual of 'fak self-update' (which converges a built-from-source
                 binary on origin/main): this catches @latest rotting when no tag is cut
                 as work lands. --check exits non-zero when @latest is stale/very-stale)
  fak dojo      run --corpus DIR [--ttl 5m|1h] [--lever a,b] [--json] [--check]
                    [--append-history] [--ledger FILE] | list [--json]
                (the prediction-vs-reality gym: scores each calibration lever's CLAIMED
                 vs REALIZED behavior over a corpus of real Claude Code transcripts and
                 trends the per-lever calibration error; --append-history records a dated
                 row in docs/dojo/history.jsonl; list shows the registered levers)
  fak dojo-rsi  fold|propose|run|loop|trend
                (the self-pacing dojo RSI loop: selects the next calibration cell by
                 novelty x value x staleness, journals KEEP/REVERT/ESCALATE rows in
                 docs/dojo/rsi-journal.jsonl, routes REPROJECT/HARVEST to the agent arm,
                 and renders the committed KEEP/REVERT trend for CI)
  fak nightrun  next | plan | run [--apply] [--loop] [--max N] | ledger | caps  [--json]
                (RUN IT ALL NIGHT: the local-capability-aware data-collection door.
                 Probes THIS box (gpu/weights/datasets/creds), ranks the feasible-here
                 collection tasks  -  the benchmark grid PLUS the curated open-witness
                 backlog  -  by novelty x value x staleness, and answers "what is the
                 single most important datum to collect here right now" (next). run is
                 DRY-RUN unless --apply; --apply executes + appends an OBSERVED row to
                 docs/nightrun/collected.jsonl, --loop collects the whole feasible queue.
                 A task the box can't run is never selected, so the loop can't claim a
                 datum the hardware can't produce)
  fak audit     verify <journal.jsonl> | export <journal.jsonl>
                (the AUDIT-TRAIL consumer: 'verify' re-reads a decision journal (the
                 'fak guard' / FAK_AUDIT_JOURNAL trail) and validates its hash chain
                 end to end  -  exit 1 naming the first broken link if a byte changed
                 since it was written; 'export' re-emits it as JSONL. A self-report is
                 not a witness  -  this is how the record is checked offline)
  fak stopfailure plan | reset-stale | archive-marker-only | clear-reviewed
                (operator surface for .dos/stop-failures breaker markers. plan is
                 read-only; reset-stale is dry-run unless --apply is passed. The
                 default one-day lens matches the guard/MCP status queue; pass
                 --since-hours 0 for all history. Only stale marker consecutive counts
                 with transcript/stream evidence are reset; marker-only stale files can
                 be moved under archive/ explicitly. Recent markers require
                 clear-reviewed --session ID after human/operator review)
  fak headroom  list | status | compress [--via NAME] [--model ID] [--emit] [FILE|-]
                (the CONTEXT-COMPRESSION seam: shrink tool outputs/logs/files before
                 they reach the model, reversibly. A pluggable AREA  -  one generic
                 Compressor interface, swappable plugins: noop (off default), native
                 (in-process structural, zero deps), headroom (bridge to a running
                 'headroom proxy'). The selected plugin folds into the result path as
                 a ResultAdmitter, so 'fak guard'/'fak serve' compress in-stream.
                 Pick with FAK_COMPRESSOR; 'compress' proves the savings with no model)
  fak vcache    status | prove | prove-telemetry
                  (the VIRTUAL PROVIDER-CACHE status/proof surface. 'status' reports
                   what is actually up: the M5 Governor is a local off-path policy
                   engine, while provider calibration/warming/recall remain issue-tracked.
                   'prove' runs the deterministic star-anchor token-savings proof and
                   exits 0 when PROVEN, 1 when REFUTED; 'prove-telemetry' proves/refutes
                   realized savings from provider usage JSONL)
  fak hook      < call.json     (spawned-hook decide; the A/B baseline transport)
  fak hooks     pre-commit | commit-msg <file>
                (in-process repo git-hook gates; exit 2 lets shell hooks fall back)
  fak hygiene   [--gates A,B,...]   (in-process --audit-tree hygiene gates; the
                make hygiene / make index-sync backstop; exit 2 lets make fall back)
  fak webbench  describe | eval | compare    (frontier web/browser agent benchmarking)
  fak swebench  describe | eval | compare    (SWE-bench Verified benchmarking)
  fak dojo      run --corpus DIR | list    (prediction-vs-reality calibration gym)
  fak dojo-rsi  fold|propose|run|loop|trend (dojo self-improvement loop)
  fak version

every tool call crosses one in-process syscall boundary: vDSO -> adjudicate ->
pre-flight/grammar -> dispatch -> context-MMU admit.
`)
}
