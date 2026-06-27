// Package nightrun is the "run it all night" center of excellence: the one
// place that answers, for an operator OR an agent, the single recurring
// question of unattended data collection —
//
//	"what is the most important datum I can collect on THIS box, right now?"
//
// and then closes the loop on it (run → capture → record → pick the next one).
//
// WHY THIS EXISTS. fak already had the PARTS of overnight benchmarking but no
// SPINE that ties them into a trivial run-it-all-night door:
//
//   - internal/benchcatalog is the MENU — what 35 benchmarks exist, and the
//     cold-start cost (offline / weights / dataset) of each. It says what CAN be
//     run; it never says what SHOULD be run next, or whether THIS box can run it.
//   - experiments/benchmark/catalog.json + tools/bench_plan.py is a next()
//     BRAIN — but it plans for a fixed roster of REMOTE bench-nodes (a100 / L4 /
//     mac) off a static registry, is PLAN-ONLY by construction, and cannot answer
//     "what can the box I am sitting on collect tonight."
//   - internal/cadencereport is the durable-JSONL-ledger + trend PATTERN, but for
//     project progress (scores / work / releases), not data collection.
//
// nightrun is the missing operator/agent front door over those parts. It is
// LOCAL-CAPABILITY-AWARE (it probes the box it runs on, so it never proposes a
// CUDA benchmark on a Mac or an HW-gated witness on a box with no GPU),
// LOOP-CLOSING (a durable collection ledger records what was gathered so next()
// skips fresh data and surfaces stale data), and UNIFIED (the candidate set spans
// both the benchmark grid AND a curated backlog of the named, still-open measured
// witnesses — the PENDING_MEASUREMENT / on-box-re-measure items that are the real
// "most important data" right now).
//
// THE SUB-CONCEPTS, named once so the namespace stays crisp (this repo is
// deliberate about that — see docs/concept-disambiguation-scorecard):
//
//   - a collection Task is one unit of "data to gather": a stable id, what the
//     box must HAVE to run it (Requirements), how important it is (Value), the
//     exact runnable command, and the Acceptance that proves it was collected. A
//     Task is WORK TO DO, never a result — so it cannot overclaim.
//   - Capabilities is the probed fact-sheet of THIS box (GPU kind, weights,
//     datasets, creds, net). Satisfies(task) is the feasibility filter.
//   - next() is the pure, deterministic selector: rank the feasible-here tasks by
//     novelty × value × staleness and return the single best one.
//   - the collection ledger (docs/nightrun/collected.jsonl) is the durable,
//     append-only record of what was gathered, on which box, when — the loop's
//     memory, and the input next() reads to compute staleness.
//
// nightrun is NOT internal/loop (that re-runs a prompt on an interval) and NOT
// internal/witness (that resolves a CLAIM from git evidence). nightrun selects
// and runs DATA-COLLECTION TASKS; "acceptance" is the artifact that proves a
// datum was gathered, distinct from a witness over a shipped claim.
//
// HONESTY BOUNDARY. next/plan are pure reads — deterministic given an injected
// now and an injected Capabilities, so a test pins them byte-for-byte. The run
// loop is DRY-RUN by default (it prints what it WOULD execute and writes
// nothing); only --apply executes real commands, and then the ledger row records
// what was OBSERVED (exit status, artifact path) and NEVER a fabricated number. A
// task the box cannot run is never selected, so the loop can never claim to have
// collected HW-gated data on hardware that cannot produce it.
//
// UNATTENDED / DETACHED RUNS. Starting `fak nightrun run --apply --loop` from a
// detached session (a `setsid`/`nohup` job, a cron line, a minimal-env container)
// has two environment requirements an interactive shell hides:
//
//   - A Go BUILD CACHE. The loop pre-builds each `go run ./cmd/<x>` bench once
//     (prebuild.go) to avoid paying compile cost per task. `go build` aborts with
//     "build cache is required, but could not be located" when neither GOCACHE nor
//     a derivable default (HOME on unix, LocalAppData on Windows) is present — so a
//     minimal-env detached run would lose EVERY go-run bench. The preflight
//     (preflightGoCache) provisions a per-run GOCACHE under the build temp dir so
//     the benches still build, and the artifact names the durable fix. For a
//     persistent cache across runs, export HOME or GOCACHE before starting.
//   - A per-task TIMEOUT for heavy collections. Each --apply attempt is bounded by
//     Task.timeout() (DefaultTaskTimeoutSec = 15 min), generous for the
//     offline/smoke lane but short for a cold model load + serve + throughput
//     sweep. Heavy witnesses carry an explicit TimeoutSec (a cold GLM load: 30 min;
//     a serve+throughput sweep: 1 h). For a one-off heavy collection, raise it
//     WITHOUT recompiling via the overlay file (DefaultOverlayRel): add a row with
//     `"timeout_sec": 3600`. A task that exceeds its budget is killed and recorded
//     OBSERVED as a timeout, never a success, so one hung run cannot stall the loop.
//
// A curated witness whose Run is a HUMAN RECIPE (a `<placeholder>`, a prose arrow,
// or a bare `script.sh   # comment` that needs operator setup) carries Manual:true
// and is recorded OutcomeSkipped — surfaced by plan/next for a human, never
// auto-run and never a spurious ledger failure (see Task.autoRunnable).
package nightrun
