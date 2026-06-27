// Package benchcatalog is the single, in-binary source of truth for "what
// benchmarks does fak have, what does each measure, and how do I run it." It
// exists because the answer used to be scattered across 18 separate cmd/*bench*
// mains plus five `fak` verbs (bench, turntax, routebench, webbench, swebench),
// each with its own bespoke flag vocabulary and no shared index. A developer who
// wanted to "run a benchmark" had to already know which binary to invoke and
// which flags it took.
//
// This registry is read by:
//   - `fak benchmarks list|describe|run` (cmd/fak/benchmarks.go)  -  the one
//     discoverable door over every benchmark surface.
//   - tools/bench_dx_scorecard.py  -  which cross-checks the registry against the
//     actual cmd/ tree, so the catalog cannot silently drift from reality (a
//     bench main with no entry, or an entry pointing at a deleted main, both
//     fail the scorecard).
//
// The registry is deliberately a flat literal, not a directory scan: it carries
// the human-meaningful PURPOSE and COLD-START NEEDS of each benchmark, which a
// directory walk cannot recover. Adding a benchmark means adding one row here  -
// the same additive-leaf discipline the rest of the kernel uses.
package benchcatalog

import "sort"

// Need classifies the cold-start cost of a benchmark  -  the single fact a newcomer
// most needs before typing a command.
type Need string

const (
	// NeedNone runs to a real result with no model weights, no GPU, no network,
	// and no API key  -  the zero-friction entry points a newcomer should try first.
	NeedNone Need = "offline"
	// NeedWeights requires local model weights (an export dir, an HF snapshot, or
	// a GGUF) and/or a GPU. The number is real but the setup is not zero-cost.
	NeedWeights Need = "weights"
	// NeedDataset requires an external dataset file (e.g. a WebVoyager export)
	// that is not committed to the repo.
	NeedDataset Need = "dataset"
)

// Kind separates a standalone `cmd/<name>` binary from a `fak <verb>` subcommand,
// because they are invoked differently (`go run ./cmd/<name>` vs `fak <verb>`).
type Kind string

const (
	KindCmd  Kind = "cmd"  // a standalone cmd/<name> main
	KindVerb Kind = "verb" // a `fak <verb>` subcommand of this binary
)

// Level classifies what KIND of number a benchmark produces, on the axis that
// matters for "is this worth collecting on a capable box": a serving/e2e bench
// emits a real hardware or graded-task number (coverage-class, worth filling per
// box); a micro/component bench is a smoke check (the floor). This is orthogonal
// to Kind (cmd vs verb, an invocation detail) and to Need (cold-start cost) - it
// is the importance axis `fak nightrun` ranks by. The zero value ("" ==
// LevelSmoke) is the safe default: an unannotated bench is treated as a smoke
// check, never silently promoted to coverage-class.
type Level string

const (
	// LevelSmoke is the zero value: an offline micro/component check, a rollup, or
	// a converter - it guards a boundary but does not produce a headline number.
	LevelSmoke Level = ""
	// LevelServing produces a real serving-throughput / latency hardware number
	// (tok/s, p50, batched decode) and needs weights and/or a GPU.
	LevelServing Level = "serving"
	// LevelE2E produces an end-to-end graded-task outcome number (solve rate,
	// safe-pass) over a real agent/task harness.
	LevelE2E Level = "e2e"
)

// Bench is one benchmark surface: what it measures, how much it costs to start,
// and the exact command that runs it.
type Bench struct {
	Name    string // the registry key, also the cmd dir / fak verb (e.g. "modelbench", "webbench")
	Kind    Kind
	Need    Need
	Level   Level  // the importance axis (serving/e2e vs smoke); zero value == LevelSmoke
	Summary string // one line: what NUMBER this benchmark produces and what it means
	// Run is the copy-pasteable command that runs the offline/default arm. For a
	// weights/dataset bench it is the smallest meaningful invocation; the flags
	// that point at assets are documented in Flags.
	Run string
	// Flags is the short list of the knobs that matter most, one "name  -  meaning"
	// per entry. Not exhaustive  -  the binary's own -h is the full surface.
	Flags []string
	// Doc is the in-repo methodology/authority doc for this benchmark's numbers,
	// or "" when the only doc is the source comment.
	Doc string
}

// Offline reports whether this benchmark runs with zero external assets  -  the
// predicate `fak benchmarks list --offline` filters on.
func (b Bench) Offline() bool { return b.Need == NeedNone }

// registry is the literal source of truth. Keep it sorted by Name; the scorecard
// asserts every cmd/*bench* main and every `fak` bench verb appears exactly once.
var registry = []Bench{
	{
		Name: "ablate", Kind: KindVerb, Need: NeedNone,
		Summary: "Self-ablation feature sweep: replays ONE frozen tool-call trace under N runtime-feature configs and reads each arm's cost/benefit (vDSO hits, denies, p50 latency, tokens) straight off the kernel counters  -  the N-arm generalization of `fak bench`, apples-to-apples on one workload hash.",
		Run:     "fak ablate --sweep vdso",
		Flags:   []string{"--sweep  -  comma list of features to ablate (known: vdso)", "--suite  -  trace suite under testdata/tau2", "--baseline  -  arm id used as the delta reference", "--engine  -  engine id (offline mock by default)", "--out  -  AblationReport JSON path"},
		Doc:     "docs/benchmarks/ABLATE-RESULTS.md",
	},
	{
		Name: "agenticbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Parent #868 rollup gate: folds the committed agentic benchmark child artifacts and reports whether any real raw-vs-fak result claim is allowed.",
		Run:     "go run ./cmd/agenticbench -out experiments/agent-live/agentic-benchmark-epic-868-status-20260626.json -md experiments/agent-live/agentic-benchmark-epic-868-status-20260626.md -external-queue experiments/agent-live/agentic-benchmark-epic-868-external-harness-queue-20260626.json -external-queue-md experiments/agent-live/agentic-benchmark-epic-868-external-harness-queue-20260626.md",
		Flags:   []string{"-root  -  repo root containing artifacts", "-out  -  rollup JSON path", "-md  -  markdown summary path", "-external-queue  -  pending external harness queue JSON path", "-external-queue-md  -  pending external harness queue markdown path", "-strict  -  exit nonzero unless #868 is complete"},
		Doc:     "docs/notes/AGENTIC-BENCHMARK-RUN-PACKETS-2026-06-25.md",
	},
	{
		Name: "bench", Kind: KindVerb, Need: NeedNone,
		Summary: "A/B ablation of the vDSO over a frozen tau2 trace  -  the per-turn adjudication work fak eliminates vs a spawned-hook baseline.",
		Run:     "fak bench --suite tau2-smoke --out report.json",
		Flags:   []string{"--suite  -  trace suite under testdata/tau2", "--out  -  report path", "--baseline-n  -  spawned-hook samples"},
		Doc:     "BENCHMARK-AUTHORITY.md",
	},
	{
		Name: "benchscore", Kind: KindCmd, Need: NeedNone,
		Summary: "Aggregates the committed benchmark score.json results into a cross-model score matrix and re-derives every ratio, so a drifted or overclaimed number fails the scan  -  the record-and-verify half of the benchmark loop.",
		Run:     "go run ./cmd/benchscore -root experiments/benchmark/runs",
		Flags:   []string{"-root  -  directory tree scanned for score.json files", "-json  -  emit JSON instead of the markdown matrix"},
		Doc:     "docs/benchmark/ARM64-QKERNEL-SCORE.md",
	},
	{
		Name: "browseractionbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Browser/computer-use action-mediation smoke: replays local browser traces through raw and fak arms and reports safe pass^1, policy breaches, minefield hits, denied actions, and evidence checkpoints.",
		Run:     "go run ./cmd/browseractionbench",
		Flags:   []string{"-suite  -  browser action mediation suite JSON", "-contract  -  emit external official-run contract", "-out  -  report JSON path", "-md  -  markdown summary path"},
		Doc:     "docs/notes/AGENTIC-BENCHMARK-RUN-PACKETS-2026-06-25.md",
	},
	{
		Name: "batchbench", Kind: KindCmd, Need: NeedWeights, Level: LevelServing,
		Summary: "Aggregate multi-user batched-decode throughput as a function of batch size B (the continuous-batching regime).",
		Run:     "go run ./cmd/batchbench -dir internal/model/.cache/smollm2-135m",
		Flags:   []string{"-dir  -  model export dir", "-quant  -  Q8_0 lane", "-reps  -  reps per batch size", "-out  -  JSON out"},
		Doc:     "docs/production-benchmark-methodology.md",
	},
	{
		Name: "causalbench", Kind: KindCmd, Need: NeedNone,
		Summary: "End-to-end demonstrator for fak's causal cache-invalidation: a write invalidates exactly the dependent reuse, no more.",
		Run:     "go run ./cmd/causalbench",
		Doc:     "",
	},
	{
		Name: "ctxbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Runs the fak security gates over a corpus of tool calls/results; default is the committed synthetic poison fixture.",
		Run:     "go run ./cmd/ctxbench",
		Flags:   []string{"-corpus  -  corpus JSON (default committed poison fixture)", "-chain  -  fold the full ResultAdmitter chain", "-out  -  JSON report"},
		Doc:     "",
	},
	{
		Name: "ctxplanbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Measures the ctxplan planned VIEW over real Claude Code session transcripts  -  context kept vs dropped.",
		Run:     "go run ./cmd/ctxplanbench",
		Doc:     "",
	},
	{
		Name: "fanbench", Kind: KindCmd, Need: NeedNone,
		Summary: "One-master-goal -> N-subagent fan-out sweep: the prefill-token work floor across a worker sweep.",
		Run:     "go run ./cmd/fanbench",
		Doc:     "docs/explainers/fleet-benchmarks.md",
	},
	{
		Name: "fleetbench", Kind: KindCmd, Need: NeedNone,
		Summary: "2-D turn-tax sweep (turns-per-agent T x fleet size A) over the real kernel; emits JSON + CSV for curve fitting.",
		Run:     "go run ./cmd/fleetbench",
		Doc:     "docs/explainers/fleet-benchmarks.md",
	},
	{
		Name: "longctxbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Renders the exact contention-free work floor for the long-context regime.",
		Run:     "go run ./cmd/longctxbench",
		Doc:     "",
	},
	{
		Name: "modelbench", Kind: KindCmd, Need: NeedWeights, Level: LevelServing,
		Summary: "In-kernel pure-Go forward-pass latency / tok-per-sec, so the kernel's model numbers are self-measured not borrowed.",
		Run:     "go run ./cmd/modelbench -dir internal/model/.cache/smollm2-135m",
		Flags:   []string{"-dir  -  fak export dir", "-hf  -  HuggingFace snapshot", "-gguf  -  GGUF checkpoint", "-quant/-lean  -  Q8_0", "-out  -  JSON out"},
		Doc:     "docs/model-engine-env.md",
	},
	{
		Name: "paritybench", Kind: KindCmd, Need: NeedNone,
		Summary: "Assembles the cross-model parity artifact from recorded per-model results (ingest + compare, no live model).",
		Run:     "go run ./cmd/paritybench",
		Doc:     "",
	},
	{
		Name: "polymodelbench", Kind: KindCmd, Need: NeedNone,
		Summary: "The measured-numbers half of the poly-model comparison  -  the runnable artifact behind the multi-model table.",
		Run:     "go run ./cmd/polymodelbench",
		Doc:     "",
	},
	{
		Name: "q8bench", Kind: KindCmd, Need: NeedWeights, Level: LevelServing,
		Summary: "Independent verifier for the int8-quantized in-kernel forward path (numerical parity vs f32).",
		Run:     "go run ./cmd/q8bench -dir internal/model/.cache/smollm2-135m",
		Flags:   []string{"-dir  -  model export dir"},
		Doc:     "",
	},
	{
		Name: "radixbench", Kind: KindCmd, Need: NeedWeights, Level: LevelServing,
		Summary: "fak's KV-cache prefix reuse vs SGLang's RadixAttention regime  -  prefix-cache hit value.",
		Run:     "go run ./cmd/radixbench -synthetic smollm2-135m",
		Flags:   []string{"-synthetic  -  weightless synthetic shape (ratios faithful)", "-hf/-dir  -  live arm", "-quant/-lean  -  Q8_0"},
		Doc:     "docs/explainers/fleet-benchmarks.md",
	},
	{
		Name: "routebench", Kind: KindVerb, Need: NeedNone,
		Summary: "Offline routing benchmark: replays a corpus of recorded cases through the router and scores routed vs single-model.",
		Run:     "fak routebench",
		Flags:   []string{"--corpus  -  cases file", "--routed/--single  -  comparison inputs", "--frontier  -  frontier model id"},
		Doc:     "",
	},
	{
		Name: "sessionbench", Kind: KindCmd, Need: NeedWeights, Level: LevelServing,
		Summary: "Net value-add of the fused agent kernel on a multi-turn session vs a tuned warm-cache baseline.",
		Run:     "go run ./cmd/sessionbench -synthetic smollm2-135m",
		Flags:   []string{"-synthetic  -  weightless shape (ratios faithful, wall-clock this-box)", "-hf/-dir  -  live arm", "-quant"},
		Doc:     "docs/production-benchmark-methodology.md",
	},
	{
		Name: "swebench", Kind: KindVerb, Need: NeedNone, Level: LevelE2E,
		Summary: "SWE-bench Verified benchmarking surface (describe | eval | compare). describe is offline; eval is gated on the harness.",
		Run:     "fak swebench describe",
		Flags:   []string{"describe  -  offline geometry", "eval  -  graded (gated)", "compare  -  side-by-side"},
		Doc:     "BENCHMARK-AUTHORITY.md",
	},
	{
		Name: "terminalbench", Kind: KindCmd, Need: NeedNone,
		Summary: "Terminal-Bench-shaped command-boundary smoke: replays local command traces through raw and fak arms and reports solve, safe resolve, blocked dangerous actions, unnecessary blocks, and command evidence.",
		Run:     "go run ./cmd/terminalbench",
		Flags:   []string{"-suite  -  Terminal-Bench-shaped command suite JSON", "-contract  -  emit external official-run contract", "-out  -  report JSON path", "-md  -  markdown summary path"},
		Doc:     "docs/notes/AGENTIC-BENCHMARK-RUN-PACKETS-2026-06-25.md",
	},
	{
		Name: "toolsandboxbench", Kind: KindCmd, Need: NeedNone,
		Summary: "ToolSandbox/tau3-shaped policy-state adapter smoke: replays the same local policy/minefield trace through raw and fak arms and reports safe pass^1 plus denied policy calls.",
		Run:     "go run ./cmd/toolsandboxbench",
		Flags:   []string{"-suite  -  ToolSandbox/tau3-shaped suite JSON", "-contract  -  emit external official-run contract", "-out  -  report JSON path", "-md  -  markdown summary path"},
		Doc:     "BENCHMARK-AUTHORITY.md",
	},
	{
		Name: "topobench", Kind: KindCmd, Need: NeedNone,
		Summary: "Fleet-topology genome search (issue #541): searches the orthogonal topology space for the cheapest fan-out shape.",
		Run:     "go run ./cmd/topobench",
		Doc:     "",
	},
	{
		Name: "turntax", Kind: KindVerb, Need: NeedNone,
		Summary: "The turn-tax A/B: replays a class-labeled trace through the real kernel  -  per-turn overhead fak removes.",
		Run:     "fak turntax --suite turntax-airline",
		Flags:   []string{"--suite  -  trace suite under testdata/turntax", "--out  -  report path", "--breakeven  -  amortization curve"},
		Doc:     "BENCHMARK-AUTHORITY.md",
	},
	{
		Name: "vcache", Kind: KindVerb, Need: NeedNone,
		Summary: "vCache 2x readiness scorecard: planned/observed cache savings, workload concentration, false-warm risk, recall risk, and hot-anchor index size.",
		Run:     "fak vcache bench --json",
		Flags: []string{
			"--telemetry  -  provider cache-read JSONL to score observed savings",
			"--anchors-file  -  JSONL/JSON/CSV ranked anchor workload",
			"--index-out  -  write the selected fak.vcache.anchor_index.v1 artifact",
			"--plan-out  -  write the selected fak.vcache.dev_plan.v1 artifact",
			"--two-x  -  multiplier gate required for success",
		},
		Doc: "docs/cli-reference.md",
	},
	{
		Name: "webbench", Kind: KindVerb, Need: NeedNone, Level: LevelE2E,
		Summary: "Frontier web/browser agent benchmarking (describe | eval | compare). describe prints the offline prefill-work geometry.",
		Run:     "fak webbench describe --dataset testdata/webvoyager/sample.json",
		Flags:   []string{"describe  -  offline geometry (needs --dataset)", "eval  -  graded (gated)", "compare  -  vs a results file"},
		Doc:     "docs/webbench-baselines.md",
	},
	{
		Name: "webbench-convert", Kind: KindCmd, Need: NeedDataset,
		Summary: "Converts a WebVoyager dataset export into the webbench task format.",
		Run:     "go run ./cmd/webbench-convert",
		Doc:     "docs/webbench-baselines.md",
	},
	{
		Name: "webbench-run", Kind: KindCmd, Need: NeedDataset, Level: LevelE2E,
		Summary: "Reproducible end-to-end webbench runner over a converted task set.",
		Run:     "go run ./cmd/webbench-run",
		Doc:     "docs/webbench-baselines.md",
	},
	{
		Name: "webbench-token-measure", Kind: KindCmd, Need: NeedDataset,
		Summary: "Measures actual token usage from real model-API webbench runs (the measured arm behind the geometry model).",
		Run:     "go run ./cmd/webbench-token-measure",
		Doc:     "docs/webbench-real-measurements-summary.md",
	},
	{
		Name: "wfmembench", Kind: KindCmd, Need: NeedNone,
		Summary: "Three-arm workflow-memory comparator (issue #434): no-memory vs naive vs fak workflow memory.",
		Run:     "go run ./cmd/wfmembench",
		Doc:     "",
	},
}

// All returns every registered benchmark, sorted by Name (deterministic order).
func All() []Bench {
	out := make([]Bench, len(registry))
	copy(out, registry)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns the benchmark with the given name and whether it was found.
func Get(name string) (Bench, bool) {
	for _, b := range registry {
		if b.Name == name {
			return b, true
		}
	}
	return Bench{}, false
}

// Offline returns just the zero-asset benchmarks  -  what a newcomer can run right
// now with no weights, GPU, dataset, or key.
func Offline() []Bench {
	var out []Bench
	for _, b := range All() {
		if b.Offline() {
			out = append(out, b)
		}
	}
	return out
}

// Names returns every registered benchmark name, sorted  -  used by the scorecard
// to assert one-to-one coverage against the cmd/ tree.
func Names() []string {
	all := All()
	out := make([]string, len(all))
	for i, b := range all {
		out[i] = b.Name
	}
	return out
}
