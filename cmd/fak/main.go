// Command fak is the Fused Agent Kernel: one statically-linked Go binary that
// runs an agentic tool loop where every tool call passes through one in-process
// policy and quarantine boundary (adjudicate -> vDSO -> pre-flight/grammar ->
// dispatch -> context-MMU admit). Verbs:
//
//	fak run       — replay a trace (or a single call) through the kernel
//	fak preflight — run only the pre-flight + grammar rungs over a call
//	fak bench     — A/B ablate the vDSO over a frozen trace, emit report.json
//	fak policy    — dump / validate the deployable capability-floor manifest
//	fak hook      — spawned-hook mode: decide one call from stdin (the baseline)
//
// The single blank import of internal/registrations is what wires every leaf
// subsystem into the frozen ABI before the kernel boots.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toollint"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "preflight":
		cmdPreflight(os.Args[2:])
	case "attest":
		cmdAttest(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "turntax":
		cmdTurnTax(os.Args[2:])
	case "agent":
		cmdAgent(os.Args[2:])
	case "recall":
		cmdRecall(os.Args[2:])
	case "dream":
		cmdDream(os.Args[2:])
	case "memory":
		cmdMemory(os.Args[2:])
	case "debug":
		cmdDebug(os.Args[2:])
	case "policy":
		cmdPolicy(os.Args[2:])
	case "lint":
		cmdLint(os.Args[2:])
	case "codelint":
		cmdCodelint(os.Args[2:])
	case "answer-shape":
		cmdAnswerShape(os.Args[2:])
	case "doctor":
		cmdDoctor(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "guard":
		cmdGuard(os.Args[2:])
	case "audit":
		cmdAudit(os.Args[2:])
	case "headroom":
		cmdHeadroom(os.Args[2:])
	case "hook":
		cmdHook()
	case "swebench":
		cmdSwebench(os.Args[2:])
	case "webbench":
		cmdWebbench(os.Args[2:])
	case "model":
		cmdModel(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println(appversion.Current())
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "fak: unknown verb %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "fak - the Fused Agent Kernel (v%s)\n\n", appversion.Current())
	fmt.Fprint(os.Stderr, `usage:
  fak run       --trace FILE [--engine inkernel] [--vdso=true] [--policy FILE]
  fak preflight --tool NAME --args JSON [--policy FILE]
  fak attest    --policy FILE [--probes FILE] [--out FILE] [--json] [--quiet]
                 (the COMPLIANCE ATTESTATION GENERATOR: prove the capability floor
                  from preflight. Runs the real adjudication fold over a probe set
                  and emits a re-checkable attestation. Default probes are DERIVED
                  from the manifest — each deny must be DENIED with its cited reason,
                  each allow/allow_prefix ALLOWED, and an unnamed tool DENIED
                  DEFAULT_DENY. --probes FILE attests arg-value cases. Exit 0 if the
                  floor is PROVEN, 1 if any probe drifts, 2 on usage error)
  fak model     load <hf://owner/repo[@rev]/file>
                (resolve an hf:// URI to a locally cached file path: Hub download with
                 HF_TOKEN auth and SHA256 verification against the Hub LFS oid. The
                 cached path is printed on stdout; --gguf and the loaders accept it)
  fak bench     --suite NAME [--out report.json] [--baseline-n 30]
                (transport A/B: in-process adjudication p50 vs spawned-hook p50)
  fak turntax   --suite NAME [--out turntax-report.json]
                [--prompt-tokens N --completion-tokens N --turn-latency-ms F]
                (TURN-TAX A/B: the extra error-code MODEL TURN a SOTA loop fires —
                 malformed args, duplicate read, poison — vs fak's 1-shot. Replays a
                 class-labeled trace through the real kernel, prices the turns it
                 deletes per lever, and keeps the safety floor on its own axis)
  fak agent     [--task STR] [--provider openai|anthropic|gemini|xai]
                [--base-url URL --model M --api-key-env VAR | --offline]
                [--max-turns N] [--out agent-report.json] [--policy FILE]   (LIVE turn-count A/B)
  fak policy    --dump | --check FILE
                (--dump writes the built-in DefaultPolicy as a manifest you edit;
                 --check validates a manifest and prints the floor it admits. The
                 capability floor — WHICH tools may be called — is a deployable
                 file, not a Go edit: dump -> edit -> --check -> --policy)
  fak lint      [--json] [--strict] [--kernel-only]
                (the STATIC TOOL LINTER: the definition-time dual of the kernel's
                 call-time re-checks. Reports a dead cache hint, an unreachable pure
                 registration, a canned answer for a write-shaped tool, or a schema
                 the model is shown but the kernel never enforces — once, instead of
                 the runtime silently papering over it every call. Exit 1 on an
                 error finding, or on any finding with --strict)
  fak codelint  [--json] [--errors-only] [--list] PATH...
                (the LANGUAGE-SERVER-PACK code linter: route each file to the pack
                 that owns its extension and report parse/compile errors — the
                 write-time check the kernel runs over CODE the agent produces
                 (Go/JSON in-process, Python/CUDA via their toolchains, degrading
                 to no-opinion where a checker is absent). The same Lint the
                 SWE-bench fleet runs on every agent file write. Exit 1 on an error)
  fak answer-shape [--text - | --file PATH | --text STR] [--max-repeat 0.5] [--max-chars N] [--ngram 3] [--json]
                (the DEGENERATION/VERBOSITY WITNESS: judge the SHAPE of a candidate
                 answer or tool result — how repetitive (looping) and how long
                 (runaway) it is — against your thresholds. The graded consumer dual
                 of the context-MMU's write-time repeat-admit rung. Reads stdin on
                 "-" or no source. Exit 1 when degenerate, so it gates a pipeline)
  fak doctor    [--text - | --file PATH | --text STR] [--max-repeat 0.5] [--max-chars N] [--ngram 3] [--json]
                (the OPERATOR DIAGNOSTIC: run the answer-shape witness over a text and
                 cross-check the real kernel admit verdict (would the context-MMU
                 quarantine it?), then RECOMMEND what to do about each finding. Exit 1
                 on any finding. The fak analogue of 'dos doctor')
  fak recall    [--dir DIR] [--out recall-report.json] [--query STR]
                (persist a finished session as a core dump, reload it in a FRESH
                 store, and demonstrate the quarantine surviving the boundary)
  fak dream     [--dir DIR] [--out-dir DIR] [--out dream-report.json]
                (offline "sleep" pass over a core image: re-screen, pre-seal
                 refuted witnesses, repair descriptors, surface duplicate aliases,
                 and write a pruned cleaned image)
  fak memory    drivers | explain | run  [--driver NAME] [--query-file PLAN.json]
                [--intent STR] [--k N] [--budget BYTES] [--dir IMAGE] [--apply]
                (the MEMORY-OPERATION ALGEBRA — build SQL, not a specific query: an
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
                 --cmd html emits a self-contained static HTML inspection report — the
                 shareable artifact a teammate opens in a browser)
  fak serve     [--addr 127.0.0.1:8080 | --stdio]
                [--provider openai|anthropic|gemini|xai --base-url URL --model M --api-key-env VAR]
                [--engine inkernel] [--gguf FILE] [--policy FILE] [--policy-check] [--require-key-env VAR] [--vdso=true]
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
                 fak_context_change) over stdin/stdout)
  fak guard     [--provider anthropic|openai|gemini|xai] [--base-url URL] [--policy FILE]
                [--api-key-env VAR] [--env VAR] [--audit FILE|off] [--no-audit] [--dump-policy] [--quiet] -- <agent command...>
                (RUN YOUR REAL AGENT THROUGH THE KERNEL: the one-command front door.
                 Starts the gateway in-process on a private loopback port, injects its
                 URL into the CHILD only (never your shell), execs the agent, and on
                 exit prints what the kernel allowed vs blocked. Default upstream is the
                 real Anthropic API in passthrough mode, so 'fak guard -- claude' wraps
                 your normal Claude Code — your key + prompt cache flow through, every
                 proposed tool call crosses the capability floor first. Every verdict is
                 appended to a durable, tamper-evident DECISION JOURNAL by default
                 (--audit FILE to relocate, --no-audit to turn off; replay with
                 'fak audit verify'). --dump-policy prints the built-in floor to edit;
                 --policy FILE enforces your own)
  fak audit     verify <journal.jsonl> | export <journal.jsonl>
                (the AUDIT-TRAIL consumer: 'verify' re-reads a decision journal (the
                 'fak guard' / FAK_AUDIT_JOURNAL trail) and validates its hash chain
                 end to end — exit 1 naming the first broken link if a byte changed
                 since it was written; 'export' re-emits it as JSONL. A self-report is
                 not a witness — this is how the record is checked offline)
  fak headroom  list | status | compress [--via NAME] [--model ID] [--emit] [FILE|-]
                (the CONTEXT-COMPRESSION seam: shrink tool outputs/logs/files before
                 they reach the model, reversibly. A pluggable AREA — one generic
                 Compressor interface, swappable plugins: noop (off default), native
                 (in-process structural, zero deps), headroom (bridge to a running
                 'headroom proxy'). The selected plugin folds into the result path as
                 a ResultAdmitter, so 'fak guard'/'fak serve' compress in-stream.
                 Pick with FAK_COMPRESSOR; 'compress' proves the savings with no model)
  fak hook      < call.json     (spawned-hook decide; the A/B baseline transport)
  fak webbench  describe | eval | compare    (frontier web/browser agent benchmarking)
  fak swebench  describe | eval | compare    (SWE-bench Verified benchmarking)
  fak version

every tool call crosses one in-process syscall boundary: vDSO -> adjudicate ->
pre-flight/grammar -> dispatch -> context-MMU admit.
`)
}

func ctx() context.Context { return context.Background() }

// fak run — replay a trace through the kernel.
func cmdRun(argv []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	trace := fs.String("trace", "", "path to a trace JSON file")
	engineID := fs.String("engine", "inkernel", "engine id (inkernel: the fused in-kernel model; mock; cassette)")
	vdso := fs.Bool("vdso", true, "enable the vDSO fast path")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: built-in DefaultPolicy)")
	_ = fs.Parse(argv)

	if *trace == "" {
		fmt.Fprintln(os.Stderr, "fak run: --trace is required")
		os.Exit(2)
	}
	applyPolicy(*policyPath)
	t, err := bench.LoadTrace(*trace)
	must(err)
	k := kernel.New(*engineID)
	k.SetVDSO(*vdso)
	res := abi.ActiveResolver()
	for i, c := range t.Calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx(), args)
		must(err)
		tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
		r, v := k.Syscall(ctx(), tc)
		fmt.Printf("[%2d] %-28s verdict=%-9s by=%-9s status=%v\n",
			i, c.Tool, verdictName(v.Kind), v.By, statusName(r.Status))
	}
	cc := k.Counters()
	fmt.Printf("\nsummary: submits=%d vdso_hits=%d engine_calls=%d denies=%d transforms=%d quarantines=%d\n",
		cc.Submits, cc.VDSOHits, cc.EngineCalls, cc.Denies, cc.Transforms, cc.Quarantines)
}

// fak preflight — run only the pre-flight/grammar rungs over one call.
func cmdPreflight(argv []string) {
	fs := flag.NewFlagSet("preflight", flag.ExitOnError)
	tool := fs.String("tool", "", "tool name")
	args := fs.String("args", "{}", "tool args as JSON")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: built-in DefaultPolicy)")
	explain := fs.Bool("explain", false, "print the full decision trace: every rung folded, what each returned, which won, and why")
	asJSON := fs.Bool("json", false, "emit the decision trace as JSON (safe to log: args digest only, never raw args)")
	_ = fs.Parse(argv)
	if *tool == "" {
		fmt.Fprintln(os.Stderr, "fak preflight: --tool is required")
		os.Exit(2)
	}
	applyPolicy(*policyPath)
	res := abi.ActiveResolver()
	ref, err := res.Put(ctx(), []byte(*args))
	must(err)
	tc := &abi.ToolCall{Tool: *tool, Args: ref}
	// --explain/--json fold the SAME chain to the SAME verdict but additionally
	// surface the per-rung Decision trace (the eight rungs preflight actually folds
	// are invisible in the default one-liner). Default output is unchanged.
	if *explain || *asJSON {
		_, d := kernel.FoldExplain(ctx(), abi.AdjudicatorsFor(tc), tc)
		if *asJSON {
			fmt.Println(d.JSON())
		} else {
			fmt.Print(d.Text())
		}
		return
	}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)
	fmt.Printf("verdict=%s reason=%s by=%s\n", verdictName(v.Kind), abi.ReasonName(v.Reason), v.By)
}

// fak bench — A/B ablate the vDSO over a frozen trace.
func cmdBench(argv []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	suite := fs.String("suite", "tau2-smoke", "trace suite name (under testdata/tau2)")
	out := fs.String("out", "report.json", "report output path")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	baselineN := fs.Int("baseline-n", 30, "spawned-hook baseline samples")
	noBaseline := fs.Bool("no-baseline", false, "skip the spawned baseline (RED gate)")
	_ = fs.Parse(argv)

	path := *tracePath
	if path == "" {
		path = filepath.Join(traceDir(), *suite+".json")
	}
	t, err := bench.LoadTrace(path)
	must(err)

	opt := bench.Options{EngineID: "mock", EngineModel: "mock-offline", BaselineN: *baselineN}
	if !*noBaseline {
		if bin, err := os.Executable(); err == nil {
			opt.BinPath = bin
		}
	}
	rep, err := bench.Run(ctx(), t, opt)
	must(err)

	must(os.WriteFile(*out, rep.JSON(), 0o644))
	// also dump the standalone baseline witness (unit 23)
	if rep.Baseline.Calls > 0 {
		bj, _ := json.MarshalIndent(map[string]any{
			"source": rep.Baseline.Source, "p50_ns": rep.Baseline.P50Ns,
			"p50_ms": float64(rep.Baseline.P50Ns) / 1e6, "calls": rep.Baseline.Calls,
		}, "", "  ")
		_ = os.WriteFile(filepath.Join(filepath.Dir(*out), "baseline.json"), bj, 0o644)
	}

	printReport(rep, *out)
}

// fak turntax — the TURN-TAX A/B. Replays a class-labeled trace through the real
// kernel and prices the extra error-code MODEL TURNS the SOTA baseline fires
// (malformed args, duplicate read, poison) that fak's 1-shot path deletes —
// keeping the deterministic safety floor on its own axis.
func cmdTurnTax(argv []string) {
	fs := flag.NewFlagSet("turntax", flag.ExitOnError)
	suite := fs.String("suite", "turntax-airline", "trace suite under testdata/turntax")
	out := fs.String("out", "turntax-report.json", "report output path")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	breakeven := fs.Bool("breakeven", false, "emit the hit-rate -> turns-saved -> amortization curve (turntax-breakeven.json) instead of a single-trace report")
	trials := fs.Int("trials", 200, "trials per hit-rate point (--breakeven only)")
	seed := fs.Int64("seed", 0x7A8BC0DE, "RNG seed for the break-even sweep (--breakeven only)")
	cm := turnbench.DefaultCostModel()
	promptTok := fs.Int("prompt-tokens", cm.PromptTokensPerTurn, "model prompt tokens per turn (cost knob)")
	complTok := fs.Int("completion-tokens", cm.CompletionTokensPerTurn, "model completion tokens per turn (cost knob)")
	latencyMs := fs.Float64("turn-latency-ms", cm.ModelTurnLatencyMs, "model round-trip latency per turn, ms (cost knob)")
	_ = fs.Parse(argv)

	cm.PromptTokensPerTurn = *promptTok
	cm.CompletionTokensPerTurn = *complTok
	cm.ModelTurnLatencyMs = *latencyMs

	if *breakeven {
		bePath := *out
		if bePath == "turntax-report.json" { // user kept the default; pick the break-even default
			bePath = "turntax-breakeven.json"
		}
		rep := turnbench.RunBreakEvenSweep(ctx(), turnbench.BaseTrace(), nil, nil, cm, *trials, *seed)
		must(os.WriteFile(bePath, rep.JSON(), 0o644))
		printBreakEven(os.Stdout, &rep)
		fmt.Printf("\nbreak-even curve written    : %s\n", bePath)
		return
	}

	path := *tracePath
	if path == "" {
		path = filepath.Join(turnTaxDir(), *suite+".json")
	}
	t, err := turnbench.LoadTrace(path)
	must(err)

	rep, err := turnbench.Run(ctx(), t, cm)
	must(err)

	must(os.WriteFile(*out, rep.JSON(), 0o644))
	turnbench.PrintReport(os.Stdout, rep)
	fmt.Printf("\nreport written              : %s\n", *out)
}

// printBreakEven renders the hit-rate curve as an operator-readable table: the
// expected turns saved per session at each addressable rate, the priced net, and
// the §3.1 self-host amortization (the real ~0.7% row is the honest answer to "is
// 9 cherry-picked").
func printBreakEven(w io.Writer, r *turnbench.BreakEvenReport) {
	fmt.Fprintf(w, "== fak turntax break-even: %s  (base %d calls, %d trials/point, hash %s) ==\n",
		r.SliceID, r.BaseCalls, r.Trials, r.BaseHash)
	fmt.Fprintf(w, "real-world addressable hit-rate (TURN-TAX §3.1, tau2-airline): %.1f%%\n\n", r.RealWorldHitRate*100)
	fmt.Fprintf(w, "%6s  %11s  %4s  %4s  %9s  %9s  %8s  %16s\n",
		"h", "mean_turns", "p50", "p90", "tok/sess", "$/sess", "lat(s)", "self-host_sess")
	for _, p := range r.Points {
		self := "n/a"
		for _, rg := range p.Regimes {
			if rg.Name == "self-host-fork" {
				if rg.SessionsToBreakEven == turnbench.NeverAmortizes {
					self = "never"
				} else {
					self = fmt.Sprintf("%.0f", rg.SessionsToBreakEven)
				}
			}
		}
		mark := ""
		if p.HitRate == r.RealWorldHitRate {
			mark = "  <- real-world rate"
		}
		fmt.Fprintf(w, "%6.3f  %11.4f  %4d  %4d  %9.0f  %9.5f  %8.2f  %16s%s\n",
			p.HitRate, p.MeanTurnsSaved, p.TurnsSaved.P50, p.TurnsSaved.P90,
			p.TokensSavedMean, p.DollarsSavedMean, p.LatencySavedSecMean, self, mark)
	}
	fmt.Fprintf(w, "\nThe turn-saving is small at the real ~%.1f%% rate (the safety floor is the reason to run the kernel there);\n", r.RealWorldHitRate*100)
	fmt.Fprintln(w, "it only becomes large in error/dup-rich regimes. The airline demo slice (9/14) is the far high end of this curve.")
}

// fak agent — the LIVE agentic loop. A real model (or the offline mock) drives a
// multi-turn tool-calling conversation TWICE over the same task: once with every
// tool call mediated by the in-process kernel (fak arm), once naive (the "now"
// baseline). It reports turns, tokens, in-syscall repairs, vDSO dedup hits,
// adjudicator denies, and MMU quarantines for each arm — the real turn-use-vs-now
// measurement the static bench could not produce.
func cmdAgent(argv []string) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)
	task := fs.String("task", agent.DefaultTask, "the user task the agent must complete")
	provider := fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	baseURL := fs.String("base-url", "", "provider base URL (OpenAI-compatible: .../v1; Gemini native: .../v1beta; Anthropic native: https://api.anthropic.com)")
	model := fs.String("model", "gemini-2.5-flash", "model id")
	apiKeyEnv := fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	offline := fs.Bool("offline", false, "use the deterministic mock planner (no network)")
	maxTurns := fs.Int("max-turns", 10, "max model turns per arm")
	out := fs.String("out", "agent-report.json", "report output path")
	logOut := fs.String("log", "", "optional path to write the per-call trace log")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: built-in DefaultPolicy)")
	_ = fs.Parse(argv)
	applyPolicy(*policyPath)

	var planner agent.Planner
	if *offline || *baseURL == "" {
		if !*offline {
			fmt.Fprintln(os.Stderr, "fak agent: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		planner = agent.NewMockPlanner(*model)
	} else {
		key := os.Getenv(*apiKeyEnv)
		if key == "" {
			// A local endpoint (e.g. the transformers shim) needs no key; a remote
			// one will return 401, which the planner surfaces clearly. Warn, proceed.
			fmt.Fprintf(os.Stderr, "fak agent: env %s is empty — proceeding with no auth header (fine for a local endpoint)\n", *apiKeyEnv)
		}
		p, err := agent.NewProviderHTTPPlanner(*provider, *baseURL, *model, key)
		must(err)
		planner = p
	}

	res, trace, err := agent.Run(ctx(), planner, *task, *maxTurns)
	must(err)

	must(os.WriteFile(*out, jsonIndent(res), 0o644))
	if *logOut != "" {
		_ = os.WriteFile(*logOut, agent.RenderTrace(trace), 0o644)
	}
	agent.PrintReport(os.Stdout, res, trace, *out)
}

func jsonIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}

// applyPolicy loads a capability-floor manifest and swaps it into the registered
// adjudicator before the kernel runs. Empty path = keep the built-in
// DefaultPolicy. A bad manifest is fatal: a misconfigured floor must fail loudly
// at startup, never silently fall back to a more permissive default.
func applyPolicy(path string) {
	if path == "" {
		return
	}
	_, err := reloadPolicy(path)
	must(err)
	fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s\n", path)
}

func reloadPolicy(path string) (policy.Runtime, error) {
	if path == "" {
		return policy.Runtime{}, errors.New("policy reload requires --policy FILE")
	}
	rt, err := policy.LoadRuntime(path)
	if err != nil {
		return policy.Runtime{}, err
	}
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	return rt, nil
}

func policyReloader(path string) gateway.PolicyReloadFunc {
	if path == "" {
		return nil
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, err := reloadPolicy(path)
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		return gateway.PolicyReloadResponse{
			Reloaded: true,
			Source:   path,
			Summary:  policy.SummaryRuntime(rt),
		}, nil
	}
}

func resetTrace(_ context.Context, traceID string) error {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return errors.New("trace_id is required")
	}
	ifc.Default.Reset(traceID)
	return nil
}

// observeTrace is the read-only complement of resetTrace (#411): it reports the
// live IFC taint high-water mark for one trace so the gateway can serve
// GET /v1/fak/trace/{trace_id} without importing IFC internals. An unseen trace
// reads "trusted" — the ledger's own clean default.
func observeTrace(_ context.Context, traceID string) (string, bool) {
	lvl := ifc.Default.Level(strings.TrimSpace(traceID))
	return taintLevelName(lvl), ifc.Dangerous(lvl)
}

// taintLevelName renders an abi.TaintLabel as its stable wire name. It mirrors
// ifc's unexported taintName (the enum is not ordered by restrictiveness, so it is
// switched, never formatted).
func taintLevelName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return "unknown"
}

func applyRuntime(rt policy.Runtime) {
	policy.ApplySources(rt)
	ifc.ConfigureDefaultPolicy(ifcPolicy(rt))
}

func ifcPolicy(rt policy.Runtime) ifc.Policy {
	p := ifc.Policy{}
	if len(rt.SafeSinks) > 0 {
		p.SafeSinks = make(map[string]bool, len(rt.SafeSinks))
		for _, tool := range rt.SafeSinks {
			p.SafeSinks[tool] = true
		}
	}
	type rule struct {
		tool string
		sink ifc.SinkClass
	}
	rules := make([]rule, 0, len(rt.AuthorizeRules))
	for _, r := range rt.AuthorizeRules {
		rules = append(rules, rule{tool: r.Tool, sink: sinkClass(r.Sink)})
	}
	if len(rules) > 0 {
		p.Authorize = func(c *abi.ToolCall, into ifc.SinkClass) bool {
			if c == nil {
				return false
			}
			for _, r := range rules {
				if c.Tool == r.tool && into == r.sink {
					return true
				}
			}
			return false
		}
	}
	return p
}

func sinkClass(name string) ifc.SinkClass {
	switch strings.ToUpper(strings.TrimSpace(name)) {
	case "EGRESS":
		return ifc.SinkEgress
	case "EXEC":
		return ifc.SinkExec
	case "DESTRUCTIVE":
		return ifc.SinkDestructive
	default:
		return ifc.SinkNone
	}
}

// fak policy — author and validate the deployable capability floor. --dump emits
// the built-in DefaultPolicy as a manifest (the starting point an adopter edits);
// --check validates a manifest against the closed refusal vocabulary and prints
// the floor it admits, so a misconfigured policy is caught BEFORE it gates a run.
func cmdPolicy(argv []string) {
	fs := flag.NewFlagSet("policy", flag.ExitOnError)
	dump := fs.Bool("dump", false, "write the built-in DefaultPolicy as a manifest to stdout")
	check := fs.String("check", "", "validate a manifest file and print the floor it admits")
	_ = fs.Parse(argv)

	switch {
	case *dump:
		os.Stdout.Write(policy.FromPolicy(adjudicator.DefaultPolicy()).JSON())
	case *check != "":
		rt, err := policy.LoadRuntime(*check)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak policy:", err)
			os.Exit(1)
		}
		fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", *check, policy.SummaryRuntime(rt))
	default:
		fmt.Fprintln(os.Stderr, "fak policy: pass --dump (emit the default manifest) or --check FILE (validate one)")
		os.Exit(2)
	}
}

// fak lint — the STATIC tool linter. The kernel never trusts a tool's self-declared
// annotations: it re-checks them every call and silently does the safe thing (the
// vDSO overrides a lying readOnlyHint from the name, pre-flight re-validates args).
// This verb is the definition-time DUAL of those call-time re-checks: it runs once
// over the configured tool surface and says OUT LOUD what the runtime would only
// ever whisper to itself — a dead cache hint, an unreachable pure registration, a
// canned answer for a write-shaped tool, a schema the model is shown but the kernel
// never enforces. Exit 1 on an error-severity finding (or any finding with
// --strict), so it can gate a build.
func cmdLint(argv []string) {
	fs := flag.NewFlagSet("lint", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	strict := fs.Bool("strict", false, "exit non-zero on ANY finding (info/warn too), not just errors")
	kernelOnly := fs.Bool("kernel-only", false, "lint only the kernel registries (skip the agent hint classifier + model-facing catalog)")
	_ = fs.Parse(argv)

	var facts []toollint.ToolFacts
	if *kernelOnly {
		facts = toollint.FromKernel()
	} else {
		agent.Configure() // register the agent's schemas, grammar, and engine first
		facts = agent.LintFacts()
	}
	rep := toollint.Lint(facts)

	if *asJSON {
		type jf struct {
			Code      string `json:"code"`
			Severity  string `json:"severity"`
			Tool      string `json:"tool"`
			Message   string `json:"message"`
			Mechanism string `json:"mechanism"`
		}
		rows := make([]jf, 0, len(rep.Findings))
		for _, f := range rep.Findings {
			rows = append(rows, jf{string(f.Code), f.Severity.String(), f.Tool, f.Message, f.Mechanism})
		}
		b, _ := json.MarshalIndent(map[string]any{
			"tools":    len(facts),
			"findings": rows,
			"errors":   rep.Errors(),
			"warnings": rep.Warnings(),
			"infos":    rep.Infos(),
		}, "", "  ")
		fmt.Println(string(b))
	} else {
		for _, f := range rep.Findings {
			fmt.Printf("%s  %-5s  %-22s  %s\n          %s\n", f.Code, f.Severity.String(), f.Tool, f.Message, f.Mechanism)
		}
		if rep.Clean() {
			fmt.Printf("lint clean: %d tool(s), no findings\n", len(facts))
		} else {
			fmt.Printf("\n%d tool(s): %d error, %d warn, %d info\n", len(facts), rep.Errors(), rep.Warnings(), rep.Infos())
		}
	}

	if code := lintExitCode(rep, *strict); code != 0 {
		os.Exit(code)
	}
}

// lintExitCode is the PURE exit-code contract for `fak lint`, factored out so it is
// unit-testable without os.Exit: 1 on any error-severity finding, or — under
// --strict — on ANY finding at all (the "gate a build on a clean surface" mode the
// help text and cmdLint doc both promise). 0 otherwise.
func lintExitCode(rep toollint.Report, strict bool) int {
	if rep.Errors() > 0 || (strict && !rep.Clean()) {
		return 1
	}
	return 0
}

// fak serve — the GATEWAY. It fronts the kernel over an OpenAI-compatible HTTP
// surface and MCP so an agent in ANY language can route its tool calls through the
// in-process syscall boundary without writing Go. The gateway is Go and ON the
// request path (it adjudicates) — in-direction; non-Go CLIENTS live in the
// adopter's repo. Construction mirrors cmdAgent: registrations is already imported
// (so the resolver + full adjudicator chain are wired), the capability floor is
// installed fail-loud, and the kernel is built bound to a registered engine.
// resolveRequiredKey resolves a secret the operator REQUIRED by naming an env
// var via a --…-key-env flag. When the flag is unset (empty name) auth was not
// requested, so it returns ok=true with an empty key. But when the flag names an
// env var that is unset or empty, it returns ok=false: the operator asked for
// auth and the secret did not land (typo, un-propagated CI env, k8s Secret
// mis-mount, pod restarted without it). For an agent kernel the safe
// default is to fail CLOSED — refuse to start — not to warn and silently serve
// unauthenticated. The lookup is injected so the decision is unit-testable
// without touching process env. (issue #213-class fail-open fix; see #255.)
func resolveRequiredKey(envName string, lookup func(string) string) (key string, ok bool) {
	if envName == "" {
		return "", true // flag not set: auth not requested.
	}
	v := lookup(envName)
	if v == "" {
		return "", false // requested but missing: caller must fail closed.
	}
	return v, true
}

// fak hook — the spawned-hook decide transport (A/B baseline). Reads one call
// from stdin, folds the adjudicator chain, writes the verdict to stdout.
func cmdHook() {
	var c bench.Call
	if err := json.NewDecoder(os.Stdin).Decode(&c); err != nil {
		// an empty/invalid call still exercises the spawn+decide path
		c = bench.Call{Tool: "noop"}
	}
	res := abi.ActiveResolver()
	args := []byte(c.Args)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, _ := res.Put(ctx(), args)
	tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"kind": verdictName(v.Kind), "reason": abi.ReasonName(v.Reason), "by": v.By,
	})
}

func printReport(rep *metrics.Report, path string) {
	fmt.Printf("== fak bench: %s ==\n", rep.Provenance.SliceID)
	fmt.Printf("in-process adjudication p50 : %d ns\n", rep.On.P50Ns)
	fmt.Printf("spawned-hook        p50     : %d ns (%.3f ms, n=%d)\n",
		rep.Baseline.P50Ns, float64(rep.Baseline.P50Ns)/1e6, rep.Baseline.Calls)
	if rep.Baseline.P50Ns > 0 && rep.On.P50Ns > 0 {
		fmt.Printf("fusion speedup (p50)        : %.0fx\n", float64(rep.Baseline.P50Ns)/float64(rep.On.P50Ns))
	}
	fmt.Printf("PRIMARY GATE                : %s  (%s)\n", rep.GatePrimary, rep.PrimaryDetail)
	fmt.Printf("secondary token delta       : %.2f%% (soft, never gates)\n", rep.TokenDeltaPct)
	fmt.Printf("vdso hit-rate               : %.3f   pollution-rate: %.3f\n",
		rep.KPIs.VDSOHitRate, rep.KPIs.ContextPollutionRate)
	fmt.Printf("workload hash               : %s   live seam: %s\n",
		rep.Provenance.WorkloadHash, rep.LiveSeam)
	fmt.Printf("report written              : %s\n", path)
}

func traceDir() string {
	// testdata sits next to the module root; resolve relative to cwd first then
	// the executable dir.
	if _, err := os.Stat(filepath.Join("testdata", "tau2")); err == nil {
		return filepath.Join("testdata", "tau2")
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Join(filepath.Dir(exe), "testdata", "tau2")
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return filepath.Join("testdata", "tau2")
}

func turnTaxDir() string {
	if _, err := os.Stat(filepath.Join("testdata", "turntax")); err == nil {
		return filepath.Join("testdata", "turntax")
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Join(filepath.Dir(exe), "testdata", "turntax")
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return filepath.Join("testdata", "turntax")
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	}
	return "K" + strconv.Itoa(int(k))
}

func statusName(s abi.Status) string {
	switch s {
	case abi.StatusOK:
		return "OK"
	case abi.StatusError:
		return "ERR"
	case abi.StatusPending:
		return "PEND"
	}
	return "?"
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak:", err)
		os.Exit(1)
	}
}

// embeddedGGUFTokenizer builds a tokenizer straight from the GGUF's own
// tokenizer.ggml.* metadata, mirroring cmd/simpledemo's embedded path. It lets
// `fak serve --gguf` serve real in-kernel chat without a separate tokenizer.json.
// Returns an error (not a panic) when the checkpoint embeds no usable BPE tokenizer,
// so the caller can fall back to the MockPlanner instead of aborting startup.
func embeddedGGUFTokenizer(ggufPath string) (*tokenizer.Tokenizer, error) {
	f, err := ggufload.Open(ggufPath)
	if err != nil {
		return nil, err
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		return nil, fmt.Errorf("no embedded BPE tokenizer in %s", filepath.Base(ggufPath))
	}
	return tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
}
