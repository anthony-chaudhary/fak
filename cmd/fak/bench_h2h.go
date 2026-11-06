package main

// bench_h2h.go — `fak bench h2h`: the engine head-to-head. `run` replays one
// seeded workload (cold prefill at several prompt sizes, long decode, and a
// growing multi-turn conversation) against every --arm, interleaved, and appends
// one row per request to the h2h ledger beside the gateway perf ledger. `report`
// folds the latest (or a named) run and lists every cell where a fak arm trails
// the best rival; --json is the agent-readable form. The workload, fold and
// fairness rules live in internal/h2hbench. The bring-up helpers serve-wait,
// plan and gpu-idle are implemented in bench_h2h_bringup.go.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/h2hbench"
)

func runBenchH2H(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help" {
		fmt.Fprintln(stderr, `usage: fak bench h2h run --arm fak=http://127.0.0.1:8080/v1 --arm llama=http://127.0.0.1:8091/v1 --model M [flags]
       fak bench h2h serve-wait --arm fak=http://127.0.0.1:8080/v1 [--model M] [--timeout 30m] [--interval 5s] [--json]
       fak bench h2h plan --model M --arm fak=http://127.0.0.1:8080/v1 [--ledger-dir DIR] [--json]
       fak bench h2h gpu-idle [--require N] [--max-util-pct 5] [--max-mem-used-mib 1024] [--interval 2s] [--timeout 60s] [--json]
       fak bench h2h report [--run ID] [--json]

Arms whose name starts with "fak" are graded against the best other arm.
Rows land in `+h2hbench.DefaultLedgerRel+` (override with --ledger).`)
		return 2
	}
	switch argv[0] {
	case "run":
		return runBenchH2HRun(stdout, stderr, argv[1:])
	case "serve-wait":
		return runBenchH2HServeWait(stdout, stderr, argv[1:])
	case "plan":
		return runBenchH2HPlan(stdout, stderr, argv[1:])
	case "gpu-idle":
		return runBenchH2HGPUIde(stdout, stderr, argv[1:])
	case "report":
		return runBenchH2HReport(stdout, stderr, argv[1:])
	}
	fmt.Fprintf(stderr, "fak bench h2h: unknown subcommand %q (want run|serve-wait|plan|gpu-idle|report)\n", argv[0])
	return 2
}

func h2hLedgerPath(flagPath string) string {
	if p := strings.TrimSpace(flagPath); p != "" {
		return p
	}
	return nightrunLedgerPath(h2hbench.DefaultLedgerRel)
}

// h2hRunFlags is the workload spec shared by `run` and `plan`: registering the
// flags once keeps plan's emitted argvs from drifting out of what run accepts.
type h2hRunFlags struct {
	arms         multiFlag
	model        *string
	host         *string
	runID        *string
	speculative  *string
	cold         *string
	reps         *int
	decode       *int
	turns        *int
	system       *int
	turnTokens   *int
	answer       *int
	concurrency  *int
	concList     *string
	timeout      *time.Duration
	keepThinking *bool
	slotsURL     *string
	gpuBusy      *string
	apiKeyEnv    *string
}

// registerH2HRunFlags binds the shared h2h run workload flags onto fs and returns
// their values. --ledger and --json stay with the individual verbs. It returns a
// pointer because the repeatable --arm flag appends into the struct's own storage
// during fs.Parse, which outlives this call.
func registerH2HRunFlags(fs *flag.FlagSet) *h2hRunFlags {
	f := &h2hRunFlags{}
	fs.Var(&f.arms, "arm", "name=OpenAI-compatible /v1 base URL (repeatable; names starting with \"fak\" are the fak arms)")
	f.model = fs.String("model", "", "model id sent to every arm (required)")
	f.host = fs.String("host", "", "machine label recorded on every row (default: hostname)")
	f.runID = fs.String("run-id", "", "run id (default: UTC timestamp)")
	f.speculative = fs.String("speculative", "0,2", "requested draft-cap matrix (default 0,2); server-default omits control; requires a server configured with a drafter; execution evidence may remain unknown")
	f.cold = fs.String("cold", "512,2048,8192", "comma-separated cold prompt sizes in tokens (empty skips)")
	f.reps = fs.Int("reps", 3, "repetitions per cold size, decode run, and multi-turn conversation")
	f.decode = fs.Int("decode", 256, "max tokens for the decode scenario (0 skips)")
	f.turns = fs.Int("turns", 6, "multi-turn conversation length (0 skips)")
	f.system = fs.Int("system", 4096, "multi-turn shared system prefix size in tokens")
	f.turnTokens = fs.Int("turn-tokens", 256, "new user tokens per multi-turn turn")
	f.answer = fs.Int("answer", 16, "max tokens for cold and multi-turn answers")
	f.concurrency = fs.Int("concurrency", 1, "in-flight client streams replaying the full scenario sequence (1 = sequential)")
	f.concList = fs.String("concurrency-list", "", "comma-separated concurrency grid run in one invocation, e.g. 1,2,4,8 (overrides --concurrency)")
	f.timeout = fs.Duration("timeout", 10*time.Minute, "per-request timeout")
	f.keepThinking = fs.Bool("keep-thinking", false, "do not send chat_template_kwargs.enable_thinking=false")
	f.slotsURL = fs.String("slots-url", "", "llama-server /slots URL on the shared upstream: rows taken while another client held a slot are flagged and kept out of the medians")
	f.gpuBusy = fs.String("gpu-busy", "auto", "amdgpu gpu_busy_percent file sampled before each request to flag a GPU shared with another process; auto = first card found, off = never")
	f.apiKeyEnv = fs.String("api-key-env", "", "env var holding a bearer token sent to every arm")
	return f
}

func runBenchH2HRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f := registerH2HRunFlags(fs)
	ledger := fs.String("ledger", "", "JSONL ledger path, or \"off\" (default: "+h2hbench.DefaultLedgerRel+")")
	jsonOut := fs.Bool("json", false, "print the folded report as JSON when done")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *f.model == "" || len(f.arms) == 0 {
		fmt.Fprintln(stderr, "fak bench h2h run: --model and at least one --arm are required")
		return 2
	}
	modes, err := h2hbench.ParseSpeculativeModes(*f.speculative)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	cfg := h2hbench.Config{
		SpeculativeModes: modes,
		RunID:            *f.runID, Host: *f.host, Model: *f.model, Reps: *f.reps, DecodeTokens: *f.decode, Turns: *f.turns,
		SystemTokens: *f.system, TurnTokens: *f.turnTokens, AnswerTokens: *f.answer, Timeout: *f.timeout,
		Client: &http.Client{}, DisableThink: !*f.keepThinking, Progress: stderr, SlotsURL: *f.slotsURL,
	}
	if cfg.RunID == "" {
		cfg.RunID = time.Now().UTC().Format("20060102T150405Z")
	}
	if cfg.Host == "" {
		cfg.Host, _ = os.Hostname()
	}
	switch strings.ToLower(strings.TrimSpace(*f.gpuBusy)) {
	case "auto":
		cfg.GPUBusyPath = h2hbench.GPUBusyAuto()
	case "off", "":
	default:
		cfg.GPUBusyPath = *f.gpuBusy
	}
	key := ""
	if *f.apiKeyEnv != "" {
		key = os.Getenv(*f.apiKeyEnv)
	}
	for _, a := range f.arms {
		arm, err := h2hbench.ParseArm(a)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run:", err)
			return 2
		}
		arm.APIKey = key
		cfg.Arms = append(cfg.Arms, arm)
	}
	if strings.TrimSpace(*f.cold) != "" {
		sizes, err := parsePositiveIntList(*f.cold)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run: --cold:", err)
			return 2
		}
		cfg.ColdSizes = sizes
	}
	path := h2hLedgerPath(*ledger)
	if !strings.EqualFold(path, "off") {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run: open ledger:", err)
			return 1
		}
		defer f.Close()
		enc := json.NewEncoder(f)
		cfg.Sink = func(r h2hbench.Row) error { return enc.Encode(r) }
		fmt.Fprintf(stderr, "h2h run %s -> %s\n", cfg.RunID, path)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	concs := []int{*f.concurrency}
	if strings.TrimSpace(*f.concList) != "" {
		cs, err := parsePositiveIntList(*f.concList)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run: --concurrency-list:", err)
			return 2
		}
		concs = cs
	}
	var all []h2hbench.Row
	var runErr error
	for _, c := range concs {
		if err := ctx.Err(); err != nil {
			runErr = err
			break
		}
		cfg.Concurrency = c
		if len(concs) > 1 {
			fmt.Fprintf(stderr, "h2h run %s concurrency=%d\n", cfg.RunID, c)
		}
		rows, err := h2hbench.Run(ctx, cfg)
		all = append(all, rows...)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run:", err)
			runErr = err
			break
		}
	}
	rep := h2hbench.Summarize(all)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(rep)
	} else {
		h2hbench.Render(stdout, rep)
	}
	if runErr != nil {
		return 1
	}
	return 0
}

func runBenchH2HReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "JSONL ledger path (default: "+h2hbench.DefaultLedgerRel+")")
	run := fs.String("run", "", "run id (default: the most recent run in the ledger)")
	jsonOut := fs.Bool("json", false, "emit the folded report as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	rows, err := h2hbench.ReadLedger(h2hLedgerPath(*ledger))
	if err != nil {
		fmt.Fprintln(stderr, "fak bench h2h report:", err)
		return 1
	}
	sel := h2hbench.SelectRun(rows, *run)
	if len(sel) == 0 {
		fmt.Fprintln(stderr, "fak bench h2h report: no rows for that run")
		return 1
	}
	rep := h2hbench.Summarize(sel)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(rep)
		return 0
	}
	h2hbench.Render(stdout, rep)
	return 0
}
