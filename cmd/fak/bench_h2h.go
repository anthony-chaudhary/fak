package main

// bench_h2h.go — `fak bench h2h`: the engine head-to-head. `run` replays one
// seeded workload (cold prefill at several prompt sizes, long decode, and a
// growing multi-turn conversation) against every --arm, interleaved, and appends
// one row per request to the h2h ledger beside the gateway perf ledger. `report`
// folds the latest (or a named) run and lists every cell where a fak arm trails
// the best rival; --json is the agent-readable form. The workload, fold and
// fairness rules live in internal/h2hbench.

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
       fak bench h2h report [--run ID] [--json]

Arms whose name starts with "fak" are graded against the best other arm.
Rows land in `+h2hbench.DefaultLedgerRel+` (override with --ledger).`)
		return 2
	}
	switch argv[0] {
	case "run":
		return runBenchH2HRun(stdout, stderr, argv[1:])
	case "report":
		return runBenchH2HReport(stdout, stderr, argv[1:])
	}
	fmt.Fprintf(stderr, "fak bench h2h: unknown subcommand %q (want run|report)\n", argv[0])
	return 2
}

func h2hLedgerPath(flagPath string) string {
	if p := strings.TrimSpace(flagPath); p != "" {
		return p
	}
	return nightrunLedgerPath(h2hbench.DefaultLedgerRel)
}

func runBenchH2HRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var arms multiFlag
	fs.Var(&arms, "arm", "name=OpenAI-compatible /v1 base URL (repeatable; names starting with \"fak\" are the fak arms)")
	model := fs.String("model", "", "model id sent to every arm (required)")
	host := fs.String("host", "", "machine label recorded on every row (default: hostname)")
	runID := fs.String("run-id", "", "run id (default: UTC timestamp)")
	cold := fs.String("cold", "512,2048,8192", "comma-separated cold prompt sizes in tokens (empty skips)")
	reps := fs.Int("reps", 3, "repetitions per cold size, decode run, and multi-turn conversation")
	decode := fs.Int("decode", 256, "max tokens for the decode scenario (0 skips)")
	turns := fs.Int("turns", 6, "multi-turn conversation length (0 skips)")
	system := fs.Int("system", 4096, "multi-turn shared system prefix size in tokens")
	turnTokens := fs.Int("turn-tokens", 256, "new user tokens per multi-turn turn")
	answer := fs.Int("answer", 16, "max tokens for cold and multi-turn answers")
	timeout := fs.Duration("timeout", 10*time.Minute, "per-request timeout")
	keepThinking := fs.Bool("keep-thinking", false, "do not send chat_template_kwargs.enable_thinking=false")
	slots := fs.String("slots-url", "", "llama-server /slots URL on the shared upstream: rows taken while another client held a slot are flagged and kept out of the medians")
	gpuBusy := fs.String("gpu-busy", "auto", "amdgpu gpu_busy_percent file sampled before each request to flag a GPU shared with another process; auto = first card found, off = never")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding a bearer token sent to every arm")
	ledger := fs.String("ledger", "", "JSONL ledger path, or \"off\" (default: "+h2hbench.DefaultLedgerRel+")")
	jsonOut := fs.Bool("json", false, "print the folded report as JSON when done")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *model == "" || len(arms) == 0 {
		fmt.Fprintln(stderr, "fak bench h2h run: --model and at least one --arm are required")
		return 2
	}
	cfg := h2hbench.Config{
		RunID: *runID, Host: *host, Model: *model, Reps: *reps, DecodeTokens: *decode, Turns: *turns,
		SystemTokens: *system, TurnTokens: *turnTokens, AnswerTokens: *answer, Timeout: *timeout,
		Client: &http.Client{}, DisableThink: !*keepThinking, Progress: stderr, SlotsURL: *slots,
	}
	if cfg.RunID == "" {
		cfg.RunID = time.Now().UTC().Format("20060102T150405Z")
	}
	if cfg.Host == "" {
		cfg.Host, _ = os.Hostname()
	}
	switch strings.ToLower(strings.TrimSpace(*gpuBusy)) {
	case "auto":
		cfg.GPUBusyPath = h2hbench.GPUBusyAuto()
	case "off", "":
	default:
		cfg.GPUBusyPath = *gpuBusy
	}
	key := ""
	if *apiKeyEnv != "" {
		key = os.Getenv(*apiKeyEnv)
	}
	for _, a := range arms {
		arm, err := h2hbench.ParseArm(a)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h run:", err)
			return 2
		}
		arm.APIKey = key
		cfg.Arms = append(cfg.Arms, arm)
	}
	if strings.TrimSpace(*cold) != "" {
		sizes, err := parsePositiveIntList(*cold)
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
	rows, err := h2hbench.Run(ctx, cfg)
	if err != nil {
		fmt.Fprintln(stderr, "fak bench h2h run:", err)
	}
	rep := h2hbench.Summarize(rows)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(rep)
	} else {
		h2hbench.Render(stdout, rep)
	}
	if err != nil {
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
