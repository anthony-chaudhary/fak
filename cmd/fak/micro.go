package main

// micro.go — the `fak micro` front door for the native in-process Go microagent
// runtime (#2029, part of epic #2000 M30). It gives the host built in
// internal/microagent (the worker pool + slot scheduler + audit sink, #2002/#2006)
// a CLI verb consistent with `fak serve`/`guard`/`dispatch`, instead of only a
// package a test constructs:
//
//	fak micro            run ONE microagent end-to-end (Mock by default).
//	fak micro host       boot the in-process host (M2) and run a small fleet.
//	fak micro … --dry-run   resolve + print the plan (backends, seats, caps) — no spend.
//
// Config precedence is flags > env (FAK_MICRO_*) > file (--config JSON) > defaults,
// the same precedence the rest of the binary uses.
//
// Generation intent: gen/second-next architectural OPTION (#2002). This verb is the
// EXPLICIT GATE the generation frame requires — nothing in the default
// serve/guard/dispatch path constructs a Host; a human has to type `fak micro`.
//   - Promotion evidence: `fak micro` drives the real microagent.Host lifecycle
//     (Spawn → worker Step loop → retire → Reap) over the shared slot scheduler and
//     one audit sink, on the deterministic Mock planner — the same host the #2002
//     smoke test witnesses at 100+ agents, now reachable by hand. Promote past Mock
//     once the dispatch path can target the host (#2030) and a density measurement
//     (#2033) confirms per-agent process weight was the binding cost.
//   - Demotion / retirement: retire this verb if #2033 shows per-agent cost is
//     dominated by provider seats/rate limits rather than local process weight (the
//     host buys no density), or if the isolation floor demands per-agent OS
//     processes anyway.
//   - Invalidating assumption: the run path uses the Mock planner as the shared
//     gateway seam and drives a per-turn Step agent, because the real internal/agent
//     loop is not yet resumable as an in-process step function (#2001 open) and the
//     kernel gateway + ToolExec floor are not yet wired here (#2030). If the real
//     loop cannot be stepped without per-agent OS state, this front door under-
//     represents the production path and must grow the subprocess ToolExec seam
//     before it carries real agents. The `--admission-*` caps are therefore RESOLVED
//     and reported here but ENFORCED on the served-gateway path (internal/gateway
//     token admission, M19); the Mock run bounds concurrency via the slot pool only.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// microConfig is the resolved plan for a `fak micro` run: the backends, seats,
// admission caps, and fleet shape after applying flags > env > file > defaults.
type microConfig struct {
	Engine                 string `json:"engine"`                   // model engine seam: mock or a running fak gateway.
	Gateway                string `json:"gateway,omitempty"`        // fak serve address for engine=gateway.
	Task                   string `json:"task,omitempty"`           // user task sent identically to every microagent.
	Model                  string `json:"model,omitempty"`          // model requested through the gateway.
	Isolation              string `json:"isolation"`                // ToolExec backend name (M13): goroutine | subprocess.
	Seats                  int    `json:"seats"`                    // slot pool K (M6/M20): max concurrent in-flight model calls; 0 ⇒ derive from workers.
	Workers                int    `json:"workers"`                  // host worker pool K (concurrent Step drivers).
	Queue                  int    `json:"queue"`                    // bounded pending-spawn queue.
	Agents                 int    `json:"agents"`                   // N microagents to run.
	Turns                  int    `json:"turns"`                    // model turns each agent takes before it retires.
	AdmissionMaxConcurrent int    `json:"admission_max_concurrent"` // M19 served-path cap (0 ⇒ unbounded); resolved here, enforced on the gateway.
	AdmissionTokenBudget   int    `json:"admission_token_budget"`   // M19 served-path token budget (0 ⇒ unbounded); resolved here, enforced on the gateway.
}

// defaultMicroConfig is the base layer of the precedence stack. hostMode picks the
// fleet size default: bare `fak micro` runs ONE agent (the "run one" contract),
// `fak micro host` runs a small fleet.
func defaultMicroConfig(hostMode bool) microConfig {
	agents := 1
	if hostMode {
		agents = microagent.DefaultWorkers
	}
	return microConfig{
		Engine:    "mock",
		Isolation: microagent.BackendGoroutine,
		Seats:     0,
		Workers:   microagent.DefaultWorkers,
		Queue:     microagent.DefaultQueue,
		Agents:    agents,
		Turns:     1,
	}
}

// slots is the effective slot-pool size: the configured seat count, or the worker
// count when seats are left at 0.
func (c microConfig) slots() int {
	if c.Seats > 0 {
		return c.Seats
	}
	return c.Workers
}

func cmdMicro(args []string) {
	if len(args) > 0 {
		switch args[0] {
		case "paired":
			cmdMicroPaired(args[1:])
			return
		case "corpus":
			cmdMicroCorpus(args[1:])
			return
		case "tree":
			cmdMicroTree(args[1:])
			return
		}
	}
	if len(args) > 0 && args[0] == "collapse" {
		cmdMicroCollapse(args[1:])
		return
	}
	// `fak micro trace <id>` is the per-agent trace readout (#2031): it renders one
	// microagent's structured timeline (legs, tokens, seat, verdicts) out of the
	// interleaved single-process fleet, either from a persisted --trace-in JSONL or
	// from a fresh deterministic Mock run.
	if len(args) > 0 && args[0] == "trace" {
		cmdMicroTrace(args[1:])
		return
	}
	hostMode := false
	if len(args) > 0 && args[0] == "host" {
		hostMode = true
		args = args[1:]
	}

	fs := flag.NewFlagSet("micro", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var (
		isolation   = fs.String("isolation", "", "ToolExec isolation backend (M13): goroutine | subprocess")
		seats       = fs.Int("seats", 0, "slot pool size (M6/M20): max concurrent in-flight model calls; 0 ⇒ = --workers")
		workers     = fs.Int("workers", 0, "host worker pool K: concurrent Step drivers")
		concurrency = fs.Int("concurrency", 0, "alias for --workers")
		queue       = fs.Int("queue", 0, "bounded pending-spawn queue size")
		nAgents     = fs.Int("n", 0, "number of microagents to run")
		agentsLong  = fs.Int("agents", 0, "alias for -n")
		turns       = fs.Int("turns", 0, "model turns each agent takes before retiring")
		engine      = fs.String("engine", "", "model engine seam (mock or gateway)")
		gateway     = fs.String("gateway", "", "running fak serve address for --engine gateway")
		model       = fs.String("model", "", "model id requested through --engine gateway")
		task        = fs.String("task", "", "user task sent identically to every microagent")
		admMax      = fs.Int("admission-max-concurrent", 0, "M19 provider concurrency cap; 0 ⇒ unbounded (resolved here, enforced on the served gateway)")
		admTok      = fs.Int("admission-token-budget", 0, "M19 provider token budget; 0 ⇒ unbounded (resolved here, enforced on the served gateway)")
		cfgFile     = fs.String("config", "", "load config from a JSON file (lowest non-default precedence)")
		dryRun      = fs.Bool("dry-run", false, "resolve and print the plan (backends, seats, caps) without spending")
		jsonOut     = fs.Bool("json", false, "emit JSON instead of a human-readable report")
		selfcheck   = fs.Bool("selfcheck", false, "run the offline kernel-to-microagent value-chain witness")
		traceOut    = fs.String("trace-out", "", "write per-agent structured traces to a JSONL file for a later `fak micro trace <id> --trace-in` readout (#2031)")
	)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak micro [host] [flags]")
		fmt.Fprintln(os.Stderr, "  fak micro              run ONE microagent end-to-end on the Mock engine")
		fmt.Fprintln(os.Stderr, "  fak micro --selfcheck  prove kernel -> session gateway -> scheduler -> two microagents offline")
		fmt.Fprintln(os.Stderr, "  fak micro host         boot the in-process host (M2) and run a small fleet")
		fmt.Fprintln(os.Stderr, "  fak micro trace <id>   print ONE microagent's structured timeline (legs, tokens, seat, verdicts)")
		fmt.Fprintln(os.Stderr, "  add --dry-run to print the resolved plan (backends, seats, caps) without spending")
		fmt.Fprintln(os.Stderr, "  add --trace-out <file> to persist per-agent traces for a later `fak micro trace <id> --trace-in`")
		fmt.Fprintln(os.Stderr, "\nconfig precedence: flags > env (FAK_MICRO_*) > file (--config) > defaults\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *selfcheck {
		if err := cmdMicroSelfcheck(*jsonOut); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro selfcheck: %v\n", err)
			os.Exit(1)
		}
		return
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })

	// Precedence: defaults < file < env < flags.
	cfg := defaultMicroConfig(hostMode)
	if *cfgFile != "" {
		if err := loadMicroConfigFile(*cfgFile, &cfg); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro: %v\n", err)
			os.Exit(1)
		}
	}
	applyMicroEnv(&cfg)
	if set["isolation"] {
		cfg.Isolation = *isolation
	}
	if set["engine"] {
		cfg.Engine = *engine
	}
	if set["gateway"] {
		cfg.Gateway = *gateway
	}
	if set["model"] {
		cfg.Model = *model
	}
	if set["task"] {
		cfg.Task = *task
	}
	if set["seats"] {
		cfg.Seats = *seats
	}
	if set["workers"] {
		cfg.Workers = *workers
	}
	if set["concurrency"] {
		cfg.Workers = *concurrency
	}
	if set["queue"] {
		cfg.Queue = *queue
	}
	if set["n"] {
		cfg.Agents = *nAgents
	}
	if set["agents"] {
		cfg.Agents = *agentsLong
	}
	if set["turns"] {
		cfg.Turns = *turns
	}
	if set["admission-max-concurrent"] {
		cfg.AdmissionMaxConcurrent = *admMax
	}
	if set["admission-token-budget"] {
		cfg.AdmissionTokenBudget = *admTok
	}

	if len(fs.Args()) > 0 {
		if strings.TrimSpace(cfg.Task) != "" {
			fmt.Fprintln(os.Stderr, "fak micro: task supplied both positionally and with --task")
			os.Exit(2)
		}
		cfg.Task = strings.Join(fs.Args(), " ")
	}
	if err := validateMicroConfig(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "fak micro: %v\n", err)
		os.Exit(2)
	}

	if *dryRun {
		printMicroPlan(cfg, hostMode, *jsonOut, true)
		return
	}
	if err := runMicro(cfg, hostMode, *jsonOut, *traceOut); err != nil {
		fmt.Fprintf(os.Stderr, "fak micro: %v\n", err)
		os.Exit(1)
	}
}

// loadMicroConfigFile overlays a JSON config file onto cfg — only keys present in
// the file override; absent keys keep the default/current value.
func loadMicroConfigFile(path string, cfg *microConfig) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config %s: %w", path, err)
	}
	if err := json.Unmarshal(raw, cfg); err != nil {
		return fmt.Errorf("parse config %s: %w", path, err)
	}
	return nil
}

// applyMicroEnv overlays FAK_MICRO_* env vars onto cfg (above file, below flags).
// An unset or unparseable var leaves the current value untouched.
func applyMicroEnv(cfg *microConfig) {
	if v, ok := os.LookupEnv("FAK_MICRO_ENGINE"); ok && strings.TrimSpace(v) != "" {
		cfg.Engine = strings.TrimSpace(v)
	}
	if v, ok := os.LookupEnv("FAK_MICRO_TASK"); ok && strings.TrimSpace(v) != "" {
		cfg.Task = strings.TrimSpace(v)
	}
	if v, ok := os.LookupEnv("FAK_MICRO_ISOLATION"); ok && strings.TrimSpace(v) != "" {
		cfg.Isolation = strings.TrimSpace(v)
	}
	microEnvInt("FAK_MICRO_SEATS", &cfg.Seats)
	microEnvInt("FAK_MICRO_WORKERS", &cfg.Workers)
	microEnvInt("FAK_MICRO_QUEUE", &cfg.Queue)
	microEnvInt("FAK_MICRO_AGENTS", &cfg.Agents)
	microEnvInt("FAK_MICRO_TURNS", &cfg.Turns)
	microEnvInt("FAK_MICRO_ADMISSION_MAX_CONCURRENT", &cfg.AdmissionMaxConcurrent)
	microEnvInt("FAK_MICRO_ADMISSION_TOKEN_BUDGET", &cfg.AdmissionTokenBudget)
}

func microEnvInt(name string, dst *int) {
	if v, ok := os.LookupEnv(name); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			*dst = n
		}
	}
}

func validateMicroConfig(cfg microConfig) error {
	backends := microagent.RegisteredBackends()
	found := false
	for _, b := range backends {
		if b == cfg.Isolation {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("unknown isolation backend %q (registered: %s)", cfg.Isolation, strings.Join(backends, ", "))
	}
	if cfg.Engine != "mock" && cfg.Engine != "gateway" {
		return fmt.Errorf("engine %q is not supported (want mock or gateway)", cfg.Engine)
	}
	if cfg.Engine == "gateway" && strings.TrimSpace(cfg.Gateway) == "" {
		return fmt.Errorf("--gateway is required with --engine gateway")
	}
	if cfg.Engine == "gateway" && strings.TrimSpace(cfg.Model) == "" {
		return fmt.Errorf("--model is required with --engine gateway")
	}
	if cfg.Workers < 1 {
		return fmt.Errorf("--workers must be >= 1 (got %d)", cfg.Workers)
	}
	if cfg.Queue < 1 {
		return fmt.Errorf("--queue must be >= 1 (got %d)", cfg.Queue)
	}
	if cfg.Agents < 1 {
		return fmt.Errorf("--n/--agents must be >= 1 (got %d)", cfg.Agents)
	}
	if cfg.Turns < 1 {
		return fmt.Errorf("--turns must be >= 1 (got %d)", cfg.Turns)
	}
	if cfg.Seats < 0 {
		return fmt.Errorf("--seats must be >= 0 (got %d)", cfg.Seats)
	}
	if cfg.AdmissionMaxConcurrent < 0 || cfg.AdmissionTokenBudget < 0 {
		return fmt.Errorf("--admission-* caps must be >= 0")
	}
	return nil
}

func unbounded(n int) string {
	if n <= 0 {
		return "unbounded"
	}
	return strconv.Itoa(n)
}

// printMicroPlan renders the resolved plan. dryRun toggles the "no spend" banner.
func printMicroPlan(cfg microConfig, hostMode, jsonOut, dryRun bool) {
	if jsonOut {
		out := struct {
			microConfig
			Mode              string   `json:"mode"`
			DryRun            bool     `json:"dry_run"`
			EffectiveSlots    int      `json:"effective_slots"`
			RegisteredBackend []string `json:"registered_backends"`
		}{
			microConfig:       cfg,
			Mode:              microMode(hostMode),
			DryRun:            dryRun,
			EffectiveSlots:    cfg.slots(),
			RegisteredBackend: microagent.RegisteredBackends(),
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
		return
	}
	banner := "resolved plan"
	if dryRun {
		banner = "resolved plan (dry-run — no spend)"
	}
	fmt.Printf("fak micro %s — %s\n", microMode(hostMode), banner)
	fmt.Printf("  engine:                    %s\n", cfg.Engine)
	if cfg.Engine == "gateway" {
		fmt.Printf("  fak gateway:               %s\n", gatewayBaseURL(cfg.Gateway))
		fmt.Printf("  requested model:           %s\n", cfg.Model)
	}
	fmt.Printf("  isolation backend:         %s   (registered: %s)\n", cfg.Isolation, strings.Join(microagent.RegisteredBackends(), ", "))
	fmt.Printf("  seats (slot pool):         %d   (effective: %d)\n", cfg.Seats, cfg.slots())
	fmt.Printf("  host workers (K):          %d\n", cfg.Workers)
	fmt.Printf("  spawn queue:               %d\n", cfg.Queue)
	fmt.Printf("  agents (N):                %d\n", cfg.Agents)
	fmt.Printf("  turns/agent:               %d\n", cfg.Turns)
	fmt.Printf("  admission max-concurrent:  %s   (M19; enforced on the served gateway)\n", unbounded(cfg.AdmissionMaxConcurrent))
	fmt.Printf("  admission token-budget:    %s   (M19; enforced on the served gateway)\n", unbounded(cfg.AdmissionTokenBudget))
	fmt.Printf("  precedence:                flags > env (FAK_MICRO_*) > file > defaults\n")
}

func microMode(hostMode bool) string {
	if hostMode {
		return "host"
	}
	return "run"
}

// driveMicro drives the real microagent.Host over ONE shared planner (Mock or a
// running fak gateway), wrapped in the cooperative slot scheduler so the seat pool
// bounds concurrent model calls. It spawns N agents, each recording its structured
// span timeline into ONE shared tracer keyed by agent id (#2031), drains them, and
// reaps the results. The returned tracer is the multiplexed per-agent trace store —
// separable by id even though every agent ran interleaved in one process.
func driveMicro(cfg microConfig) (*microSink, *metrics.MicroTracer, []microagent.Result, error) {
	sink, tracer, results, _, err := driveMicroObserved(cfg)
	return sink, tracer, results, err
}

type microObservation struct {
	Answer string
	Usage  agent.Usage
	Model  string
}

func driveMicroObserved(cfg microConfig) (*microSink, *metrics.MicroTracer, []microagent.Result, map[string]microObservation, error) {
	// One scheduler and one session table wrap the ONE shared planner for the
	// entire host. The gateway engine points that planner at a running fak serve;
	// the mock engine keeps the deterministic offline path.
	tbl := session.NewTable()
	base := microagent.Gateway(agent.NewMockPlanner("mock"))
	if cfg.Engine == "gateway" {
		base = agent.NewHTTPPlanner(gatewayBaseURL(cfg.Gateway), cfg.Model, defaultGatewayBearerToken())
	}
	sched := microagent.NewScheduler(cfg.slots())
	defer sched.Close()
	gw := microagent.NewSessionGateway(microagent.NewSchedulingGateway(base, sched), tbl)

	sink := &microSink{counts: map[microagent.EventKind]int{}}
	tracer := metrics.NewMicroTracer()
	seat := fmt.Sprintf("slot-pool/%d", cfg.slots())
	h, err := microagent.NewHost(gw, microagent.Config{
		Workers: cfg.Workers, Queue: cfg.Queue, Sessions: tbl, Audit: sink,
	})
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer h.Close()

	agents := make(map[string]*microTurnAgent, cfg.Agents)
	for i := 0; i < cfg.Agents; i++ {
		id := fmt.Sprintf("micro-%03d", i)
		runner := &microTurnAgent{id: id, turns: cfg.Turns, task: cfg.Task, tracer: tracer, seat: seat}
		agents[id] = runner
		if err := h.Spawn(id, runner); err != nil {
			return nil, nil, nil, nil, fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("drain: %w (live=%d)", err, h.Live())
	}
	results := h.Reap()
	sort.Slice(results, func(i, j int) bool { return results[i].ID < results[j].ID })
	observed := make(map[string]microObservation, len(agents))
	for id, runner := range agents {
		answer, usage, model := runner.snapshot()
		observed[id] = microObservation{Answer: answer, Usage: usage, Model: model}
	}
	return sink, tracer, results, observed, nil
}

// runMicro drives the fleet and reports. It optionally persists the per-agent
// traces to a JSONL file (--trace-out) for a later `fak micro trace <id> --trace-in`
// readout. This is the live end-to-end witness the acceptance names.
func runMicro(cfg microConfig, hostMode, jsonOut bool, traceOut string) error {
	sink, tracer, results, observed, err := driveMicroObserved(cfg)
	if err != nil {
		return err
	}
	if traceOut != "" {
		if err := writeTraceFile(traceOut, tracer); err != nil {
			return fmt.Errorf("write --trace-out %s: %w", traceOut, err)
		}
	}

	done, failed := 0, 0
	for _, r := range results {
		if r.Done && r.Err == nil {
			done++
		} else {
			failed++
		}
	}

	if jsonOut {
		type resultOut struct {
			ID               string `json:"id"`
			Steps            int    `json:"steps"`
			Done             bool   `json:"done"`
			Err              string `json:"err,omitempty"`
			Answer           string `json:"answer,omitempty"`
			Model            string `json:"model,omitempty"`
			PromptTokens     int    `json:"prompt_tokens,omitempty"`
			CompletionTokens int    `json:"completion_tokens,omitempty"`
			TotalTokens      int    `json:"total_tokens,omitempty"`
		}
		out := struct {
			Mode    string      `json:"mode"`
			Engine  string      `json:"engine"`
			Slots   int         `json:"slots"`
			Agents  int         `json:"agents"`
			Done    int         `json:"done"`
			Failed  int         `json:"failed"`
			Results []resultOut `json:"results"`
		}{Mode: microMode(hostMode), Engine: cfg.Engine, Slots: cfg.slots(), Agents: cfg.Agents, Done: done, Failed: failed}
		for _, r := range results {
			obs := observed[r.ID]
			ro := resultOut{ID: r.ID, Steps: r.Steps, Done: r.Done, Answer: obs.Answer, Model: obs.Model, PromptTokens: obs.Usage.PromptTokens, CompletionTokens: obs.Usage.CompletionTokens, TotalTokens: obs.Usage.TotalTokens}
			if r.Err != nil {
				ro.Err = r.Err.Error()
			}
			out.Results = append(out.Results, ro)
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			return err
		}
	} else {
		fmt.Printf("fak micro %s — %d agent(s) on %s, slots=%d\n", microMode(hostMode), cfg.Agents, cfg.Engine, cfg.slots())
		for _, r := range results {
			status := "done"
			if !r.Done || r.Err != nil {
				status = "FAILED"
			}
			line := fmt.Sprintf("  %-12s %s  (%d step(s))", r.ID, status, r.Steps)
			if r.Err != nil {
				line += "  err=" + r.Err.Error()
			}
			fmt.Println(line)
		}
		fmt.Printf("  spawn=%d done=%d error=%d cancel=%d  |  %d done / %d failed\n",
			sink.count(microagent.EventSpawn), sink.count(microagent.EventDone),
			sink.count(microagent.EventError), sink.count(microagent.EventCancel), done, failed)
		if traceOut != "" {
			fmt.Printf("  traces:                    %d agent(s) written to %s  (read one: fak micro trace <id> --trace-in %s)\n",
				len(tracer.IDs()), traceOut, traceOut)
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d of %d agent(s) did not complete", failed, cfg.Agents)
	}
	return nil
}

// microTurnAgent is one hosted agent that takes `turns` model turns through the
// host-shared gateway and then retires — the minimal in-process agent loop (the
// resumable internal/agent RunArm stepping is the still-open #2001 extraction).
// Each turn it records its structured trace legs (seat → step → verdict) into the
// shared tracer under its own id, so the interleaved fleet stays separable (#2031).
type microTurnAgent struct {
	id     string
	turns  int
	task   string
	took   int
	mu     sync.Mutex
	answer string
	usage  agent.Usage
	model  string
	tracer *metrics.MicroTracer // nil ⇒ no tracing
	seat   string               // the slot-pool seat this agent's calls draw from
}

func (a *microTurnAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.took++
	// Seat leg: every model call draws one seat from the shared slot pool (M6/M20).
	a.trace(metrics.MicroSpan{Kind: metrics.SpanSeat, Label: "acquire", Seat: a.seat})
	prompt := strings.TrimSpace(a.task)
	if prompt == "" {
		prompt = fmt.Sprintf("micro %s turn %d", a.id, a.took)
	}
	msg := []agent.Message{{Role: agent.RoleUser, Content: prompt}}
	comp, err := gw.Complete(microagent.WithTrace(ctx, a.id), msg, nil)
	if err != nil {
		a.trace(metrics.MicroSpan{Kind: metrics.SpanStep, Label: fmt.Sprintf("turn %d", a.took), Verdict: "ERROR"})
		return false, err
	}
	// Step leg: the model turn, with the tokens the completion reports. The Mock
	// planner reports no usage, so fall back to a byte-count estimate of the prompt
	// (a SIMULATED count — the real gateway usage flows in once #2030 wires it).
	tokens := 0
	if comp != nil {
		tokens = comp.Usage.TotalTokens
		a.mu.Lock()
		a.answer = comp.Message.Content
		a.usage = comp.Usage
		a.model = comp.Model
		a.mu.Unlock()
	}
	if tokens == 0 {
		tokens = len(msg[0].Content)
	}
	a.trace(metrics.MicroSpan{Kind: metrics.SpanStep, Label: fmt.Sprintf("turn %d", a.took), Tokens: tokens})
	// Verdict leg: the Mock planner admits every call; the real adjudication verdict
	// flows in on the served-gateway path (#2030).
	a.trace(metrics.MicroSpan{Kind: metrics.SpanVerdict, Label: "mock-planner", Verdict: "ALLOW"})
	return a.took >= a.turns, nil
}

func (a *microTurnAgent) snapshot() (string, agent.Usage, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.answer, a.usage, a.model
}

// trace records one span for this agent, if tracing is on.
func (a *microTurnAgent) trace(s metrics.MicroSpan) {
	if a.tracer != nil {
		a.tracer.Record(a.id, s)
	}
}

// microSink is the host's ONE audit sink for the CLI run — it just tallies event
// kinds so the run can report the lifecycle at the end.
type microSink struct {
	mu     sync.Mutex
	counts map[microagent.EventKind]int
}

func (s *microSink) Record(ev microagent.Event) {
	s.mu.Lock()
	s.counts[ev.Kind]++
	s.mu.Unlock()
}

func (s *microSink) count(k microagent.EventKind) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.counts[k]
}

// writeTraceFile persists every per-agent trace as JSONL (one trace per line).
func writeTraceFile(path string, tracer *metrics.MicroTracer) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := tracer.WriteJSONL(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// parseMicroTraceArgs binds fs against args and peels out the single <id>
// positional, accepting it before OR after the flags. Go's flag package stops at
// the first non-flag token, so a lone Parse would reject `trace <id> --trace-in x`
// — the very form the run summary prints. Loop-parse instead, consuming one
// positional per pass, so both orderings resolve to the same id and flag values.
func parseMicroTraceArgs(fs *flag.FlagSet, args []string) (string, error) {
	var positionals []string
	rest := args
	for {
		if err := fs.Parse(rest); err != nil {
			return "", err
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		positionals = append(positionals, rest[0])
		rest = rest[1:]
	}
	if len(positionals) != 1 {
		return "", fmt.Errorf("want exactly one <id>, got %d", len(positionals))
	}
	return positionals[0], nil
}

// cmdMicroTrace implements `fak micro trace <id>` (#2031): the per-agent trace
// readout. Without --trace-in it runs a fresh deterministic Mock fleet with tracing
// on and renders agent <id>'s timeline; with --trace-in it reads a trace JSONL a
// prior run persisted (--trace-out) — the cross-process separability path. Either
// way the output is ONE agent's timeline pulled out of the interleaved fleet.
func cmdMicroTrace(args []string) {
	fs := flag.NewFlagSet("micro trace", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var (
		traceIn = fs.String("trace-in", "", "read a persisted trace JSONL (from `fak micro --trace-out`) instead of running a fresh fleet")
		nAgents = fs.Int("n", microagent.DefaultWorkers, "fleet size to run when no --trace-in is given")
		turns   = fs.Int("turns", 3, "turns per agent when no --trace-in is given")
		jsonOut = fs.Bool("json", false, "emit the trace as JSON instead of a rendered timeline")
	)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak micro trace <id> [flags]")
		fmt.Fprintln(os.Stderr, "  print ONE microagent's structured timeline (legs, tokens, seat, verdicts)")
		fmt.Fprintln(os.Stderr, "  with no --trace-in, runs a fresh deterministic Mock fleet (ids micro-000, micro-001, …)")
		fs.PrintDefaults()
	}
	id, err := parseMicroTraceArgs(fs, args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak micro trace: %v\n", err)
		fs.Usage()
		os.Exit(2)
	}

	var tracer *metrics.MicroTracer
	if *traceIn != "" {
		f, err := os.Open(*traceIn)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak micro trace: open --trace-in: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		tracer, err = metrics.ReadTracesJSONL(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak micro trace: %v\n", err)
			os.Exit(1)
		}
	} else {
		cfg := defaultMicroConfig(true)
		cfg.Agents = *nAgents
		cfg.Turns = *turns
		if err := validateMicroConfig(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro trace: %v\n", err)
			os.Exit(2)
		}
		if _, tracer, _, err = driveMicro(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro trace: %v\n", err)
			os.Exit(1)
		}
	}

	tr, ok := tracer.Trace(id)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak micro trace: no trace for %q (known ids: %s)\n", id, strings.Join(tracer.IDs(), ", "))
		os.Exit(1)
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(tr); err != nil {
			fmt.Fprintf(os.Stderr, "fak micro trace: %v\n", err)
			os.Exit(1)
		}
		return
	}
	fmt.Print(tr.Render())
}
