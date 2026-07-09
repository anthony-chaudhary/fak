package main

// micro.go — the `fak micro` front door for the native in-process Go microagent
// runtime (#2029, part of epic #2000 M30). It gives the host built in
// internal/microagent (the worker pool + slot scheduler + audit sink, #2002/#2006)
// a CLI verb consistent with `fak serve`/`guard`/`dispatch`, instead of only a
// package a test constructs:
//
//	fak micro            run ONE microagent end-to-end on the Mock engine.
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
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// microConfig is the resolved plan for a `fak micro` run: the backends, seats,
// admission caps, and fleet shape after applying flags > env > file > defaults.
type microConfig struct {
	Engine                 string `json:"engine"`                   // model engine seam; only "mock" is supported in-process today.
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
		engine      = fs.String("engine", "", "model engine seam (only \"mock\" is supported in-process today)")
		admMax      = fs.Int("admission-max-concurrent", 0, "M19 provider concurrency cap; 0 ⇒ unbounded (resolved here, enforced on the served gateway)")
		admTok      = fs.Int("admission-token-budget", 0, "M19 provider token budget; 0 ⇒ unbounded (resolved here, enforced on the served gateway)")
		cfgFile     = fs.String("config", "", "load config from a JSON file (lowest non-default precedence)")
		dryRun      = fs.Bool("dry-run", false, "resolve and print the plan (backends, seats, caps) without spending")
		jsonOut     = fs.Bool("json", false, "emit JSON instead of a human-readable report")
	)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak micro [host] [flags]")
		fmt.Fprintln(os.Stderr, "  fak micro              run ONE microagent end-to-end on the Mock engine")
		fmt.Fprintln(os.Stderr, "  fak micro host         boot the in-process host (M2) and run a small fleet")
		fmt.Fprintln(os.Stderr, "  add --dry-run to print the resolved plan (backends, seats, caps) without spending")
		fmt.Fprintln(os.Stderr, "\nconfig precedence: flags > env (FAK_MICRO_*) > file (--config) > defaults\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
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

	if err := validateMicroConfig(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "fak micro: %v\n", err)
		os.Exit(2)
	}

	if *dryRun {
		printMicroPlan(cfg, hostMode, *jsonOut, true)
		return
	}
	if cfg.Engine != "mock" {
		fmt.Fprintf(os.Stderr, "fak micro: --engine %q not supported in-process yet (only \"mock\"); use --dry-run to inspect the plan\n", cfg.Engine)
		os.Exit(2)
	}
	if err := runMicro(cfg, hostMode, *jsonOut); err != nil {
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

// runMicro drives the real microagent.Host over the Mock planner as the ONE shared
// gateway seam, wrapped in the cooperative slot scheduler so the seat pool bounds
// concurrent model calls. It spawns N agents, drains them, reaps the results, and
// reports. This is the live end-to-end witness the acceptance names.
func runMicro(cfg microConfig, hostMode, jsonOut bool) error {
	// The ONE shared gateway: the deterministic offline Mock planner (the same Mock
	// the offline demo path uses), wrapped in the slot scheduler so no more than
	// `slots` model calls are ever in flight across the whole fleet.
	base := microagent.Gateway(agent.NewMockPlanner("mock"))
	sched := microagent.NewScheduler(cfg.slots())
	defer sched.Close()
	gw := microagent.NewSchedulingGateway(base, sched)

	sink := &microSink{counts: map[microagent.EventKind]int{}}
	tbl := session.NewTable()
	h, err := microagent.NewHost(gw, microagent.Config{
		Workers:  cfg.Workers,
		Queue:    cfg.Queue,
		Sessions: tbl,
		Audit:    sink,
	})
	if err != nil {
		return err
	}
	defer h.Close()

	for i := 0; i < cfg.Agents; i++ {
		id := fmt.Sprintf("micro-%03d", i)
		if err := h.Spawn(id, &microTurnAgent{id: id, turns: cfg.Turns}); err != nil {
			return fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		return fmt.Errorf("drain: %w (live=%d)", err, h.Live())
	}
	results := h.Reap()
	sort.Slice(results, func(i, j int) bool { return results[i].ID < results[j].ID })

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
			ID    string `json:"id"`
			Steps int    `json:"steps"`
			Done  bool   `json:"done"`
			Err   string `json:"err,omitempty"`
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
			ro := resultOut{ID: r.ID, Steps: r.Steps, Done: r.Done}
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
	}
	if failed > 0 {
		return fmt.Errorf("%d of %d agent(s) did not complete", failed, cfg.Agents)
	}
	return nil
}

// microTurnAgent is one hosted agent that takes `turns` model turns through the
// host-shared gateway and then retires — the minimal in-process agent loop (the
// resumable internal/agent RunArm stepping is the still-open #2001 extraction).
type microTurnAgent struct {
	id    string
	turns int
	took  int
}

func (a *microTurnAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.took++
	msg := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("micro %s turn %d", a.id, a.took)}}
	if _, err := gw.Complete(ctx, msg, nil); err != nil {
		return false, err
	}
	return a.took >= a.turns, nil
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
