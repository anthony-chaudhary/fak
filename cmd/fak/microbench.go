package main

// microbench.go — `fak microbench` (#2008, part of epic #2000 M8): the per-agent
// RSS + CPU density witness for the native in-process microagent runtime.
//
// The project claims "ultra-light memory/CPU, 1000s of agents in one process";
// this verb turns that claim into a measured number, with ZERO provider spend:
//
//  1. in-process cells: for each N (default 100 and 1000) it boots the real
//     internal/microagent Host on the deterministic Mock planner (the exact host
//     `fak micro` drives), parks all N agents at steady state simultaneously,
//     and reports RSS/agent, host steady + peak RSS, goroutines, CPU, and wall.
//  2. guarded-CLI baseline: today's production shape is ~2 OS processes per
//     agent (guard + external CLI). At small N it spawns idle copies of THIS
//     binary as a stand-in pair and measures their resident cost per agent.
//     The real external CLI is heavier than an idle fak process, so the stated
//     delta is a floor, not a flattering estimate.
//  3. delta: baseline RSS/agent ÷ in-process RSS/agent at the largest N — a
//     number, not a claim.
//
// Rows append as JSONL (schema fak-microbench/1) to the gitignored live ledger
// .fak/nightrun/microbench.jsonl by default; a reviewed snapshot is published to
// docs/nightrun/microbench.jsonl by an explicit by-path commit, per the
// docs/nightrun README honesty boundary.
//
// Measurement notes:
//   - every probe runs debug.FreeOSMemory() first so RSS deltas reflect live
//     residency, not un-scavenged heap;
//   - host peak RSS is the OS process-lifetime peak (Windows peakWorkingSetSize /
//     unix VmHWM-style), so cells run in ascending N order to keep each cell's
//     peak attributable;
//   - an axis the platform cannot read stays 0 in the row — never fabricated.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// microbenchSchema tags each JSONL row.
const microbenchSchema = "fak-microbench/1"

// microbenchDefaultOut is the gitignored live ledger (docs/nightrun/README:
// live writers target .fak/nightrun/; tracked docs snapshots are explicit
// by-path publications, never a background write target).
const microbenchDefaultOut = ".fak/nightrun/microbench.jsonl"

// microbenchBaselineNote states what the baseline cell actually measured, so a
// reader of the ledger cannot mistake the proxy for the real external CLI.
const microbenchBaselineNote = "proxy: idle fak processes stand in for the guard + external-CLI pair (2 OS processes/agent); a real agent CLI is heavier, so the delta is a floor"

// microbenchRow is one durable JSONL row: an in-process density cell
// (kind=inprocess), a guarded-CLI baseline cell (kind=baseline), or the folded
// comparison (kind=delta). Axes a platform cannot read stay 0.
type microbenchRow struct {
	Schema string `json:"schema"`
	TS     string `json:"ts"`
	Kind   string `json:"kind"` // inprocess | baseline | delta
	GOOS   string `json:"goos"`
	NumCPU int    `json:"num_cpu"`

	// inprocess + baseline
	N                int     `json:"n,omitempty"`
	Turns            int     `json:"turns,omitempty"`
	Workers          int     `json:"workers,omitempty"`
	Done             int     `json:"done,omitempty"`
	Failed           int     `json:"failed,omitempty"`
	Engine           string  `json:"engine,omitempty"`
	RSSBeforeBytes   uint64  `json:"rss_before_bytes,omitempty"`
	RSSSteadyBytes   uint64  `json:"rss_steady_bytes,omitempty"`
	RSSAfterBytes    uint64  `json:"rss_after_bytes,omitempty"`
	RSSPerAgentBytes uint64  `json:"rss_per_agent_bytes,omitempty"`
	GoroutinesBefore int     `json:"goroutines_before,omitempty"`
	GoroutinesSteady int     `json:"goroutines_steady,omitempty"`
	HostPeakRSSBytes uint64  `json:"host_peak_rss_bytes,omitempty"` // process-lifetime OS peak at cell end
	CPUSeconds       float64 `json:"cpu_seconds,omitempty"`
	WallSeconds      float64 `json:"wall_seconds,omitempty"`

	// baseline only
	ProcsPerAgent     int    `json:"procs_per_agent,omitempty"`
	ChildMeanRSSBytes uint64 `json:"child_mean_rss_bytes,omitempty"`
	Note              string `json:"note,omitempty"`

	// delta only
	DeltaAtN                  int     `json:"delta_at_n,omitempty"`
	BaselineRSSPerAgentBytes  uint64  `json:"baseline_rss_per_agent_bytes,omitempty"`
	InprocessRSSPerAgentBytes uint64  `json:"inprocess_rss_per_agent_bytes,omitempty"`
	BaselineOverInprocessRSS  float64 `json:"baseline_over_inprocess_rss_factor,omitempty"`
}

func cmdMicroBench(args []string) {
	fs := flag.NewFlagSet("microbench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var (
		nList     = fs.String("n", "100,1000", "comma-separated in-process fleet sizes to measure")
		turns     = fs.Int("turns", 1, "mock model turns each agent takes before parking at steady state")
		baselineN = fs.Int("baseline-n", 8, "guarded-CLI baseline fleet size (0 disables the baseline + delta rows)")
		jsonOut   = fs.Bool("json", false, "emit the rows as JSON to stdout instead of the human report")
		outPath   = fs.String("out", microbenchDefaultOut, "append JSONL rows to this ledger (\"\" disables the artifact)")
		asChild   = fs.Bool("as-child", false, "internal: run as ONE idle baseline child (report own RSS, park until stdin closes)")
	)
	fs.Usage = func() {
		writeUsageLines(os.Stderr,
			"usage: fak microbench [flags]",
			"  measure per-agent RSS + CPU density of the in-process microagent host (#2008)",
			"  vs the guarded-CLI ~2-OS-processes-per-agent baseline — mock engine, no spend",
		)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *asChild {
		runMicrobenchChild()
		return
	}
	sizes, err := parseIntListCSV(*nList)
	if err != nil || len(sizes) == 0 {
		fmt.Fprintf(os.Stderr, "fak microbench: -n wants a non-empty int list like \"100,1000\" (got %q)\n", *nList)
		os.Exit(2)
	}
	for _, n := range sizes {
		if n < 1 {
			fmt.Fprintf(os.Stderr, "fak microbench: -n sizes must be >= 1 (got %d)\n", n)
			os.Exit(2)
		}
	}
	if *turns < 1 || *baselineN < 0 {
		fmt.Fprintln(os.Stderr, "fak microbench: --turns must be >= 1 and --baseline-n >= 0")
		os.Exit(2)
	}
	// Ascending N keeps each cell's process-lifetime peak-RSS reading attributable.
	sort.Ints(sizes)

	var rows []microbenchRow
	for _, n := range sizes {
		row, err := runMicrobenchCell(n, *turns)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak microbench: in-process N=%d: %v\n", n, err)
			os.Exit(1)
		}
		rows = append(rows, row)
	}
	var baselineErr error
	if *baselineN > 0 {
		brow, err := runMicrobenchBaseline(*baselineN)
		if err != nil {
			baselineErr = err
		} else {
			rows = append(rows, brow, microbenchDeltaRow(brow, rows[len(rows)-1]))
		}
	}

	if *outPath != "" {
		if err := microbenchAppendJSONL(*outPath, rows); err != nil {
			exitf(1, "fak microbench: write %s: %v\n", *outPath, err)
		}
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rows)
	} else {
		printMicrobenchReport(rows, *outPath)
	}
	if baselineErr != nil {
		fmt.Fprintf(os.Stderr, "fak microbench: guarded-CLI baseline failed (in-process rows above are still valid): %v\n", baselineErr)
		os.Exit(1)
	}
}

// microbenchProbe takes one instant resource reading of THIS process, after
// forcing a GC + scavenge so RSS reflects live residency. It rides the exported
// harnessres Sampler seam: Start takes an immediate sample and Stop takes a
// final one, so a Start/Stop pair with a never-firing ticker is a point probe.
func microbenchProbe() harnessres.Snapshot {
	debug.FreeOSMemory()
	s := harnessres.New()
	s.Start(time.Hour)
	return s.Stop()
}

// microbenchAgent is one hosted density agent: it takes `turns` mock model
// turns through the shared gateway, signals arrival, then PARKS (holding its
// worker goroutine + session entry, but no seat) until the bench has sampled
// steady state, and retires when hold closes.
type microbenchAgent struct {
	id      string
	turns   int
	took    int
	arrived func()          // called exactly once, at first arrival at steady state
	hold    <-chan struct{} // closed by the bench after sampling
}

func (a *microbenchAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.took++
	msg := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("microbench %s turn %d", a.id, a.took)}}
	if _, err := gw.Complete(ctx, msg, nil); err != nil {
		return false, err
	}
	if a.took < a.turns {
		return false, nil
	}
	a.arrived()
	select {
	case <-a.hold:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// runMicrobenchCell measures ONE in-process density cell: N agents on the real
// microagent Host (Workers=N — one Step-driver goroutine per agent, the shape
// the density claim is about), all parked live simultaneously at steady state.
func runMicrobenchCell(n, turns int) (microbenchRow, error) {
	row := microbenchRow{
		Schema:  microbenchSchema,
		TS:      time.Now().UTC().Format(time.RFC3339),
		Kind:    "inprocess",
		GOOS:    runtime.GOOS,
		NumCPU:  runtime.NumCPU(),
		N:       n,
		Turns:   turns,
		Workers: n,
		Engine:  "mock",
	}
	// The same seams `fak micro` drives: Mock planner as the ONE shared gateway,
	// wrapped in the slot scheduler (slots=N so seating never throttles the fleet
	// we are trying to hold live).
	base := microagent.Gateway(agent.NewMockPlanner("mock"))
	sched := microagent.NewScheduler(n)
	defer sched.Close()
	gw := microagent.NewSchedulingGateway(base, sched)

	before := microbenchProbe()
	start := time.Now()

	h, err := microagent.NewHost(gw, microagent.Config{
		Workers:  n,
		Queue:    n,
		Sessions: session.NewTable(),
	})
	if err != nil {
		return row, err
	}
	defer h.Close()

	hold := make(chan struct{})
	var arrived sync.WaitGroup
	arrived.Add(n)
	arrivedOnce := func() { arrived.Done() }
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("bench-%04d", i)
		if err := h.Spawn(id, &microbenchAgent{id: id, turns: turns, arrived: arrivedOnce, hold: hold}); err != nil {
			close(hold)
			return row, fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	steadyCh := make(chan struct{})
	go func() { arrived.Wait(); close(steadyCh) }()
	select {
	case <-steadyCh:
	case <-time.After(60 * time.Second):
		close(hold)
		return row, fmt.Errorf("fleet did not reach steady state within 60s (live=%d)", h.Live())
	}

	steady := microbenchProbe()
	close(hold)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		return row, fmt.Errorf("drain: %w (live=%d)", err, h.Live())
	}
	for _, r := range h.Reap() {
		if r.Done && r.Err == nil {
			row.Done++
		} else {
			row.Failed++
		}
	}
	wall := time.Since(start)
	h.Close() // retire the N worker goroutines before the recovery probe
	after := microbenchProbe()

	row.RSSBeforeBytes = before.Kernel.RSSBytes
	row.RSSSteadyBytes = steady.Kernel.RSSBytes
	row.RSSAfterBytes = after.Kernel.RSSBytes
	if steady.Kernel.RSSBytes > before.Kernel.RSSBytes {
		row.RSSPerAgentBytes = (steady.Kernel.RSSBytes - before.Kernel.RSSBytes) / uint64(n)
	}
	row.GoroutinesBefore = before.GoroutinesPeak
	row.GoroutinesSteady = steady.GoroutinesPeak
	row.HostPeakRSSBytes = after.Kernel.PeakRSSBytes
	if before.Kernel.HaveCPU && after.Kernel.HaveCPU {
		row.CPUSeconds = after.Kernel.CPUSeconds() - before.Kernel.CPUSeconds()
	}
	row.WallSeconds = wall.Seconds()
	return row, nil
}

// microbenchChildReport is the one JSON line a baseline child prints.
type microbenchChildReport struct {
	RSSBytes     uint64 `json:"rss_bytes"`
	PeakRSSBytes uint64 `json:"peak_rss_bytes"`
}

// runMicrobenchChild is the hidden `--as-child` mode: settle, self-report RSS,
// then park resident until the parent closes stdin — so the parent samples with
// every child alive simultaneously.
func runMicrobenchChild() {
	snap := microbenchProbe()
	_ = json.NewEncoder(os.Stdout).Encode(microbenchChildReport{
		RSSBytes:     snap.Kernel.RSSBytes,
		PeakRSSBytes: snap.Kernel.PeakRSSBytes,
	})
	_, _ = io.Copy(io.Discard, os.Stdin)
}

// runMicrobenchBaseline measures the guarded-CLI contrast at small N: today's
// production shape is ~2 OS processes per agent (guard + external CLI). It
// spawns N idle copies of THIS binary, holds them all resident at once, folds
// their self-reported RSS, and prices one agent at 2 processes.
func runMicrobenchBaseline(n int) (microbenchRow, error) {
	row := microbenchRow{
		Schema: microbenchSchema,
		TS:     time.Now().UTC().Format(time.RFC3339),
		Kind:   "baseline",
		GOOS:   runtime.GOOS,
		NumCPU: runtime.NumCPU(),
		N:      n,
	}
	exe, err := os.Executable()
	if err != nil {
		return row, fmt.Errorf("resolve own executable: %w", err)
	}
	type child struct {
		cmd   *exec.Cmd
		stdin io.WriteCloser
	}
	var children []child
	killAll := func() {
		for _, c := range children {
			if c.cmd.Process != nil {
				_ = c.cmd.Process.Kill()
			}
		}
		for _, c := range children {
			_ = c.cmd.Wait()
		}
	}
	reports := make(chan microbenchChildReport, n)
	errs := make(chan error, n)
	start := time.Now()
	for i := 0; i < n; i++ {
		cmd := exec.Command(exe, "microbench", "--as-child")
		windowgate.ConfigureBackgroundCommand(cmd) // windowless on Windows: no console flash from a background parent
		stdin, err := cmd.StdinPipe()
		if err != nil {
			killAll()
			return row, err
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			killAll()
			return row, err
		}
		cmd.Stderr = io.Discard
		if err := cmd.Start(); err != nil {
			killAll()
			return row, fmt.Errorf("start baseline child: %w", err)
		}
		children = append(children, child{cmd: cmd, stdin: stdin})
		go func(r io.Reader) {
			var rep microbenchChildReport
			if err := json.NewDecoder(r).Decode(&rep); err != nil {
				errs <- err
				return
			}
			reports <- rep
		}(stdout)
	}
	var totalRSS uint64
	deadline := time.After(60 * time.Second)
	for got := 0; got < n; got++ {
		select {
		case rep := <-reports:
			totalRSS += rep.RSSBytes
		case err := <-errs:
			killAll()
			return row, fmt.Errorf("baseline child report: %w", err)
		case <-deadline:
			killAll()
			return row, fmt.Errorf("baseline children did not all report within 60s")
		}
	}
	// Every child resident simultaneously: release them and fold their CPU.
	for _, c := range children {
		_ = c.stdin.Close()
	}
	var cpu float64
	for _, c := range children {
		_ = c.cmd.Wait()
		if st := c.cmd.ProcessState; st != nil {
			cpu += st.UserTime().Seconds() + st.SystemTime().Seconds()
		}
	}
	row.ChildMeanRSSBytes = totalRSS / uint64(n)
	row.ProcsPerAgent = 2
	row.RSSPerAgentBytes = 2 * row.ChildMeanRSSBytes
	row.CPUSeconds = cpu
	row.WallSeconds = time.Since(start).Seconds()
	row.Note = microbenchBaselineNote
	return row, nil
}

// microbenchDeltaRow folds the baseline-vs-in-process comparison into ONE row:
// the acceptance's "delta stated as a number, not a claim".
func microbenchDeltaRow(baseline, inproc microbenchRow) microbenchRow {
	factor, note := microbenchDeltaFactor(baseline.RSSPerAgentBytes, inproc.RSSPerAgentBytes)
	return microbenchRow{
		Schema:                    microbenchSchema,
		TS:                        time.Now().UTC().Format(time.RFC3339),
		Kind:                      "delta",
		GOOS:                      runtime.GOOS,
		NumCPU:                    runtime.NumCPU(),
		DeltaAtN:                  inproc.N,
		BaselineRSSPerAgentBytes:  baseline.RSSPerAgentBytes,
		InprocessRSSPerAgentBytes: inproc.RSSPerAgentBytes,
		BaselineOverInprocessRSS:  factor,
		Note:                      note,
	}
}

// microbenchDeltaFactor is baseline ÷ in-process RSS per agent. A sub-1-byte
// in-process reading is floor-clamped to 1 B and disclosed, never inflated.
func microbenchDeltaFactor(baselinePerAgent, inprocPerAgent uint64) (float64, string) {
	if baselinePerAgent == 0 {
		return 0, "baseline per-agent RSS unreadable on this platform; no factor"
	}
	if inprocPerAgent == 0 {
		return float64(baselinePerAgent), "in-process per-agent RSS measured below the 1-byte floor; factor computed against a 1 B floor"
	}
	return float64(baselinePerAgent) / float64(inprocPerAgent), ""
}

// microbenchAppendJSONL appends one row per line to the ledger, creating the
// directory on first write.
func microbenchAppendJSONL(path string, rows []microbenchRow) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			f.Close()
			return err
		}
	}
	return f.Close()
}

// printMicrobenchReport renders the human summary: one line per cell, then the
// baseline and the delta.
func printMicrobenchReport(rows []microbenchRow, outPath string) {
	fmt.Println("fak microbench — in-process microagent density vs guarded-CLI baseline (mock engine, no spend)")
	for _, r := range rows {
		switch r.Kind {
		case "inprocess":
			fmt.Printf("  in-process N=%d (turns=%d): rss/agent %s  steady %s (host peak %s)  goroutines %d  cpu %.2fs  wall %.2fs  done %d/%d\n",
				r.N, r.Turns, humanBytes(int64(r.RSSPerAgentBytes)), humanBytes(int64(r.RSSSteadyBytes)),
				humanBytes(int64(r.HostPeakRSSBytes)), r.GoroutinesSteady, r.CPUSeconds, r.WallSeconds, r.Done, r.N)
		case "baseline":
			fmt.Printf("  baseline N=%d: %d OS procs/agent x %s mean idle-process rss = %s/agent  cpu %.2fs  wall %.2fs\n",
				r.N, r.ProcsPerAgent, humanBytes(int64(r.ChildMeanRSSBytes)), humanBytes(int64(r.RSSPerAgentBytes)),
				r.CPUSeconds, r.WallSeconds)
			fmt.Printf("    (%s)\n", r.Note)
		case "delta":
			fmt.Printf("  delta @N=%d: baseline %s/agent / in-process %s/agent = %.1fx lighter\n",
				r.DeltaAtN, humanBytes(int64(r.BaselineRSSPerAgentBytes)),
				humanBytes(int64(r.InprocessRSSPerAgentBytes)), r.BaselineOverInprocessRSS)
			if r.Note != "" {
				fmt.Printf("    (%s)\n", r.Note)
			}
		}
	}
	if outPath != "" {
		fmt.Printf("  jsonl: %s (+%d row(s))\n", outPath, len(rows))
	}
}
