package main

// stallscan.go — `fak stallscan`, the durable answer to "the machine locked up
// again but usage looks fine." It reads the CHURN signals that reveal a
// low-usage stall (soft-fault storm, scheduler/syscall thrash, process-spawn
// bursts) — the ones no CPU%/RAM/disk meter shows — classifies them via the
// pure internal/stallscan.Classify, and either prints one fingerprint
// (snapshot) or runs a cheap self-monitor loop that appends a rolling JSONL and
// flags each stall as it happens.
//
//	fak stallscan                  # one fingerprint of right now (human)
//	fak stallscan --json           # that fingerprint as JSON
//	fak stallscan --watch          # background self-monitor: sample, classify, append JSONL, alert on stall
//	fak stallscan --watch --interval 10s --log <path>
//
// WHY IT EXISTS. On this box the stalls are kernel-path contention (page-fault
// + scheduler locks) driven by process/thread churn across many concurrent
// sessions. Point-in-time CPU/RAM/disk all read low, so the ordinary reapers
// never fire. This verb samples the axes that DO move, so a recurrence is one
// command to confirm — and the --watch loop leaves a durable trail so we start
// the next investigation from evidence, not from scratch. See
// internal/stallscan for the classification and its live-calibrated thresholds.
//
// COST DISCIPLINE. Windows process enumeration is itself expensive (a naive
// Get-CimInstance poll can hit >100k IO-ops/sec — it becomes part of the
// problem). So each tick does exactly ONE Get-Counter batch plus ONE
// Get-CimInstance process snapshot, at a default 15s interval. Do not lower the
// interval below a few seconds or add per-tick enumeration.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

func cmdStallscan(argv []string) { os.Exit(runStallscan(os.Stdout, os.Stderr, argv)) }

// runStallscan is the testable core. Exit 0 ok / calm, 1 runtime error, 2 usage,
// 3 a stall was observed (snapshot mode only — so a script can gate on it).
func runStallscan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("stallscan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the fingerprint as JSON")
	watch := fs.Bool("watch", false, "run a self-monitor loop: sample, classify, append JSONL, alert on each stall")
	interval := fs.Duration("interval", 15*time.Second, "sample interval in --watch mode (keep >= a few seconds; enumeration is expensive)")
	logPath := fs.String("log", "", "rolling JSONL path for --watch (default: <host stall dir>/stallscan.jsonl)")
	topN := fs.Int("top", 6, "how many top-IO processes to show")
	once := fs.Bool("once", false, "in --watch mode, take exactly one sample then exit (for tests/cron)")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *watch {
		return runStallscanWatch(stdout, stderr, *interval, *logPath, *topN, *once)
	}

	// Snapshot mode: one sample, classify, render.
	sample, err := gatherStallSample(*topN)
	if err != "" {
		fmt.Fprintf(stderr, "fak stallscan: %s\n", err)
		return 1
	}
	v := stallscan.Classify(sample, stallscan.DefaultThresholds())
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, stallFingerprint(sample, v), "fak stallscan")
	}
	renderStallFingerprint(stdout, sample, v, *topN)
	if v.Level == stallscan.LevelStall {
		return 3
	}
	return 0
}

// stallFingerprint is the JSON/JSONL record: the sample plus its verdict.
func stallFingerprint(s stallscan.Sample, v stallscan.Verdict) map[string]any {
	return map[string]any{
		"schema":  "fak.stallscan.v1",
		"ts":      time.Now().UTC().Format(time.RFC3339Nano),
		"sample":  s,
		"verdict": v,
	}
}

// runStallscanWatch is the self-monitor loop. Cheap by construction: one sample
// per interval, append one JSONL line, print a one-liner only when the verdict
// is elevated/stall (calm ticks are logged but not printed, to stay quiet).
func runStallscanWatch(stdout, stderr io.Writer, interval time.Duration, logPath string, topN int, once bool) int {
	if interval < 3*time.Second {
		interval = 3 * time.Second // hard floor: never become the load we measure
	}
	if logPath == "" {
		logPath = defaultStallLogPath()
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		fmt.Fprintf(stderr, "fak stallscan: cannot create log dir: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "fak stallscan --watch: interval=%s log=%s (Ctrl-C to stop)\n", interval, logPath)

	tick := func() int {
		sample, gerr := gatherStallSample(topN)
		if gerr != "" {
			fmt.Fprintf(stderr, "fak stallscan: sample error: %s\n", gerr)
			return 1
		}
		v := stallscan.Classify(sample, stallscan.DefaultThresholds())
		appendStallJSONL(logPath, stallFingerprint(sample, v))
		if v.Level != stallscan.LevelCalm {
			fmt.Fprintf(stdout, "%s  %-8s %-18s %s\n",
				time.Now().UTC().Format("15:04:05"), v.Level, v.Cause, stallJoinReasons(v.Reasons))
		}
		return -1 // continue
	}

	if once {
		if rc := tick(); rc >= 0 {
			return rc
		}
		return 0
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	if rc := tick(); rc >= 0 { // sample immediately, don't wait a full interval
		return rc
	}
	for range t.C {
		if rc := tick(); rc >= 0 {
			return rc
		}
	}
	return 0
}

func stallJoinReasons(rs []string) string {
	out := ""
	for i, r := range rs {
		if i > 0 {
			out += "; "
		}
		out += r
	}
	return out
}

// appendStallJSONL appends one compact JSON line; best-effort by contract — a
// failed append must never crash the monitor.
func appendStallJSONL(path string, rec map[string]any) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	_, _ = f.Write(append(b, '\n'))
}

// defaultStallLogPath resolves a durable, host-local rolling log location.
func defaultStallLogPath() string {
	if d := os.Getenv("FAK_STALL_DIR"); d != "" {
		return filepath.Join(d, "stallscan.jsonl")
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" { // Windows host
		return filepath.Join(la, "Fleet", "stallscan.jsonl")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "stallscan.jsonl")
}

// renderStallFingerprint prints the human snapshot.
func renderStallFingerprint(w io.Writer, s stallscan.Sample, v stallscan.Verdict, topN int) {
	fmt.Fprintf(w, "stall level : %s\n", v.Level)
	fmt.Fprintf(w, "cause       : %s\n", v.Cause)
	fmt.Fprintf(w, "faults/sec  : %0.f total  (hard %0.f = %.1f%%, demand-zero %0.f, transition %0.f)\n",
		s.TotalFaultsPerSec, s.HardFaultsPerSec, stallPct(s.HardFaultsPerSec, s.TotalFaultsPerSec),
		s.DemandZeroFaultsPerSec, s.TransitionFaultsPerSec)
	fmt.Fprintf(w, "scheduler   : %0.f ctx-switch/sec, %0.f syscall/sec\n", s.ContextSwitchesPerSec, s.SystemCallsPerSec)
	fmt.Fprintf(w, "census      : %d procs, %d threads (delta %+d)\n", s.ProcessCount, s.ThreadCount, s.ProcessDelta)
	fmt.Fprintf(w, "not-the-cause: %d MB RAM free, disk queue %.1f\n", s.AvailableMB, s.DiskQueueLen)
	if len(v.Reasons) > 0 {
		fmt.Fprintf(w, "reasons     : %s\n", stallJoinReasons(v.Reasons))
	}
	top := stallscan.SortTopIO(s.TopIO, topN)
	if len(top) > 0 {
		fmt.Fprintf(w, "top IO-ops/sec:\n")
		for _, p := range top {
			fmt.Fprintf(w, "  %-24s pid %-7d %0.f ops/s\n", p.Name, p.PID, p.Ops)
		}
	}
	if v.Level == stallscan.LevelStall {
		fmt.Fprintf(w, "\nVERDICT: a stall is in progress — see `fak stallscan --watch` to record recurrence.\n")
	}
}

func stallPct(a, b float64) float64 {
	if b <= 0 {
		return 0
	}
	return 100 * a / b
}

// sortProcIOByOps is a small local helper for tests/renderers that need the raw
// slice sorted without the cap SortTopIO applies.
func sortProcIOByOps(in []stallscan.ProcIO) []stallscan.ProcIO {
	cp := make([]stallscan.ProcIO, len(in))
	copy(cp, in)
	sort.SliceStable(cp, func(i, j int) bool { return cp[i].Ops > cp[j].Ops })
	return cp
}
