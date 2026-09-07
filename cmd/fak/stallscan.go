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
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

func cmdStallscan(argv []string) { os.Exit(runStallscan(os.Stdout, os.Stderr, argv)) }

// runStallscan is the testable core. Exit 0 ok / calm, 1 runtime error, 2 usage,
// 3 a stall was observed, 4 a reboot is advised — a leak crossed the reboot
// high-water — so a script can gate on either (human snapshot mode; a stall
// outranks a reboot page, being the acute condition). --json carries the same
// verdict + reboot block in-band and returns 0/1 as before.
func runStallscan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("stallscan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the fingerprint as JSON")
	watch := fs.Bool("watch", false, "run a self-monitor loop: sample, classify, append JSONL, alert on each stall")
	interval := fs.Duration("interval", 15*time.Second, "sample interval in --watch mode (keep >= a few seconds; enumeration is expensive)")
	logPath := fs.String("log", "", "rolling JSONL path for --watch (default: <host stall dir>/stallscan.jsonl)")
	maxBytes := fs.Int64("max-bytes", 16<<20, "maximum watch log size before retaining the newest half (0 disables the bound)")
	topN := fs.Int("top", 6, "how many top-IO processes to show")
	once := fs.Bool("once", false, "in --watch mode, take exactly one sample then exit (for tests/cron)")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *watch {
		return runStallscanWatch(stdout, stderr, *interval, *logPath, *topN, *once, *maxBytes)
	}

	// Snapshot mode: one sample, classify, render.
	sample, err := gatherStallSample(*topN)
	if err != "" {
		fmt.Fprintf(stderr, "fak stallscan: %s\n", err)
		return 1
	}
	v := stallscan.Classify(sample, stallscan.DefaultThresholds())
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, stallFingerprintSkewed(sample, v, stallscanSkewGuard(stallscanBuildSkew())), "fak stallscan")
	}
	renderStallFingerprint(stdout, sample, v, *topN)
	if v.Level == stallscan.LevelStall {
		return 3
	}
	if adv := stallscan.AdviseReboot(sample, stallscan.DefaultRebootThresholds()); adv.Advised {
		return 4
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
		// The reboot-threshold decision travels in every record so the --watch
		// JSONL trail shows exactly when a leak crossed the reboot high-water, not
		// just that it was elevated. Pure over the same sample (issue #3668).
		"reboot": stallscan.AdviseReboot(s, stallscan.DefaultRebootThresholds()),
	}
}

// runStallscanWatch is the self-monitor loop. Cheap by construction: one sample
// per interval, append one JSONL line, print a one-liner only when the verdict
// is elevated/stall (calm ticks are logged but not printed, to stay quiet).
func runStallscanWatch(stdout, stderr io.Writer, interval time.Duration, logPath string, topN int, once bool, maxBytes int64) int {
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
	lock, err := acquireStallWatchLock(logPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak stallscan --watch: cannot own log %s: %v\n", logPath, err)
		return 1
	}
	defer func() {
		_ = flock.Unlock(lock)
		_ = lock.Close()
	}()
	watchID, err := newStallWatchID()
	if err != nil {
		fmt.Fprintf(stderr, "fak stallscan --watch: cannot create watch identity: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "fak stallscan --watch: interval=%s log=%s watch=%s (Ctrl-C to stop)\n", interval, logPath, watchID)

	// Per-PID first-seen census for the GROWTH (trajectory) axis: a leak is a
	// slope, not a level, so we compare each sample against the count a process
	// had when this monitor first observed it. This lets ClassifyWithBaseline warn
	// on a process that is CLIMBING while still under the absolute leak line —
	// robust to per-interval noise (the baseline is first-seen, not last-tick).
	baseHandles := map[int]stallscan.ProcHandles{}
	baseThreads := map[int]stallscan.ProcThreads{}
	started := time.Now().UTC()
	var sequence uint64

	tick := func() int {
		sample, gerr := gatherStallSample(topN)
		if gerr != "" {
			fmt.Fprintf(stderr, "fak stallscan: sample error: %s\n", gerr)
			return 1
		}
		baseline := updateGrowthBaseline(baseHandles, baseThreads, sample)
		v := stallscan.ClassifyWithBaseline(baseline, sample, stallscan.DefaultThresholds())
		sequence++
		rec := stallWatchRecord(
			stallFingerprintSkewed(sample, v, stallscanSkewGuard(stallscanBuildSkew())),
			watchID, sequence, started, time.Now().UTC(), interval,
		)
		if err := appendStallJSONL(logPath, rec, maxBytes); err != nil {
			fmt.Fprintf(stderr, "fak stallscan --watch: evidence write failed: %v\n", err)
			return 1
		}
		if v.Level != stallscan.LevelCalm {
			fmt.Fprintf(stdout, "%s  %-8s %-18s %s\n",
				time.Now().UTC().Format("15:04:05"), v.Level, v.Cause, stallJoinReasons(v.Reasons))
		}
		// A leak past the reboot high-water is the page the loop exists to catch:
		// prompt a reboot BEFORE the freeze, not after (issue #3668).
		if adv := stallscan.AdviseReboot(sample, stallscan.DefaultRebootThresholds()); adv.Advised {
			at := time.Now().UTC().Format("15:04:05")
			fmt.Fprintf(stdout, "%s  REBOOT   %s\n", at, adv.Reason)
			// A second process over the same line is its own driver of the reboot,
			// not a detail of the first — print every crosser the headline used to
			// mask, so the operator weighs all of them (issue #4614).
			for _, also := range adv.SecondaryCrossers() {
				fmt.Fprintf(stdout, "%s  REBOOT   also: %s\n", at, also.Reason)
			}
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

// updateGrowthBaseline maintains the per-PID first-seen census across --watch
// ticks and returns a baseline Sample for stallscan.ClassifyWithBaseline. A PID
// seen for the first time — or reused by a differently-named process — is
// recorded at its current count (so its climb is 0 this tick); a PID absent from
// the current sample is dropped so a later reuse of that number starts fresh. The
// returned Sample carries, for each process in cur, the count it had when first
// observed, aligned by PID so the classifier can diff them.
func updateGrowthBaseline(baseH map[int]stallscan.ProcHandles, baseT map[int]stallscan.ProcThreads, cur stallscan.Sample) stallscan.Sample {
	curH := make(map[int]bool, len(cur.TopHandles))
	for _, p := range cur.TopHandles {
		curH[p.PID] = true
		if b, ok := baseH[p.PID]; !ok || b.Name != p.Name {
			baseH[p.PID] = p // first sight (or PID reuse): baseline is now
		}
	}
	for pid := range baseH {
		if !curH[pid] {
			delete(baseH, pid)
		}
	}
	curT := make(map[int]bool, len(cur.TopThreads))
	for _, p := range cur.TopThreads {
		curT[p.PID] = true
		if b, ok := baseT[p.PID]; !ok || b.Name != p.Name {
			baseT[p.PID] = p
		}
	}
	for pid := range baseT {
		if !curT[pid] {
			delete(baseT, pid)
		}
	}
	out := stallscan.Sample{}
	for _, p := range cur.TopHandles {
		out.TopHandles = append(out.TopHandles, baseH[p.PID])
	}
	for _, p := range cur.TopThreads {
		out.TopThreads = append(out.TopThreads, baseT[p.PID])
	}
	return out
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

// acquireStallWatchLock gives one watcher exclusive ownership of a JSONL path.
func stallWatchRecord(rec map[string]any, id string, sequence uint64, started, sampled time.Time, interval time.Duration) map[string]any {
	rec["watch"] = map[string]any{
		"id": id, "sequence": sequence, "started_at": started,
		"sampled_at": sampled, "interval_ms": interval.Milliseconds(),
		"healthy": true,
	}
	return rec
}

func acquireStallWatchLock(path string) (*os.File, error) {
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := flock.TryLock(lock); err != nil {
		_ = lock.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, fmt.Errorf("another watcher owns this JSONL")
		}
		return nil, err
	}
	return lock, nil
}

func newStallWatchID() (string, error) {
	var id [8]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(id[:]), nil
}

var boundStallLogForWatch = boundStallLog

func appendStallJSONL(path string, rec map[string]any, maxBytes int64) error {
	b, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("encode record: %w", err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open log: %w", err)
	}
	if _, err = f.Write(append(b, '\n')); err != nil {
		_ = f.Close()
		return fmt.Errorf("append record: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close log: %w", err)
	}
	if maxBytes > 0 {
		if err := boundStallLogForWatch(path, maxBytes); err != nil {
			return fmt.Errorf("rotate log: %w", err)
		}
	}
	return nil
}

// boundStallLog prevents a long-lived watcher from becoming the resource leak it
// diagnoses. It preserves complete newest JSONL records via atomic replacement.
func boundStallLog(path string, maxBytes int64) error {
	info, err := os.Stat(path)
	if err != nil || info.Size() <= maxBytes {
		return err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	start := len(b) - int(maxBytes/2)
	if start < 0 {
		start = 0
	}
	if i := bytes.IndexByte(b[start:], '\n'); i >= 0 {
		start += i + 1
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".stallscan-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err = tmp.Write(b[start:]); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return err
	}
	_ = os.Remove(path) // Rename cannot replace an existing destination on Windows.
	return os.Rename(tmpPath, path)
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
	adv := stallscan.AdviseReboot(s, stallscan.DefaultRebootThresholds())
	fmt.Fprintf(w, "stall level : %s\n", v.Level)
	fmt.Fprintf(w, "cause       : %s\n", v.Cause)
	fmt.Fprintf(w, "faults/sec  : %0.f total  (hard %0.f = %.1f%%, demand-zero %0.f, transition %0.f)\n",
		s.TotalFaultsPerSec, s.HardFaultsPerSec, stallPct(s.HardFaultsPerSec, s.TotalFaultsPerSec),
		s.DemandZeroFaultsPerSec, s.TransitionFaultsPerSec)
	fmt.Fprintf(w, "scheduler   : %0.f ctx-switch/sec, %0.f syscall/sec\n", s.ContextSwitchesPerSec, s.SystemCallsPerSec)
	fmt.Fprintf(w, "census      : %d procs, %d threads (delta %+d)\n", s.ProcessCount, s.ThreadCount, s.ProcessDelta)
	if s.VRAMTotalBytes > 0 {
		vramPct := float64(s.VRAMCommittedBytes) / float64(s.VRAMTotalBytes) * 100
		fmt.Fprintf(w, "vram        : %d / %d MB committed (%.1f%%)",
			s.VRAMCommittedBytes/(1024*1024), s.VRAMTotalBytes/(1024*1024), vramPct)
		if s.VRAMSharedBytes > 0 {
			fmt.Fprintf(w, ", %d MB shared aperture", s.VRAMSharedBytes/(1024*1024))
		}
		if vramPct >= 90.0 {
			fmt.Fprintf(w, " — WARNING: VRAM committed approaches capacity (paging risk)")
		}
		fmt.Fprintf(w, "\n")
	}
	if s.SystemHandleTotal > 0 {
		fmt.Fprintf(w, "handles     : %d system-wide", s.SystemHandleTotal)
		if v.HandleLeakProcess != "" {
			fmt.Fprintf(w, "  — LEAK SUSPECT: %s pid %d holds %d handles", v.HandleLeakProcess, v.HandleLeakPID, v.HandleLeakCount)
		}
		fmt.Fprintf(w, "\n")
	}
	if v.ThreadLeakProcess != "" {
		fmt.Fprintf(w, "threads     : THREAD-LEAK SUSPECT (terminal lag): %s pid %d holds %d threads\n", v.ThreadLeakProcess, v.ThreadLeakPID, v.ThreadLeakCount)
	}
	if v.HandleGrowthProcess != "" {
		fmt.Fprintf(w, "handle-grow : LEAK TRAJECTORY: %s pid %d climbed +%d to %d handles since first seen\n", v.HandleGrowthProcess, v.HandleGrowthPID, v.HandleGrowthDelta, v.HandleGrowthCount)
	}
	if v.ThreadGrowthProcess != "" {
		fmt.Fprintf(w, "thread-grow : THREAD-LEAK TRAJECTORY: %s pid %d climbed +%d to %d threads since first seen\n", v.ThreadGrowthProcess, v.ThreadGrowthPID, v.ThreadGrowthDelta, v.ThreadGrowthCount)
	}
	if adv.Advised {
		fmt.Fprintf(w, "reboot      : ADVISED (%s axis): %s pid %d at %d (>= %d high-water)\n", adv.Axis, adv.Process, adv.PID, adv.Count, adv.Threshold)
		for _, also := range adv.SecondaryCrossers() {
			fmt.Fprintf(w, "              ALSO (%s axis): %s pid %d at %d (>= %d high-water)\n", also.Axis, also.Process, also.PID, also.Count, also.Threshold)
		}
	}
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
	} else if adv.Advised {
		fmt.Fprintf(w, "\nVERDICT: %s\n", adv.Reason)
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

func stallFingerprintSkewed(s stallscan.Sample, v stallscan.Verdict, skew *stallBuildSkew) map[string]any {
	rec := stallFingerprint(s, v)
	if skew != nil {
		rec["build_skew"] = skew
	}
	return rec
}
