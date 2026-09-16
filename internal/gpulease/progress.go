package gpulease

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// HolderProgress is the typed, READ-ONLY verdict on whether the process holding
// the GPU lease is a healthy long-running GPU job or a wedged one (#13131).
//
// Motivation. The lease is an flock and the lockfile carries only the holder's
// pid, so a refused `fak up` could say nothing but "stop the holder process and
// retry". On 2026-09-15 an M3 Pro `modelbench` prefill sweep legitimately held
// the lease for >17 minutes at 66% device utilization while `fak up` was refused
// 14 consecutive times; following the shipped advice would have killed a valid
// benchmark. This verdict is what lets the refusal name a progressing holder as
// legitimate instead of advising an unconditional kill.
//
// Fail-closed. Every rung below withholds evidence rather than fabricating it: an
// unreadable, raced, or metadata-free lockfile yields HolderProgressUnknown, NEVER
// a fabricated HolderProgressStalled. The probe NEVER kills, signals, or mutates
// the holder; it only reads.
type HolderProgress string

const (
	// HolderProgressLiveProgressing: the recorded holder is alive AND its CPU time
	// advanced across the sampling window — a side effect of real work the holder
	// cannot fake by asserting it. A peer bench holding the lease here is
	// legitimate; wait for it rather than killing it.
	HolderProgressLiveProgressing HolderProgress = "LIVE_PROGRESSING"
	// HolderProgressStalled: the recorded holder is alive but consumed no CPU over
	// a window long enough to be decisive. This is reachable ONLY when the caller
	// supplies a window at or above the CPU-time signal's resolution; the default
	// probe returns UNKNOWN instead, because a short CPU quiet spell is exactly
	// what a healthy Metal-waiting holder looks like and must not be libelled.
	HolderProgressStalled HolderProgress = "STALLED"
	// HolderProgressDead: the recorded pid no longer identifies a running process,
	// so the OS is about to (or already did) drop the flock. Advice to release is
	// safe here.
	HolderProgressDead HolderProgress = "DEAD"
	// HolderProgressUnknown: the lock is held but no progress evidence could be
	// read (raced/unreadable lockfile, no pid, or an unobservable CPU-time
	// signal). This is the fail-closed default and must never be upgraded to
	// STALLED on missing data.
	HolderProgressUnknown HolderProgress = "UNKNOWN"
)

const (
	// DefaultProgressWindow is the bounded interval over which the holder's CPU
	// time is sampled. It is short enough that a `fak up` refusal is not visibly
	// delayed, and long enough for a working holder to accumulate a resolvable
	// amount of CPU.
	//
	// The window is deliberately NOT long enough to declare a quiet holder
	// stalled. macOS `ps -o time=` resolves to centiseconds (10ms), and the
	// 2026-09-15 holder's CPU fluctuated 0–38% between GPU waits — so a short CPU
	// quiet spell is what a HEALTHY Metal-waiting holder looks like. The default
	// probe therefore reports UNKNOWN for a quiet-but-alive holder rather than a
	// fabricated STALLED.
	DefaultProgressWindow = 250 * time.Millisecond
	// DefaultMinCPUTicks is the cumulative CPU time that must elapse across the
	// window for the holder to count as evidently progressing.
	DefaultMinCPUTicks = 10 * time.Millisecond
	// decisiveStallWindow is the minimum sampling window at which a ZERO CPU
	// delta is decisive enough to call a holder stalled. Below it, a zero delta
	// fails closed to UNKNOWN. It is an order of magnitude above the signal's
	// 10ms resolution and far above any healthy per-GPU-wait quiet spell.
	decisiveStallWindow = 30 * time.Second
)

// HolderMetaSuffix is the sidecar the exclusive holder writes beside its lockfile
// with its pid and acquire stamp. It exists because the flock lockfile itself must
// carry ONLY the pid (readHolderPID, and the shared-reader path never writes it),
// so the holder's identity and age live in a separate record that a read-only
// probe can parse.
const HolderMetaSuffix = ".holder"

// HolderMetaPath returns the sidecar path Acquire writes for the lockfile at
// path.
func HolderMetaPath(path string) string {
	if path == "" {
		path = DefaultPath()
	}
	return path + HolderMetaSuffix
}

// HolderProgressOptions tunes a ProbeHolderProgress call. The zero value is
// correct for production: the default bounded window, the real clock, the real
// CPU-time reader, and the real liveness probe.
type HolderProgressOptions struct {
	// Path is the lockfile to inspect. Empty means DefaultPath().
	Path string
	// Window overrides DefaultProgressWindow. Zero uses the default; negative
	// degenerates the verdict to liveness only (LIVE_PROGRESSING for any alive
	// holder) — an explicit opt-out for callers that cannot afford to sample.
	Window time.Duration
	// MinCPUTicks overrides DefaultMinCPUTicks.
	MinCPUTicks time.Duration
	// Alive overrides the liveness probe (tests only; nil means processalive.Check).
	Alive func(pid int) bool
	// CPUTime overrides the cumulative-CPU-time reader (tests only; nil means the
	// /bin/ps-backed reader capped by Window).
	CPUTime func(pid int) (time.Duration, bool)
	// Sleep overrides the inter-sample wait (tests only; nil means time.Sleep).
	// It lets a test exercise a decisive-window verdict without waiting out the
	// window for real; production always uses the real clock.
	Sleep func(time.Duration)
}

// HolderProbe is the evidence behind a HolderProgress verdict, carried so a
// surface can show WHY the verdict was reached rather than only the word.
type HolderProbe struct {
	// Verdict is the typed, fail-closed holder-progress verdict.
	Verdict HolderProgress `json:"verdict"`
	// Held reports whether the lease is currently locked by an exclusive holder.
	// False means the free-lease case (Verdict is UNKNOWN and PID is HolderNotBusy).
	Held bool `json:"held"`
	// PID is the recorded holder pid, or HolderUnknownPID when unreadable.
	PID int `json:"pid"`
	// CPUDelta is the CPU time the holder consumed across the bounded window, or
	// 0 when the signal was unobservable.
	CPUDelta time.Duration `json:"cpu_delta,omitempty"`
	// Detail is a one-line human explanation of the evidence considered.
	Detail string `json:"detail,omitempty"`
}

// ProbeHolderProgress probes the holder of the lease at opts.Path and returns the
// typed, fail-closed verdict plus the evidence behind it.
//
// It is READ-ONLY: it takes no lock beyond a probe that it immediately drops, and
// it never signals or kills the holder. The lease acquisition semantics
// (exclusive/shared, NoWait, PID metadata) are untouched.
func ProbeHolderProgress(opts HolderProgressOptions) HolderProbe {
	aliveFn := processalive.Check
	if opts.Alive != nil {
		aliveFn = opts.Alive
	}
	path := opts.Path
	if path == "" {
		path = DefaultPath()
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, Detail: "lockfile unreadable: " + err.Error()}
	}
	defer f.Close()

	// The lease must actually be held by an exclusive holder for any holder
	// verdict to be meaningful. A free lease is not a "dead holder".
	pid, busy := busyHolderPID(f)
	if !busy {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: false, PID: HolderNotBusy, Detail: "lease is free"}
	}
	if pid == HolderSharedReaders {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: pid, Detail: "held by shared inspection readers; no exclusive holder to judge"}
	}
	if pid < 0 {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: pid, Detail: "exclusive lock held but holder pid is unreadable"}
	}

	// Liveness first: a pid the process table no longer knows is DEAD regardless
	// of any progress reading.
	if !aliveFn(pid) {
		return HolderProbe{Verdict: HolderProgressDead, Held: true, PID: pid, Detail: "recorded holder pid is not a running process"}
	}
	if opts.Window < 0 {
		return HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: pid, Detail: "progress rung disabled; holder pid is alive"}
	}

	// Progress second: sample the holder's cumulative CPU time across a bounded
	// window. Advancing CPU is a side effect of real work the holder cannot fake
	// by asserting it; an unobservable reading fails closed to UNKNOWN.
	window := opts.Window
	if window == 0 {
		window = DefaultProgressWindow
	}
	minTicks := opts.MinCPUTicks
	if minTicks == 0 {
		minTicks = DefaultMinCPUTicks
	}
	cpuFn := opts.CPUTime
	if cpuFn == nil {
		cpuFn = func(p int) (time.Duration, bool) { return readProcCPUTime(p) }
	}

	sleep := time.Sleep
	if opts.Sleep != nil {
		sleep = opts.Sleep
	}

	before, ok := cpuFn(pid)
	if !ok {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: pid, Detail: "holder pid is alive but its CPU-time signal is unobservable"}
	}
	sleep(window)
	after, ok := cpuFn(pid)
	if !ok {
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: pid, Detail: "holder pid is alive but its CPU-time signal became unobservable mid-sample"}
	}
	delta := after - before
	if delta < 0 {
		// A pid reused by a different process can reset the counter; that is
		// unreadable evidence, not a stall.
		return HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: pid, Detail: "holder pid CPU-time counter went backwards (pid reuse?); evidence unreadable"}
	}

	probe := HolderProbe{Held: true, PID: pid, CPUDelta: delta}
	if delta >= minTicks {
		probe.Verdict = HolderProgressLiveProgressing
		probe.Detail = "holder pid is alive and consumed " + delta.Round(time.Millisecond).String() + " CPU over " + window.String()
		return probe
	}
	// A zero/short delta is NOT proof of a wedge: a healthy holder blocked in a
	// Metal GPU wait legitimately shows no CPU. Only a window at or above the
	// decisive stall bound turns a quiet holder into a STALLED verdict; below it
	// the honest answer is UNKNOWN.
	if window < decisiveStallWindow {
		probe.Verdict = HolderProgressUnknown
		probe.Detail = "holder pid is alive and consumed only " + delta.Round(time.Millisecond).String() + " CPU over " + window.String() + " — too short to tell a GPU wait from a wedge"
		return probe
	}
	probe.Verdict = HolderProgressStalled
	probe.Detail = "holder pid is alive but consumed only " + delta.Round(time.Millisecond).String() + " CPU over " + window.String()
	return probe
}

// readProcCPUTime reads pid's cumulative CPU time via /bin/ps. It is read-only
// and never signals the process: `ps -o time=` is an OS accounting read, the
// no-spawn-preferring alternative being a per-platform syscall this package does
// not need to grow for a bounded sample.
func readProcCPUTime(pid int) (time.Duration, bool) {
	if pid <= 0 {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "/bin/ps", "-o", "time=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		// A non-zero exit usually means the pid vanished mid-read; unobservable.
		return 0, false
	}
	return parsePSDuration(strings.TrimSpace(string(out)))
}

// parsePSDuration parses BSD `ps -o time=` output. macOS renders either
// [[dd-]hh:]mm:ss (with hundredths on some versions) or a plain seconds form, so
// the parser accepts the colon form and a bare fractional-seconds fallback.
func parsePSDuration(s string) (time.Duration, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	days := 0
	if i := strings.IndexByte(s, '-'); i >= 0 {
		d, err := strconv.Atoi(s[:i])
		if err != nil {
			return 0, false
		}
		days = d
		s = s[i+1:]
	}
	parts := strings.Split(s, ":")
	if len(parts) < 2 || len(parts) > 3 {
		return 0, false
	}
	var hours, minutes int
	var seconds float64
	secIdx := len(parts) - 1
	for i, p := range parts {
		if i == secIdx {
			v, err := strconv.ParseFloat(p, 64)
			if err != nil {
				return 0, false
			}
			seconds = v
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return 0, false
		}
		if i == 0 && len(parts) == 3 {
			hours = v
		} else {
			minutes = v
		}
	}
	total := time.Duration(days)*24*time.Hour +
		time.Duration(hours)*time.Hour +
		time.Duration(minutes)*time.Minute +
		time.Duration(seconds*float64(time.Second))
	return total, true
}

// writeHolderMeta records this process as the exclusive holder in the sidecar
// beside the lockfile. Best-effort: a caller that fails to write it still holds
// the flock, and a later probe simply fails closed to UNKNOWN.
func writeHolderMeta(lockPath string, pid int, now time.Time) {
	rec := "pid=" + strconv.Itoa(pid) + "\nstarted_unix=" + strconv.FormatInt(now.Unix(), 10) + "\n"
	// 0o644 matches the lockfile; the record is identity, not a secret.
	_ = os.WriteFile(HolderMetaPath(lockPath), []byte(rec), 0o644)
}

// clearHolderMeta removes the sidecar on release so a freed lease never carries a
// stale holder identity into the next acquirer's probe.
func clearHolderMeta(lockPath string) {
	if err := os.Remove(HolderMetaPath(lockPath)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return
	}
}
