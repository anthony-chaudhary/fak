package gpulease

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// holdExclusiveLikeARealHolder locks path exclusively and writes the same holder
// pid sidecar a real Acquire would, returning the open file so the caller can
// release it. This reproduces a live holder without forking.
func holdExclusiveLikeARealHolder(t *testing.T, path string, pid int, acquiredAt time.Time) *os.File {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open holder: %v", err)
	}
	if err := flock.TryLock(f); err != nil {
		f.Close()
		t.Fatalf("lock holder: %v", err)
	}
	if _, err := f.WriteAt([]byte(strconvItoa(pid)+"\n"), 0); err != nil {
		f.Close()
		t.Fatalf("write holder pid: %v", err)
	}
	writeHolderMeta(path, pid, acquiredAt)
	return f
}

func strconvItoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// release removes the sidecar and drops the lock, in the order Release uses.
func release(t *testing.T, path string, holder *os.File) {
	t.Helper()
	clearHolderMeta(path)
	_ = flock.Unlock(holder)
	holder.Close()
}

// progressingCPU returns a CPUTime func whose reading advances by step on every
// second call, modelling a holder doing real work across the sample window.
func progressingCPU(step time.Duration) func(int) (time.Duration, bool) {
	var call int
	return func(int) (time.Duration, bool) {
		call++
		return time.Duration(call) * step, true
	}
}

// TestHolderProgressLiveProgressing proves the load-bearing case from #13131: an
// ALIVE holder whose CPU time advances across the bounded window is
// LIVE_PROGRESSING, so the refusal names a legitimate peer bench instead of
// advising a kill.
func TestHolderProgressLiveProgressing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "live.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	defer release(t, path, holder)

	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:        path,
		Window:      time.Millisecond,
		MinCPUTicks: time.Millisecond,
		Alive:       func(int) bool { return true },
		CPUTime:     progressingCPU(20 * time.Millisecond),
	})
	if probe.Verdict != HolderProgressLiveProgressing {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressLiveProgressing, probe.Detail)
	}
	if !probe.Held {
		t.Fatal("probe.Held = false, want true")
	}
	if probe.PID != os.Getpid() {
		t.Fatalf("probe.PID = %d, want %d", probe.PID, os.Getpid())
	}
	if probe.CPUDelta != 20*time.Millisecond {
		t.Fatalf("probe.CPUDelta = %v, want 20ms", probe.CPUDelta)
	}
}

// TestHolderProgressStalled proves an ALIVE holder that consumed no CPU across a
// window at or above the decisive bound is STALLED — the one case where release
// advice is safe.
func TestHolderProgressStalled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stalled.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	defer release(t, path, holder)

	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:        path,
		Window:      decisiveStallWindow,
		MinCPUTicks: time.Millisecond,
		Alive:       func(int) bool { return true },
		CPUTime:     func(int) (time.Duration, bool) { return 42 * time.Second, true },
		Sleep:       func(time.Duration) {},
	})
	if probe.Verdict != HolderProgressStalled {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressStalled, probe.Detail)
	}
}

// TestHolderProgressQuietHolderBelowDecisiveWindowIsUnknown is the anti-libel
// guard: an alive holder with NO CPU advance across a short window must be
// UNKNOWN, because that is indistinguishable from a healthy Metal GPU wait. Only
// a decisive window may promote quiet to STALLED.
func TestHolderProgressQuietHolderBelowDecisiveWindowIsUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quiet.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	defer release(t, path, holder)

	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:        path,
		Window:      time.Millisecond,
		MinCPUTicks: time.Second,
		Alive:       func(int) bool { return true },
		CPUTime:     func(int) (time.Duration, bool) { return 42 * time.Second, true },
	})
	if probe.Verdict != HolderProgressUnknown {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressUnknown, probe.Detail)
	}
}

// TestHolderProgressDead proves a recorded holder pid the process table no longer
// knows is DEAD.
func TestHolderProgressDead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dead.lease")
	holder := holdExclusiveLikeARealHolder(t, path, 424242, time.Now())
	defer release(t, path, holder)

	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:  path,
		Alive: func(pid int) bool { return false },
	})
	if probe.Verdict != HolderProgressDead {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressDead, probe.Detail)
	}
	if probe.PID != 424242 {
		t.Fatalf("probe.PID = %d, want 424242", probe.PID)
	}
}

// TestHolderProgressFailsClosedUnknownOnUnobservableCPUTime is the fail-closed
// contract: an alive holder whose CPU-time signal cannot be read MUST yield
// UNKNOWN, never a fabricated STALLED.
func TestHolderProgressFailsClosedUnknownOnUnobservableCPUTime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nocpu.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	defer release(t, path, holder)

	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:        path,
		Window:      time.Millisecond,
		MinCPUTicks: time.Second, // would be STALLED if a delta were fabricated
		Alive:       func(int) bool { return true },
		CPUTime:     func(int) (time.Duration, bool) { return 0, false },
	})
	if probe.Verdict != HolderProgressUnknown {
		t.Fatalf("verdict = %q, want fail-closed %q (detail=%q)", probe.Verdict, HolderProgressUnknown, probe.Detail)
	}
}

// TestHolderProgressFailsClosedUnknownOnCPUGoingBackwards proves a pid-reuse
// counter reset (CPU delta < 0) is unreadable evidence, not a stall.
func TestHolderProgressFailsClosedUnknownOnCPUGoingBackwards(t *testing.T) {
	path := filepath.Join(t.TempDir(), "backwards.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	defer release(t, path, holder)

	var call int
	probe := ProbeHolderProgress(HolderProgressOptions{
		Path:   path,
		Window: time.Millisecond,
		Alive:  func(int) bool { return true },
		CPUTime: func(int) (time.Duration, bool) {
			call++
			if call == 1 {
				return 10 * time.Second, true
			}
			return time.Second, true
		},
	})
	if probe.Verdict != HolderProgressUnknown {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressUnknown, probe.Detail)
	}
}

// TestHolderProgressUnknownOnUnreadableLockfile proves the other fail-closed
// rung: a lockfile path that cannot be opened yields UNKNOWN, not an error or a
// fabricated verdict.
func TestHolderProgressUnknownOnUnreadableLockfile(t *testing.T) {
	probe := ProbeHolderProgress(HolderProgressOptions{Path: t.TempDir()})
	if probe.Verdict != HolderProgressUnknown {
		t.Fatalf("verdict = %q, want %q (detail=%q)", probe.Verdict, HolderProgressUnknown, probe.Detail)
	}
}

// TestHolderProgressUnknownWhenLeaseFree proves a free lease is not reported as a
// dead holder — there is simply no holder to judge.
func TestHolderProgressUnknownWhenLeaseFree(t *testing.T) {
	path := filepath.Join(t.TempDir(), "free.lease")
	if _, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644); err != nil {
		t.Fatalf("open free lease: %v", err)
	}
	probe := ProbeHolderProgress(HolderProgressOptions{Path: path})
	if probe.Verdict != HolderProgressUnknown {
		t.Fatalf("verdict = %q, want %q", probe.Verdict, HolderProgressUnknown)
	}
	if probe.Held {
		t.Fatal("probe.Held = true for a free lease")
	}
}

// TestAcquireWritesAndReleaseClearsHolderMeta proves the sidecar lifecycle is
// wired to the real Acquire/Release, so a refused peer has a holder identity to
// read and a freed lease never carries a stale one.
func TestAcquireWritesAndReleaseClearsHolderMeta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lifecycle.lease")
	lease, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if _, err := os.Stat(HolderMetaPath(path)); err != nil {
		t.Fatalf("sidecar missing after Acquire: %v", err)
	}
	// The live holder is this very process: an idle go test process may show no
	// CPU advance over the short default window, so the honest verdict here is
	// LIVE_PROGRESSING (CPU advanced) OR UNKNOWN (too short to tell) — never a
	// fabricated DEAD or STALLED.
	probe := ProbeHolderProgress(HolderProgressOptions{Path: path})
	if probe.Verdict != HolderProgressLiveProgressing && probe.Verdict != HolderProgressUnknown {
		t.Fatalf("self-held lease verdict = %q, want %q or %q (detail=%q)",
			probe.Verdict, HolderProgressLiveProgressing, HolderProgressUnknown, probe.Detail)
	}
	if probe.PID != os.Getpid() {
		t.Fatalf("self-held probe.PID = %d, want %d", probe.PID, os.Getpid())
	}
	lease.Release()
	if _, err := os.Stat(HolderMetaPath(path)); !os.IsNotExist(err) {
		t.Fatalf("sidecar survived Release: err=%v", err)
	}
}

// TestHolderProgressNeverMutatesHolder proves the probe is read-only: it must not
// drop the holder's flock nor delete the holder's sidecar.
func TestHolderProgressNeverMutatesHolder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "readonly.lease")
	holder := holdExclusiveLikeARealHolder(t, path, os.Getpid(), time.Now())
	meta := HolderMetaPath(path)
	before, err := os.Stat(meta)
	if err != nil {
		t.Fatalf("stat sidecar: %v", err)
	}

	_ = ProbeHolderProgress(HolderProgressOptions{
		Path:    path,
		Window:  time.Millisecond,
		Alive:   func(int) bool { return true },
		CPUTime: progressingCPU(10 * time.Millisecond),
	})

	if _, err := os.Stat(meta); err != nil {
		t.Fatalf("probe removed the holder sidecar: %v", err)
	}
	after, err := os.Stat(meta)
	if err != nil {
		t.Fatalf("re-stat sidecar: %v", err)
	}
	if !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("probe rewrote the holder sidecar: mtime %v -> %v", before.ModTime(), after.ModTime())
	}
	// The holder must still hold the lock: a fresh exclusive open must be refused.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open probe file: %v", err)
	}
	defer f.Close()
	if pid, busy := busyHolderPID(f); !busy {
		t.Fatalf("probe released the holder's lock (pid=%d, busy=false)", pid)
	}

	release(t, path, holder)
}

// TestParsePSDuration pins the BSD `ps -o time=` parser against the forms macOS
// renders, including the days and hours variants.
func TestParsePSDuration(t *testing.T) {
	tests := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"0:00.00", 0, true},
		{"0:01.50", 1500 * time.Millisecond, true},
		{"12:34", 12*time.Minute + 34*time.Second, true},
		{"1:02:03", time.Hour + 2*time.Minute + 3*time.Second, true},
		{"2-01:02:03", 48*time.Hour + time.Hour + 2*time.Minute + 3*time.Second, true},
		{"", 0, false},
		{"garbage", 0, false},
		{"1:2:3:4", 0, false},
	}
	for _, tc := range tests {
		got, ok := parsePSDuration(tc.in)
		if ok != tc.ok {
			t.Fatalf("parsePSDuration(%q) ok=%v, want %v", tc.in, ok, tc.ok)
		}
		if ok && got != tc.want {
			t.Fatalf("parsePSDuration(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
