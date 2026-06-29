package gpulease

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestNoWaitBusyThenFree proves the core invariant the panic fix relies on: while
// one lease is held, a second NoWait Acquire is refused (ErrBusy), and once the
// first is released the second succeeds. flock treats separate opens of the same
// file independently, so two Acquire calls in one process contend exactly as two
// processes would.
func TestNoWaitBusyThenFree(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	a, err := Acquire(Options{Path: path})
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}

	if _, err := Acquire(Options{Path: path, NoWait: true}); !errors.Is(err, ErrBusy) {
		t.Fatalf("second acquire while held: got %v, want ErrBusy", err)
	}

	a.Release()

	b, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	b.Release()
}

// TestWaitTimesOut proves a waiting Acquire honors its Timeout and emits exactly one
// waiting notice (so a queued bench is observable, not silent).
func TestWaitTimesOut(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	held, err := Acquire(Options{Path: path})
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	defer held.Release()

	var notices int
	start := time.Now()
	_, err = Acquire(Options{
		Path:      path,
		Timeout:   60 * time.Millisecond,
		pollEvery: 10 * time.Millisecond,
		Logf:      func(string, ...any) { notices++ },
	})
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("waiting acquire: got %v, want ErrTimeout", err)
	}
	if time.Since(start) < 60*time.Millisecond {
		t.Fatalf("returned before the timeout elapsed")
	}
	if notices != 1 {
		t.Fatalf("waiting notices = %d, want exactly 1", notices)
	}
}

// TestWaitThenSucceed covers the queue's happy path (the actual point of the lease):
// a blocking Acquire that has to WAIT and then WINS once the holder releases — the
// branch TestWaitTimesOut does not reach.
func TestWaitThenSucceed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	held, err := Acquire(Options{Path: path})
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	go func() {
		time.Sleep(40 * time.Millisecond)
		held.Release()
	}()

	start := time.Now()
	l, err := Acquire(Options{Path: path, pollEvery: 5 * time.Millisecond, Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("waiting acquire: %v", err)
	}
	defer l.Release()
	if time.Since(start) < 35*time.Millisecond {
		t.Fatalf("acquired before the holder released (%v)", time.Since(start))
	}
}

// TestReleaseOnProcessExit proves the invariant the whole fix rests on: when a holding
// PROCESS exits without calling Release, the OS drops the flock so the next process can
// take the lease. It re-execs the test binary as a child that acquires then exits; the
// parent then must be able to acquire. If the flock leaked past process death, the
// parent's NoWait acquire would fail with ErrBusy.
func TestReleaseOnProcessExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	if p := os.Getenv("GPULEASE_HELPER_PATH"); p != "" {
		// Child: acquire (proving we genuinely held it) and exit without Release.
		if _, err := Acquire(Options{Path: p, NoWait: true}); err != nil {
			os.Stderr.WriteString("child acquire failed: " + err.Error() + "\n")
			os.Exit(3)
		}
		os.Stdout.WriteString("ACQUIRED\n")
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReleaseOnProcessExit")
	cmd.Env = append(os.Environ(), "GPULEASE_HELPER_PATH="+path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "ACQUIRED") {
		t.Fatalf("child did not acquire the lease; output:\n%s", out)
	}

	// Child has exited; the flock it held must be gone.
	l, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("lease not released on child process exit: %v", err)
	}
	l.Release()
}

// deadPID returns a pid that is (almost certainly) not a live process, by spawning a
// trivial child and waiting for it to exit. The kernel will not have recycled the
// number by the time the test reads it microseconds later.
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=NoSuchTestZZZ")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn helper: %v", err)
	}
	pid := cmd.Process.Pid
	_ = cmd.Wait() // reap; the pid is now dead
	return pid
}

// TestStealStaleHolder proves the field-bug fix: a lockfile that records a DEAD holder
// pid but carries no live OS lock (the Windows abnormal-exit orphan state) must be
// stolen by the next Acquire instead of wedging it forever. Before the steal, this
// configuration left a NoWait Acquire returning ErrBusy on Windows for as long as the
// orphaned LockFileEx region survived (~56 min in the field); after it, Acquire breaks
// the stale lock and succeeds.
func TestStealStaleHolder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	// Seed a stale lockfile: a dead pid, no flock held (file exists, unlocked).
	if err := os.WriteFile(path, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatalf("seed stale lockfile: %v", err)
	}

	l, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("stale lock not stolen: %v", err)
	}
	defer l.Release()

	// And the steal recorded OUR pid, so a later waiter names us, not the dead holder.
	got, _ := os.ReadFile(path)
	if want := strconv.Itoa(os.Getpid()); strings.TrimSpace(string(got)) != want {
		t.Fatalf("after steal, lockfile pid = %q, want %q", strings.TrimSpace(string(got)), want)
	}
}

// TestNoStealFromLiveHolder proves the steal's safety gate: while a LIVE process (this
// one) genuinely holds the lock, a second NoWait Acquire must still be refused — the
// stale-holder steal must NEVER break a lock whose recorded pid is alive, or two
// concurrent GPU jobs would stack and reproduce the jetsam cascade the lease prevents.
func TestNoStealFromLiveHolder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	held, err := Acquire(Options{Path: path}) // records THIS live pid, holds the flock
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	defer held.Release()

	if _, err := Acquire(Options{Path: path, NoWait: true}); !errors.Is(err, ErrBusy) {
		t.Fatalf("second acquire while a LIVE holder owns the lock: got %v, want ErrBusy", err)
	}
}

// TestReleaseIdempotent guards the double-Release / nil paths.
func TestReleaseIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	l, err := Acquire(Options{Path: path})
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	l.Release()
	l.Release() // no-op, must not panic
	var nilLease *Lease
	nilLease.Release() // no-op, must not panic
}
