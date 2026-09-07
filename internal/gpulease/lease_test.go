package gpulease

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const crossProcessLeaseModeEnv = "GPULEASE_CROSS_PROCESS_MODE"

type leaseProcess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stderr *strings.Builder
}

func startLeaseProcess(t *testing.T, path string, mode Mode) *leaseProcess {
	t.Helper()

	cmd := exec.Command(os.Args[0], "-test.run=^TestCrossProcessSharedExclusiveAdmission$")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("helper stdin: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("helper stdout: %v", err)
	}
	stderr := new(strings.Builder)
	cmd.Stderr = stderr
	cmd.Env = append(os.Environ(),
		crossProcessLeaseModeEnv+"="+mode.String(),
		"GPULEASE_CROSS_PROCESS_PATH="+path,
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start %s helper: %v", mode, err)
	}

	p := &leaseProcess{cmd: cmd, stdin: stdin, stderr: stderr}
	t.Cleanup(func() {
		if p.cmd != nil {
			_ = p.cmd.Process.Kill()
			_ = p.cmd.Wait()
			p.cmd = nil
		}
	})
	ready, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || strings.TrimSpace(ready) != "READY" {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		p.cmd = nil
		t.Fatalf("%s helper readiness: line=%q err=%v stderr=%q", mode, ready, err, stderr.String())
	}
	return p
}

func (p *leaseProcess) stop(t *testing.T) {
	t.Helper()
	if p == nil || p.cmd == nil {
		return
	}
	if err := p.stdin.Close(); err != nil {
		t.Fatalf("close helper stdin: %v", err)
	}
	if err := p.cmd.Wait(); err != nil {
		t.Fatalf("helper exit: %v stderr=%q", err, p.stderr.String())
	}
	p.cmd = nil
}

func (p *leaseProcess) kill(t *testing.T) {
	t.Helper()
	if p == nil || p.cmd == nil {
		return
	}
	if _, err := p.stdin.Write([]byte{'K'}); err != nil {
		t.Fatalf("request abrupt helper exit: %v", err)
	}
	_ = p.stdin.Close()
	if err := p.cmd.Wait(); err == nil {
		t.Fatal("abrupt helper exit reported success")
	}
	p.cmd = nil
}

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
	} else {
		var busy *BusyError
		if !errors.As(err, &busy) {
			t.Fatalf("second acquire error type = %T, want *BusyError", err)
		}
		if busy.Path != path || busy.PID != os.Getpid() {
			t.Fatalf("busy metadata = path %q pid %d, want path %q pid %d", busy.Path, busy.PID, path, os.Getpid())
		}
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

	waitedCalled := false
	l, err := Acquire(Options{
		Path:      path,
		pollEvery: 2 * time.Millisecond,
		Timeout:   5 * time.Second,
		Logf: func(format string, args ...any) {
			waitedCalled = true
			held.Release()
		},
	})
	if err != nil {
		t.Fatalf("waiting acquire: %v", err)
	}
	defer l.Release()
	if !waitedCalled {
		t.Fatal("acquired without waiting for the holder to release")
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

// TestSharedReadersConcurrent proves invariant (a): concurrent shared readers
// share the machine lock simultaneously while excluding exclusive writers.
func TestSharedReadersConcurrent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	// Acquire first shared reader via AcquireShared
	r1, err := AcquireShared(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("r1 AcquireShared: %v", err)
	}
	defer r1.Release()

	if !r1.Shared() || r1.Mode() != ModeShared {
		t.Fatalf("r1: Shared()=%v Mode()=%v, want true/ModeShared", r1.Shared(), r1.Mode())
	}

	// Acquire second shared reader via Mode: ModeShared
	r2, err := Acquire(Options{Path: path, Mode: ModeShared, NoWait: true})
	if err != nil {
		t.Fatalf("r2 Acquire(ModeShared): %v", err)
	}
	defer r2.Release()

	if !r2.Shared() || r2.Mode() != ModeShared {
		t.Fatalf("r2: Shared()=%v Mode()=%v, want true/ModeShared", r2.Shared(), r2.Mode())
	}

	// Acquire third shared reader via Shared: true
	r3, err := Acquire(Options{Path: path, Shared: true, NoWait: true})
	if err != nil {
		t.Fatalf("r3 Acquire(Shared:true): %v", err)
	}
	defer r3.Release()

	// While readers are holding the lease simultaneously, exclusive Acquire must be refused.
	if _, err := Acquire(Options{Path: path, NoWait: true}); !errors.Is(err, ErrBusy) {
		t.Fatalf("exclusive acquire while shared readers hold lease: want ErrBusy, got %v", err)
	}

	// Release all readers
	r1.Release()
	r2.Release()
	r3.Release()

	// Exclusive writer can now succeed immediately.
	w, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("exclusive acquire after all readers release: %v", err)
	}
	if w.Shared() || w.Mode() != ModeExclusive {
		t.Fatalf("exclusive lease: Shared()=%v Mode()=%v, want false/ModeExclusive", w.Shared(), w.Mode())
	}
	w.Release()
}

// TestExclusiveWriterBlocksUntilAllReadersRelease proves invariant (b): an exclusive
// writer blocks until ALL concurrent shared readers release the lock.
func TestExclusiveWriterBlocksUntilAllReadersRelease(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	r1, err := AcquireShared(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("r1 acquire: %v", err)
	}
	defer r1.Release()

	r2, err := AcquireShared(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("r2 acquire: %v", err)
	}
	defer r2.Release()

	type result struct {
		lease *Lease
		err   error
	}
	writerCh := make(chan result, 1)
	waitingCh := make(chan struct{})
	var waitOnce sync.Once

	go func() {
		w, err := Acquire(Options{
			Path:      path,
			pollEvery: 2 * time.Millisecond,
			Timeout:   5 * time.Second,
			Logf: func(format string, args ...any) {
				waitOnce.Do(func() { close(waitingCh) })
			},
		})
		writerCh <- result{lease: w, err: err}
	}()

	// Wait until writer enters the waiting state.
	select {
	case <-waitingCh:
	case <-time.After(5 * time.Second):
		t.Fatal("writer never logged waiting notice")
	}

	// Verify writer has not acquired while both readers hold lease.
	select {
	case res := <-writerCh:
		t.Fatalf("writer acquired prematurely while readers active: %v", res.err)
	default:
	}

	// Release only the first reader; r2 still holds.
	r1.Release()

	// Verify writer still has not acquired because r2 is still held.
	time.Sleep(20 * time.Millisecond)
	select {
	case res := <-writerCh:
		t.Fatalf("writer acquired while r2 still held lease: %v", res.err)
	default:
	}

	// Release the second reader; now writer must acquire.
	r2.Release()

	select {
	case res := <-writerCh:
		if res.err != nil {
			t.Fatalf("writer acquire failed: %v", res.err)
		}
		defer res.lease.Release()
		if res.lease.Shared() || res.lease.Mode() != ModeExclusive {
			t.Fatal("writer lease is not exclusive")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("writer timed out after all readers released")
	}
}

// TestSharedReaderBlocksBehindExclusiveWriter proves invariant (c): a shared reader
// blocks behind an active exclusive writer and proceeds only when the writer releases.
func TestSharedReaderBlocksBehindExclusiveWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	w, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("exclusive acquire: %v", err)
	}
	defer w.Release()

	// Immediate NoWait shared acquire must fail with ErrBusy.
	if _, err := AcquireShared(Options{Path: path, NoWait: true}); !errors.Is(err, ErrBusy) {
		t.Fatalf("shared acquire while writer active: want ErrBusy, got %v", err)
	} else {
		var busy *BusyError
		if !errors.As(err, &busy) {
			t.Fatalf("expected *BusyError, got %T", err)
		}
		if busy.PID != os.Getpid() {
			t.Fatalf("busy PID = %d, want %d", busy.PID, os.Getpid())
		}
	}

	type result struct {
		lease *Lease
		err   error
	}
	readerCh := make(chan result, 1)
	waitingCh := make(chan struct{})
	var waitOnce sync.Once

	go func() {
		r, err := AcquireShared(Options{
			Path:      path,
			pollEvery: 2 * time.Millisecond,
			Timeout:   5 * time.Second,
			Logf: func(format string, args ...any) {
				waitOnce.Do(func() { close(waitingCh) })
			},
		})
		readerCh <- result{lease: r, err: err}
	}()

	// Wait until reader enters waiting state.
	select {
	case <-waitingCh:
	case <-time.After(5 * time.Second):
		t.Fatal("reader never logged waiting notice")
	}

	// Verify reader has not acquired while writer holds lease.
	select {
	case res := <-readerCh:
		t.Fatalf("reader acquired prematurely while writer active: %v", res.err)
	default:
	}

	// Release writer. Reader should now succeed.
	w.Release()

	select {
	case res := <-readerCh:
		if res.err != nil {
			t.Fatalf("reader acquire failed: %v", res.err)
		}
		defer res.lease.Release()
		if !res.lease.Shared() || res.lease.Mode() != ModeShared {
			t.Fatal("reader lease is not shared")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reader timed out after writer released")
	}
}

// TestReleaseSharedOnProcessExit proves that the OS releases a shared lease
// when the holding process exits without calling Release().
func TestReleaseSharedOnProcessExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	if p := os.Getenv("GPULEASE_HELPER_SHARED_PATH"); p != "" {
		if _, err := AcquireShared(Options{Path: p, NoWait: true}); err != nil {
			os.Stderr.WriteString("child acquire shared failed: " + err.Error() + "\n")
			os.Exit(3)
		}
		os.Stdout.WriteString("ACQUIRED_SHARED\n")
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestReleaseSharedOnProcessExit")
	cmd.Env = append(os.Environ(), "GPULEASE_HELPER_SHARED_PATH="+path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "ACQUIRED_SHARED") {
		t.Fatalf("child did not acquire shared lease; output:\n%s", out)
	}

	// Child has exited; exclusive writer must now be able to acquire.
	l, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("exclusive lease not acquired after child process exit: %v", err)
	}
	l.Release()
}

// TestCrossProcessSharedExclusiveAdmission exercises the OS lock across
// independent processes. Shared inspection is intentionally host-local: every
// participant must execute on the hardware host and name the same lockfile.
func TestCrossProcessSharedExclusiveAdmission(t *testing.T) {
	if mode := os.Getenv(crossProcessLeaseModeEnv); mode != "" {
		path := os.Getenv("GPULEASE_CROSS_PROCESS_PATH")
		var (
			lease *Lease
			err   error
		)
		switch mode {
		case ModeShared.String():
			lease, err = AcquireShared(Options{Path: path, NoWait: true})
		case ModeExclusive.String():
			lease, err = Acquire(Options{Path: path, NoWait: true})
		default:
			err = fmt.Errorf("unknown helper mode %q", mode)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "helper acquire: %v\n", err)
			os.Exit(3)
		}
		fmt.Fprintln(os.Stdout, "READY")
		var control [1]byte
		n, _ := os.Stdin.Read(control[:])
		if n == 1 && control[0] == 'K' {
			os.Exit(9) // leave the lease to process teardown, without Release
		}
		lease.Release()
		os.Exit(0)
	}

	path := filepath.Join(t.TempDir(), "cross-process-gpu.lease")
	r1 := startLeaseProcess(t, path, ModeShared)
	r2 := startLeaseProcess(t, path, ModeShared)

	_, err := Acquire(Options{Path: path, NoWait: true})
	var busy *BusyError
	if !errors.As(err, &busy) || busy.PID != 0 {
		t.Fatalf("exclusive behind shared children: got %v, want BusyError with PID 0", err)
	}

	r1.stop(t)
	_, err = Acquire(Options{Path: path, NoWait: true})
	busy = nil
	if !errors.As(err, &busy) || busy.PID != 0 {
		t.Fatalf("exclusive behind final shared child: got %v, want BusyError with PID 0", err)
	}

	r2.stop(t)
	w, err := Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("exclusive after final reader release: %v", err)
	}
	w.Release()

	writer := startLeaseProcess(t, path, ModeExclusive)
	_, err = AcquireShared(Options{Path: path, NoWait: true})
	busy = nil
	if !errors.As(err, &busy) || busy.PID != writer.cmd.Process.Pid {
		t.Fatalf("shared behind exclusive child: got %v, want BusyError with PID %d", err, writer.cmd.Process.Pid)
	}
	writer.stop(t)

	r, err := AcquireShared(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("shared after writer release: %v", err)
	}
	r.Release()

	doomed := startLeaseProcess(t, path, ModeShared)
	doomed.kill(t)
	w, err = Acquire(Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("exclusive after shared holder process death: %v", err)
	}
	w.Release()
}
