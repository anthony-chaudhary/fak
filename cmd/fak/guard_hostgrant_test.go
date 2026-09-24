package main

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostgrant"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const guardHostGrantChildEnv = "GO_WANT_GUARD_HOSTGRANT_CHILD"

// TestGuardHostGrantChildProcess is re-executed as a real child. It announces
// that Start completed, then stays alive until its stdin is closed.
func TestGuardHostGrantChildProcess(t *testing.T) {
	if os.Getenv(guardHostGrantChildEnv) != "1" {
		return
	}
	marker := os.Getenv("GUARD_HOSTGRANT_CHILD_MARKER")
	if marker == "" {
		t.Fatal("GUARD_HOSTGRANT_CHILD_MARKER is required")
	}
	if err := os.WriteFile(marker, []byte("started\n"), 0o600); err != nil {
		t.Fatalf("write started marker: %v", err)
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
}

func TestGuardHostGrantSerializesRealChildren(t *testing.T) {
	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", filepath.Join(t.TempDir(), "host-grants.json"))
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "1")

	first := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "first.started"))
	firstJob, firstRelease, err := startGuardChildWithHostGrant(context.Background(), first.cmd, windowgate.ManagedJobConfig{})
	if err != nil {
		t.Fatalf("start first guarded child: %v", err)
	}
	waitGuardHostGrantMarker(t, first.marker)

	second := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "second.started"))
	secondJob, secondRelease, err := startGuardChildWithHostGrant(context.Background(), second.cmd, windowgate.ManagedJobConfig{})
	if !errors.Is(err, hostgrant.ErrFull) {
		t.Fatalf("start second guarded child error = %v, want hostgrant.ErrFull", err)
	}
	if secondJob != nil || secondRelease != nil || second.cmd.Process != nil {
		t.Fatalf("refused child launched or returned lifecycle: job=%v release=%v process=%v", secondJob, secondRelease != nil, second.cmd.Process)
	}
	if _, statErr := os.Stat(second.marker); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("refused child marker stat error = %v, want not-exist", statErr)
	}
	_ = second.stdin.Close()

	finishGuardHostGrantChild(t, first, firstJob, firstRelease)

	third := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "third.started"))
	thirdJob, thirdRelease, err := startGuardChildWithHostGrant(context.Background(), third.cmd, windowgate.ManagedJobConfig{})
	if err != nil {
		t.Fatalf("start third guarded child after release: %v", err)
	}
	waitGuardHostGrantMarker(t, third.marker)
	finishGuardHostGrantChild(t, third, thirdJob, thirdRelease)
}

func TestGuardHostGrantStartFailureReleases(t *testing.T) {
	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", filepath.Join(t.TempDir(), "host-grants.json"))
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "1")

	missing := filepath.Join(t.TempDir(), "missing-guard-child")
	if _, _, err := startGuardChildWithHostGrant(context.Background(), exec.Command(missing), windowgate.ManagedJobConfig{}); err == nil {
		t.Fatal("start missing child succeeded")
	}

	child := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "after-failure.started"))
	job, release, err := startGuardChildWithHostGrant(context.Background(), child.cmd, windowgate.ManagedJobConfig{})
	if err != nil {
		t.Fatalf("start after failed registration/start: %v", err)
	}
	waitGuardHostGrantMarker(t, child.marker)
	finishGuardHostGrantChild(t, child, job, release)
}

func TestGuardHostGrantDuplicateRequestIsFenced(t *testing.T) {
	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", filepath.Join(t.TempDir(), "host-grants.json"))
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "2")
	t.Setenv("FAK_GUARD_HOSTGRANT_REQUEST_ID", "fixed-guard-attempt")

	first := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "first.started"))
	firstJob, firstRelease, err := startGuardChildWithHostGrant(context.Background(), first.cmd, windowgate.ManagedJobConfig{})
	if err != nil {
		t.Fatalf("start first fixed-ID child: %v", err)
	}
	waitGuardHostGrantMarker(t, first.marker)

	second := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "second.started"))
	secondJob, secondRelease, err := startGuardChildWithHostGrant(context.Background(), second.cmd, windowgate.ManagedJobConfig{})
	if !errors.Is(err, hostgrant.ErrFenced) {
		t.Fatalf("duplicate fixed-ID start error = %v, want hostgrant.ErrFenced", err)
	}
	if secondJob != nil || secondRelease != nil || second.cmd.Process != nil {
		t.Fatalf("fenced child launched or returned lifecycle: job=%v release=%v process=%v", secondJob, secondRelease != nil, second.cmd.Process)
	}
	if _, statErr := os.Stat(second.marker); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("fenced child marker stat error = %v, want not-exist", statErr)
	}
	_ = second.stdin.Close()

	finishGuardHostGrantChild(t, first, firstJob, firstRelease)
}

func TestGuardHostGrantRequestIDWithoutStoreFailsClosed(t *testing.T) {
	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", "")
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "")
	t.Setenv("FAK_GUARD_HOSTGRANT_REQUEST_ID", "orphan-request")

	child := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "must-not-start.started"))
	job, release, err := startGuardChildWithHostGrant(context.Background(), child.cmd, windowgate.ManagedJobConfig{})
	if err == nil {
		t.Fatal("request ID without host-grant store unexpectedly admitted")
	}
	if job != nil || release != nil || child.cmd.Process != nil {
		t.Fatalf("invalid config launched or returned lifecycle: job=%v release=%v process=%v", job, release != nil, child.cmd.Process)
	}
	if _, statErr := os.Stat(child.marker); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("invalid-config child marker stat error = %v, want not-exist", statErr)
	}
	_ = child.stdin.Close()
}

func TestGuardHostGrantReserveMismatchDoesNotStartChild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host-grants.json")
	seed := hostgrant.Store{
		Path:           path,
		Capacity:       hostgrant.Vector{Processes: 2},
		ControlReserve: hostgrant.Vector{Processes: 1},
	}
	grant, err := seed.TryAcquire(context.Background(), hostgrant.Request{
		ID:    "ordinary-first",
		Owner: hostgrant.Owner{ID: "seed-owner", PID: os.Getpid(), StartedAt: time.Now().UTC()},
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   time.Minute,
	})
	if err != nil {
		t.Fatalf("seed reserved store: %v", err)
	}
	t.Cleanup(func() { _ = seed.Release(context.Background(), grant) })

	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", path)
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "2")
	t.Setenv("FAK_GUARD_HOSTGRANT_REQUEST_ID", "ordinary-second")
	child := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "must-not-start.started"))
	job, release, err := startGuardChildWithHostGrant(context.Background(), child.cmd, windowgate.ManagedJobConfig{})
	if !errors.Is(err, hostgrant.ErrCapacityMismatch) {
		t.Fatalf("start with missing reserve configuration error = %v, want ErrCapacityMismatch", err)
	}
	if job != nil || release != nil || child.cmd.Process != nil {
		t.Fatalf("reserve-mismatch child launched or returned lifecycle: job=%v release=%v process=%v", job, release != nil, child.cmd.Process)
	}
	if _, statErr := os.Stat(child.marker); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("reserve-mismatch child marker stat error = %v, want not-exist", statErr)
	}
	_ = child.stdin.Close()
}

func TestGuardHostGrantUnconfiguredIsNoop(t *testing.T) {
	t.Setenv("FAK_GUARD_HOSTGRANT_PATH", "")
	t.Setenv("FAK_GUARD_HOSTGRANT_CAPACITY", "")
	t.Setenv("FAK_GUARD_HOSTGRANT_REQUEST_ID", "")

	child := newGuardHostGrantChild(t, filepath.Join(t.TempDir(), "unconfigured.started"))
	job, release, err := startGuardChildWithHostGrant(context.Background(), child.cmd, windowgate.ManagedJobConfig{})
	if err != nil {
		t.Fatalf("unconfigured guarded start: %v", err)
	}
	waitGuardHostGrantMarker(t, child.marker)
	finishGuardHostGrantChild(t, child, job, release)
}

type guardHostGrantTestChild struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	marker string
}

func newGuardHostGrantChild(t *testing.T, marker string) guardHostGrantTestChild {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	cmd := exec.Command(executable, "-test.run=^TestGuardHostGrantChildProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), guardHostGrantChildEnv+"=1", "GUARD_HOSTGRANT_CHILD_MARKER="+marker)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("create child stdin pipe: %v", err)
	}
	return guardHostGrantTestChild{cmd: cmd, stdin: stdin, marker: marker}
}

func waitGuardHostGrantMarker(t *testing.T, marker string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(marker); err == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("child did not create started marker %q", marker)
}

func finishGuardHostGrantChild(t *testing.T, child guardHostGrantTestChild, job *windowgate.JobObject, release func() error) {
	t.Helper()
	if err := child.stdin.Close(); err != nil {
		t.Errorf("close child stdin: %v", err)
	}
	if err := child.cmd.Wait(); err != nil {
		t.Errorf("wait child: %v", err)
	}
	if err := job.Close(); err != nil {
		t.Errorf("close child job: %v", err)
	}
	if err := release(); err != nil {
		t.Errorf("release host grant: %v", err)
	}
}
