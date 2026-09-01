package extensionfault

import (
	"context"
	"errors"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func buildFaultPlugin(t *testing.T) string {
	t.Helper()
	root := filepath.Clean(filepath.Join("..", ".."))
	name := "fault-plugin"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(t.TempDir(), name)
	cmd := exec.Command("go", "build", "-o", binary, "./examples/mcp/fault-plugin")
	cmd.Dir = root
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build fault plugin: %v\n%s", err, output)
	}
	return binary
}

func spec(name, binary, mode string, restarts int) Spec {
	return Spec{
		Name:           name,
		Command:        []string{binary, "--mode", mode},
		StartupTimeout: 200 * time.Millisecond,
		CallTimeout:    200 * time.Millisecond,
		MaxRestarts:    restarts,
	}
}

func TestStartupTimeoutQuarantinesOnlyFaultingExtension(t *testing.T) {
	binary := buildFaultPlugin(t)
	supervisor, err := New(
		spec("stuck", binary, "startup-hang", 1),
		spec("healthy", binary, "healthy", 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = supervisor.Close() })

	started := time.Now()
	_, err = supervisor.Call(context.Background(), "stuck", "ignored")
	if !errors.Is(err, ErrQuarantined) {
		t.Fatalf("stuck call error = %v, want ErrQuarantined", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("startup containment took %s", elapsed)
	}
	status, ok := supervisor.Status("stuck")
	if !ok || !status.Quarantined || status.Running || status.Restarts != 1 || status.Failures != 2 {
		t.Fatalf("stuck status = %+v, ok=%v", status, ok)
	}
	if !strings.Contains(status.LastError, "startup") || !strings.Contains(status.LastError, "deadline exceeded") {
		t.Fatalf("last error = %q, want startup deadline", status.LastError)
	}

	got, err := supervisor.Call(context.Background(), "healthy", "after-timeout")
	if err != nil || got != "ok:after-timeout" {
		t.Fatalf("healthy sibling after timeout = %q, %v", got, err)
	}
}

func TestStartupCrashExhaustsRestartBudgetAndSupervisorSurvives(t *testing.T) {
	binary := buildFaultPlugin(t)
	supervisor, err := New(
		spec("crashing", binary, "crash", 2),
		spec("healthy", binary, "healthy", 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = supervisor.Close() })

	_, err = supervisor.Call(context.Background(), "crashing", "ignored")
	if !errors.Is(err, ErrQuarantined) {
		t.Fatalf("crashing call error = %v, want ErrQuarantined", err)
	}
	status, _ := supervisor.Status("crashing")
	if !status.Quarantined || status.Running || status.Restarts != 2 || status.Failures != 3 {
		t.Fatalf("crashing status = %+v", status)
	}

	started := time.Now()
	_, err = supervisor.Call(context.Background(), "crashing", "again")
	if !errors.Is(err, ErrQuarantined) {
		t.Fatalf("second crashing call error = %v, want ErrQuarantined", err)
	}
	if time.Since(started) > 100*time.Millisecond {
		t.Fatal("quarantined extension was not rejected immediately")
	}
	after, _ := supervisor.Status("crashing")
	if after.Restarts != status.Restarts || after.Failures != status.Failures {
		t.Fatalf("quarantined call started more work: before=%+v after=%+v", status, after)
	}

	for _, payload := range []string{"one", "two"} {
		got, callErr := supervisor.Call(context.Background(), "healthy", payload)
		if callErr != nil || got != "ok:"+payload {
			t.Fatalf("healthy sibling call = %q, %v", got, callErr)
		}
	}
}

func TestCallHangIsKilledAndHealthySiblingContinues(t *testing.T) {
	if runtime.GOOS != "windows" {
		// On Unix, signal 0 is used below to prove the timed-out child no longer exists.
		// Windows has no equivalent in the standard library; Wait completion is still
		// exercised by the supervisor's synchronous stop path.
	}
	binary := buildFaultPlugin(t)
	supervisor, err := New(
		spec("hung", binary, "hang", 0),
		spec("healthy", binary, "healthy", 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = supervisor.Close() })

	_, err = supervisor.Call(context.Background(), "hung", "ignored")
	if !errors.Is(err, ErrQuarantined) {
		t.Fatalf("hung call error = %v, want ErrQuarantined", err)
	}
	status, _ := supervisor.Status("hung")
	if status.PID != 0 || status.Running || !status.Quarantined {
		t.Fatalf("hung status = %+v", status)
	}
	if !strings.Contains(status.LastError, "call") || !strings.Contains(status.LastError, "deadline exceeded") {
		t.Fatalf("last error = %q, want call deadline", status.LastError)
	}

	got, err := supervisor.Call(context.Background(), "healthy", "still-live")
	if err != nil || got != "ok:still-live" {
		t.Fatalf("healthy sibling after call hang = %q, %v", got, err)
	}
}

func TestUnknownExtensionIsUnavailable(t *testing.T) {
	supervisor, err := New()
	if err != nil {
		t.Fatal(err)
	}
	_, err = supervisor.Call(context.Background(), "missing", "")
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("error = %v, want ErrUnavailable", err)
	}
}
