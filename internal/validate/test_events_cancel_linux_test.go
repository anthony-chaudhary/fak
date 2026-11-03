//go:build linux

package validate

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=500ms lane=default
func TestValidateEventCancellationContainsOwnedDescendant(t *testing.T) {
	const roleEnv = "FAK_VALIDATE_OWNED_CANCEL_ROLE"
	const readyEnv = "FAK_VALIDATE_OWNED_CANCEL_READY"
	switch os.Getenv(roleEnv) {
	case "leaf":
		if err := os.WriteFile(os.Getenv(readyEnv), []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			os.Exit(3)
		}
		time.Sleep(5 * time.Second) // Bounded even if the pre-fix cancellation leaks us.
		os.Exit(0)
	case "child":
		child := exec.Command(os.Args[0], "-test.run=^TestValidateEventCancellationContainsOwnedDescendant$")
		child.Env = append(os.Environ(), roleEnv+"=leaf")
		if err := child.Start(); err != nil {
			os.Exit(4)
		}
		_ = child.Wait()
		os.Exit(0)
	}
	ready := filepath.Join(t.TempDir(), "owned-leaf.pid")
	t.Setenv(roleEnv, "child")
	t.Setenv(readyEnv, ready)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{os.Args[0], "-test.run=^TestValidateEventCancellationContainsOwnedDescendant$"}}
	done := make(chan bool, 1)
	go func() { done <- executeValidateEventTests(ctx, witness, []string{"fixture"}) }()
	var pid int
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(ready); err == nil {
			pid, _ = strconv.Atoi(string(b))
			if pid > 0 {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if pid <= 0 {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
		}
		t.Fatal("owned leaf never reported readiness")
	}
	alive := func() bool {
		b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
		if os.IsNotExist(err) {
			return false
		}
		if err != nil {
			t.Fatalf("inspect owned leaf: %v", err)
		}
		end := strings.LastIndexByte(string(b), ')')
		if end < 0 || len(b) <= end+2 {
			t.Fatal("malformed owned leaf process state")
		}
		return b[end+2] != 'Z' && b[end+2] != 'X'
	}
	// No PID-based cleanup: even a failed pre-fix run has a self-expiring leaf.
	// Wait for its bounded lifetime rather than risking a recycled PID kill.
	t.Cleanup(func() {
		deadline := time.Now().Add(6 * time.Second)
		for alive() && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if alive() {
			t.Error("bounded fixture descendant did not terminate")
		}
	})
	if !alive() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
		}
		t.Fatal("leaf not alive before cancellation")
	}
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Fatal("cancelled execution reported success")
		}
	case <-time.After(time.Second):
		t.Fatal("validator did not return after cancellation")
	}
	deadline = time.Now().Add(time.Second)
	for alive() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if alive() {
		t.Fatal("owned descendant survived validator cancellation")
	}
	if witness.Complete || !witness.TimedOut || witness.DescendantTeardown != "unproven" || witness.DrainStatus != "unknown" {
		t.Fatalf("cancellation evidence overstated: %+v", witness)
	}
}
