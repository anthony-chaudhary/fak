package launchshim

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func holdUpdate(t *testing.T) (string, string, func()) {
	t.Helper()
	d := t.TempDir()
	target, prior := filepath.Join(d, "fak"), filepath.Join(d, "fak.prior")
	if err := os.WriteFile(target, []byte("new"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prior, []byte("prior"), 0o755); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(updateState{Target: target, Prior: prior})
	if err := os.WriteFile(UpdateStatePath(target), b, 0o600); err != nil {
		t.Fatal(err)
	}
	return target, prior, func() { _ = os.Remove(UpdateStatePath(target)) }
}

func TestResolveExecutablePoliciesAtHeldTransactionBoundary(t *testing.T) {
	target, prior, release := holdUpdate(t)
	if got, err := ResolveExecutable(target, UpdatePolicyPrior, time.Second); err != nil || got != prior {
		t.Fatalf("prior=%q err=%v", got, err)
	}
	if _, err := ResolveExecutable(target, UpdatePolicyFail, time.Second); err == nil || !strings.Contains(err.Error(), "retry") {
		t.Fatalf("fail err=%v", err)
	}
	done := make(chan string, 1)
	go func() { got, _ := ResolveExecutable(target, UpdatePolicyWait, time.Second); done <- got }()
	select {
	case <-done:
		t.Fatal("wait returned before transaction completed")
	case <-time.After(30 * time.Millisecond):
	}
	release()
	if got := <-done; got != target {
		t.Fatalf("wait selected %q want %q", got, target)
	}
}

func TestResolveExecutableWaitIsBounded(t *testing.T) {
	target, _, _ := holdUpdate(t)
	started := time.Now()
	if _, err := ResolveExecutable(target, UpdatePolicyWait, 25*time.Millisecond); err == nil {
		t.Fatal("wait unexpectedly succeeded")
	}
	if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
		t.Fatalf("unbounded wait: %s", elapsed)
	}
}
