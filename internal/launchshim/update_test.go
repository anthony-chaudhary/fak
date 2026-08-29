package launchshim

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func holdUpdate(t *testing.T) (target, prior string, release func()) {
	t.Helper()
	dir := t.TempDir()
	target = filepath.Join(dir, "fak")
	prior = filepath.Join(dir, "fak-prior")
	if err := os.WriteFile(target, []byte("new"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prior, []byte("prior"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteUpdateState(target, prior); err != nil {
		t.Fatal(err)
	}
	return target, prior, func() { _ = os.Remove(UpdateStatePath(target)) }
}

func TestUpdatePolicyPrecedenceAndBound(t *testing.T) {
	t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "")
	t.Setenv("FAK_UPDATE_LAUNCH_WAIT", "")
	config := Config{UpdateLaunchPolicy: UpdatePolicyWait, UpdateLaunchWaitMS: 2500}

	policy, wait, err := UpdatePolicy(config, "", "")
	if err != nil || policy != UpdatePolicyWait || wait != 2500*time.Millisecond {
		t.Fatalf("config policy=%q wait=%s err=%v", policy, wait, err)
	}
	t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "fail")
	t.Setenv("FAK_UPDATE_LAUNCH_WAIT", "3s")
	policy, wait, err = UpdatePolicy(config, "", "")
	if err != nil || policy != UpdatePolicyWait || wait != 2500*time.Millisecond {
		t.Fatalf("legacy environment overrode config: policy=%q wait=%s err=%v", policy, wait, err)
	}
	policy, wait, err = UpdatePolicy(config, "prior", "1h")
	if err != nil || policy != UpdatePolicyPrior || wait != maxUpdateWait {
		t.Fatalf("flag policy=%q wait=%s err=%v", policy, wait, err)
	}
	if _, _, err := UpdatePolicy(config, "prompt", ""); err == nil {
		t.Fatal("interactive policy unexpectedly accepted")
	}
	if _, _, err := UpdatePolicy(config, "wait", "0"); err == nil {
		t.Fatal("non-positive wait unexpectedly accepted")
	}
}

func TestUpdatePolicyLoadsFromLaunchConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", path)
	t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "fail")
	t.Setenv("FAK_UPDATE_LAUNCH_WAIT", "3s")
	if err := Save(Config{
		UpdateLaunchPolicy: UpdatePolicyWait,
		UpdateLaunchWaitMS: 1250,
		Providers:          map[string]Provider{},
	}); err != nil {
		t.Fatal(err)
	}
	config, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	policy, wait, err := UpdatePolicy(config, "", "")
	if err != nil || policy != UpdatePolicyWait || wait != 1250*time.Millisecond {
		t.Fatalf("policy=%q wait=%s err=%v config=%+v", policy, wait, err, config)
	}
}

func TestResolveExecutablePoliciesAtHeldTransactionBoundary(t *testing.T) {
	target, prior, release := holdUpdate(t)
	defer release()
	if got, err := ResolveExecutable(target, UpdatePolicyPrior, time.Second); err != nil || got != prior {
		t.Fatalf("prior=%q err=%v", got, err)
	}
	if _, err := ResolveExecutable(target, UpdatePolicyFail, time.Second); err == nil || !strings.Contains(err.Error(), "retry") {
		t.Fatalf("fail err=%v", err)
	}

	oldNow, oldSleep := updateNow, updateSleep
	sleepEntered := make(chan struct{})
	releaseSleep := make(chan struct{})
	updateNow = func() time.Time { return time.Unix(100, 0) }
	updateSleep = func(time.Duration) {
		close(sleepEntered)
		<-releaseSleep
	}
	t.Cleanup(func() { updateNow, updateSleep = oldNow, oldSleep })

	type result struct {
		path string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		got, err := ResolveExecutable(target, UpdatePolicyWait, time.Second)
		done <- result{got, err}
	}()
	<-sleepEntered
	select {
	case got := <-done:
		t.Fatalf("wait returned before transaction completed: %+v", got)
	default:
	}
	release()
	close(releaseSleep)
	got := <-done
	if got.err != nil || got.path != target {
		t.Fatalf("wait selected=%q err=%v, want %q", got.path, got.err, target)
	}
}

func TestResolveExecutableWaitIsDeterministicallyBounded(t *testing.T) {
	target, _, release := holdUpdate(t)
	defer release()
	oldNow, oldSleep := updateNow, updateSleep
	start := time.Unix(100, 0)
	calls := 0
	updateNow = func() time.Time {
		calls++
		if calls == 1 {
			return start
		}
		return start.Add(25 * time.Millisecond)
	}
	updateSleep = func(time.Duration) {}
	t.Cleanup(func() { updateNow, updateSleep = oldNow, oldSleep })

	if _, err := ResolveExecutable(target, UpdatePolicyWait, 20*time.Millisecond); err == nil || !strings.Contains(err.Error(), "20ms") {
		t.Fatalf("bounded wait error=%v", err)
	}
}

func TestStableExecutableBindingRoundTrip(t *testing.T) {
	dir := t.TempDir()
	stable := filepath.Join(dir, "fak-launch")
	target := filepath.Join(dir, "fak")
	if err := os.WriteFile(stable, []byte("stable"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("target"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := BindStableExecutable(stable, target); err != nil {
		t.Fatal(err)
	}
	got, err := StableExecutable(stable)
	if err != nil || got != target {
		t.Fatalf("binding=%q err=%v want %q", got, err, target)
	}
	b, err := os.ReadFile(stableBindingPath(stable))
	if err != nil {
		t.Fatal(err)
	}
	var binding map[string]string
	if err := json.Unmarshal(b, &binding); err != nil {
		t.Fatal(err)
	}
	if binding["schema"] != stableBindingSchema {
		t.Fatalf("binding=%v", binding)
	}
}

func TestResolveExecutableRejectsMismatchedState(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak")
	prior := filepath.Join(dir, "prior")
	if err := os.WriteFile(target, []byte("target"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prior, []byte("prior"), 0o755); err != nil {
		t.Fatal(err)
	}
	state := UpdateState{Schema: UpdateStateSchema, Target: filepath.Join(dir, "other"), Prior: prior}
	b, _ := json.Marshal(state)
	if err := os.WriteFile(UpdateStatePath(target), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveExecutable(target, UpdatePolicyPrior, time.Second); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatched state error=%v", err)
	}
}
