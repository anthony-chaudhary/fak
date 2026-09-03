package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchguard"
)

func init() {
	guardCrashRSIAdmit = func(string) (func(bool) error, launchguard.Decision, error) {
		return func(bool) error { return nil }, launchguard.Decision{Outcome: launchguard.Admitted}, nil
	}
}

func stubGuardCrashRSIAdmission(t *testing.T) {
	t.Helper()
	old := guardCrashRSIAdmit
	guardCrashRSIAdmit = func(string) (func(bool) error, launchguard.Decision, error) {
		return func(bool) error { return nil }, launchguard.Decision{Outcome: launchguard.Admitted}, nil
	}
	t.Cleanup(func() { guardCrashRSIAdmit = old })
}

func useRealGuardCrashRSIAdmission(t *testing.T, dir string) {
	t.Helper()
	oldAdmit, oldDir := guardCrashRSIAdmit, guardCrashRSIDir
	guardCrashRSIDir = dir
	guardCrashRSIAdmit = admitGuardCrashRSILaunch
	t.Cleanup(func() {
		guardCrashRSIAdmit = oldAdmit
		guardCrashRSIDir = oldDir
	})
}

func TestGuardCrashRSIFirstEligibleCrashLaunchesBoundedTaggedInvestigation(t *testing.T) {
	stubGuardCrashRSIAdmission(t)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	var got []guardCrashRSIRequest
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		got = append(got, req)
		return nil
	}
	var stderr bytes.Buffer
	if !guardMaybeLaunchCrashRSI(&stderr, new(guardRSISession), "RID-secret-looking-source", "codex", "NONZERO_EXIT", 17, 0) {
		t.Fatal("first eligible crash did not launch")
	}
	if len(got) != 1 {
		t.Fatalf("launch count=%d, want 1", len(got))
	}
	req := got[0]
	if !strings.HasPrefix(req.Tag, "guard-crash-rsi/") || req.Source == "" || !strings.HasSuffix(req.Tag, req.Source) {
		t.Fatalf("tag/source = %q/%q", req.Tag, req.Source)
	}
	for _, want := range []string{"ORIGINAL fak guard child crash", req.Tag, req.Source, "NONZERO_EXIT", "exit_code: 17", req.Workspace} {
		if !strings.Contains(req.Prompt, want) {
			t.Fatalf("prompt missing %q: %s", want, req.Prompt)
		}
	}
	if strings.Contains(req.Prompt, "RID-secret-looking-source") {
		t.Fatalf("prompt leaked raw guard identity: %s", req.Prompt)
	}
	if !strings.Contains(stderr.String(), req.Tag) {
		t.Fatalf("status missing tag: %s", stderr.String())
	}
}

func TestGuardCrashRSIOnlyFirstCrashAndNeverRecurses(t *testing.T) {
	stubGuardCrashRSIAdmission(t)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }
	session := new(guardRSISession)
	guardMaybeLaunchCrashRSI(nil, session, "trace", "claude", "SIGNAL", -1, 0)
	guardMaybeLaunchCrashRSI(nil, session, "trace", "claude", "SIGNAL", -1, 1)
	if launches != 1 {
		t.Fatalf("launches=%d, want exactly 1", launches)
	}
	t.Setenv(guardCrashRSIMarkerEnv, "guard-crash-rsi/already")
	guardMaybeLaunchCrashRSI(nil, new(guardRSISession), "other", "claude", "OOM", 137, 0)
	if launches != 1 {
		t.Fatalf("recursive session launched: %d", launches)
	}
}

func TestGuardCrashRSIUnsafeAndNonCrashCasesSkip(t *testing.T) {
	stubGuardCrashRSIAdmission(t)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }
	cases := []struct {
		trace, agent, class string
		code                int
	}{
		{"", "codex", "NONZERO_EXIT", 1},
		{"trace", "unknown", "NONZERO_EXIT", 1},
		{"trace", "codex", "", 1},
		{"trace", "codex", "CLEAN_EXIT", 0},
	}
	for _, tc := range cases {
		guardMaybeLaunchCrashRSI(nil, new(guardRSISession), tc.trace, tc.agent, tc.class, tc.code, 0)
	}
	if launches != 0 {
		t.Fatalf("unsafe/non-crash launches=%d", launches)
	}
}

func TestGuardCrashRSILaunchFailureIsFailOpen(t *testing.T) {
	stubGuardCrashRSIAdmission(t)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { return errors.New("synthetic launch failure") }
	var stderr bytes.Buffer
	if guardMaybeLaunchCrashRSI(&stderr, new(guardRSISession), "trace", "codex", "NONZERO_EXIT", 9, 0) {
		t.Fatal("failed launch reported success")
	}
	if !strings.Contains(stderr.String(), "synthetic launch failure") {
		t.Fatalf("missing fail-open diagnostic: %s", stderr.String())
	}
}

func TestGuardCrashRSILaunchguardRefusalSkipsLaunch(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldAdmit, oldLaunch := guardCrashRSIAdmit, guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSIAdmit, guardCrashRSILaunch = oldAdmit, oldLaunch })
	guardCrashRSIAdmit = func(string) (func(bool) error, launchguard.Decision, error) {
		return nil, launchguard.Decision{Outcome: launchguard.Quarantined}, nil
	}
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }
	var stderr bytes.Buffer
	if guardMaybeLaunchCrashRSI(&stderr, new(guardRSISession), "trace", "codex", "NONZERO_EXIT", 9, 0) {
		t.Fatal("quarantined launch reported success")
	}
	if launches != 0 || !strings.Contains(stderr.String(), "quarantined") {
		t.Fatalf("launches=%d stderr=%q", launches, stderr.String())
	}
}

func TestGuardCrashRSILaunchFailureRecordsFailure(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldAdmit, oldLaunch := guardCrashRSIAdmit, guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSIAdmit, guardCrashRSILaunch = oldAdmit, oldLaunch })
	var finished []bool
	guardCrashRSIAdmit = func(string) (func(bool) error, launchguard.Decision, error) {
		return func(success bool) error { finished = append(finished, success); return nil }, launchguard.Decision{Outcome: launchguard.Admitted}, nil
	}
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { return errors.New("boom") }
	guardMaybeLaunchCrashRSI(nil, new(guardRSISession), "trace", "codex", "NONZERO_EXIT", 9, 0)
	if len(finished) != 1 || finished[0] {
		t.Fatalf("finished=%v, want [false]", finished)
	}
}

func TestGuardCrashRSIEnvironmentExcludesSecretsAndCarriesOnlyMarker(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "do-not-forward")
	t.Setenv("ANTHROPIC_API_KEY", "do-not-forward")
	t.Setenv("FAK_SECRET_ORIGINAL_ARG", "--token=do-not-forward")
	env := guardCrashRSIEnvironment("guard-crash-rsi/source")
	joined := strings.Join(env, "\n")
	for _, forbidden := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FAK_SECRET_ORIGINAL_ARG", "do-not-forward"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("environment leaked %q: %s", forbidden, joined)
		}
	}
	if !strings.Contains(joined, guardCrashRSIMarkerEnv+"=guard-crash-rsi/source") {
		t.Fatalf("environment missing recursion marker: %v", env)
	}
	_ = os.Getenv("PATH") // documents that bootstrap paths may be retained without asserting host shape.
}

func TestGuardCrashRSIHostWideConcurrentClaimAdmitsExactlyOne(t *testing.T) {
	dir := t.TempDir()
	useRealGuardCrashRSIAdmission(t, dir)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldLaunch := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = oldLaunch })

	var launchMu sync.Mutex
	var launches int
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		launchMu.Lock()
		launches++
		launchMu.Unlock()
		time.Sleep(10 * time.Millisecond)
		return nil
	}

	const workers = 20
	start := make(chan struct{})
	type outcome struct {
		ok     bool
		stderr string
	}
	results := make(chan outcome, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			var stderr bytes.Buffer
			session := new(guardRSISession)
			ok := guardMaybeLaunchCrashRSI(&stderr, session, "RID-concurrent-trace", "codex", "NONZERO_EXIT", 1, 0)
			results <- outcome{ok: ok, stderr: stderr.String()}
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	var successes int
	var refused int
	for r := range results {
		if r.ok {
			successes++
			if !strings.Contains(r.stderr, "spawned crash RSI session") {
				t.Errorf("success stderr missing spawned message: %q", r.stderr)
			}
		} else {
			refused++
			if !strings.Contains(r.stderr, "duplicate-active") {
				t.Errorf("refused stderr missing duplicate-active: %q", r.stderr)
			}
			if strings.Contains(r.stderr, "RID-concurrent-trace") {
				t.Errorf("refused stderr leaked raw trace ID: %q", r.stderr)
			}
		}
	}
	if successes != 1 || refused != workers-1 {
		t.Fatalf("successes=%d refused=%d, want 1 success and %d refused", successes, refused, workers-1)
	}
	launchMu.Lock()
	totalLaunches := launches
	launchMu.Unlock()
	if totalLaunches != 1 {
		t.Fatalf("total launches=%d, want 1", totalLaunches)
	}
}

func TestGuardCrashRSICooldownRefusesSubsequentLaunchesAcrossProcesses(t *testing.T) {
	dir := t.TempDir()
	useRealGuardCrashRSIAdmission(t, dir)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldLaunch := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = oldLaunch })

	launches := 0
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		launches++
		return nil
	}

	var stderr1 bytes.Buffer
	ok1 := guardMaybeLaunchCrashRSI(&stderr1, new(guardRSISession), "stable-trace-alpha", "claude", "SIGNAL", 9, 0)
	if !ok1 {
		t.Fatalf("first launch failed: %s", stderr1.String())
	}
	if launches != 1 {
		t.Fatalf("launches=%d, want 1", launches)
	}

	// An independent guard process observing the same crash within cooldown is refused.
	var stderr2 bytes.Buffer
	ok2 := guardMaybeLaunchCrashRSI(&stderr2, new(guardRSISession), "stable-trace-alpha", "claude", "SIGNAL", 9, 0)
	if ok2 {
		t.Fatal("second launch within cooldown should have been refused")
	}
	if launches != 1 {
		t.Fatalf("launches=%d, want still 1", launches)
	}
	if !strings.Contains(stderr2.String(), "crash RSI launch refused") || !strings.Contains(stderr2.String(), "duplicate-active") {
		t.Fatalf("stderr2 missing refusal: %s", stderr2.String())
	}
	if !strings.Contains(stderr2.String(), "retry-after=") {
		t.Fatalf("stderr2 missing retry-after: %s", stderr2.String())
	}
	if strings.Contains(stderr2.String(), "stable-trace-alpha") {
		t.Fatalf("stderr2 leaked raw trace ID: %s", stderr2.String())
	}
}

func TestGuardCrashRSIDifferentTagsLaunchIndependently(t *testing.T) {
	dir := t.TempDir()
	useRealGuardCrashRSIAdmission(t, dir)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldLaunch := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = oldLaunch })

	var launchedTags []string
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		launchedTags = append(launchedTags, req.Tag)
		return nil
	}

	ok1 := guardMaybeLaunchCrashRSI(nil, new(guardRSISession), "trace-one", "codex", "NONZERO_EXIT", 1, 0)
	ok2 := guardMaybeLaunchCrashRSI(nil, new(guardRSISession), "trace-two", "codex", "NONZERO_EXIT", 1, 0)
	if !ok1 || !ok2 {
		t.Fatalf("independent launches: ok1=%v ok2=%v", ok1, ok2)
	}
	if len(launchedTags) != 2 {
		t.Fatalf("launched tags count=%d, want 2", len(launchedTags))
	}
	if launchedTags[0] == launchedTags[1] {
		t.Fatalf("tags collided: %v", launchedTags)
	}
}

func TestGuardCrashRSIStaleClaimRecovery(t *testing.T) {
	dir := t.TempDir()
	useRealGuardCrashRSIAdmission(t, dir)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldLaunch := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = oldLaunch })

	var launches int
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		launches++
		return nil
	}

	trace := "stale-recovery-trace"
	source := guardRSIDigest(trace)
	tag := "guard-crash-rsi/" + source
	id := launchguard.StableIdentity(tag)

	// Write an owner file for a dead PID older than 15 minutes (e.g. 20 minutes ago).
	staleTime := time.Now().Add(-20 * time.Minute)
	ownerPath := filepath.Join(dir, id+".owner")
	ownerData := fmt.Sprintf(`{"pid":9999999,"token":"stale-token","created_at":%q}`, staleTime.Format(time.RFC3339Nano))
	if err := os.WriteFile(ownerPath, []byte(ownerData), 0o600); err != nil {
		t.Fatal(err)
	}

	var stderr bytes.Buffer
	ok := guardMaybeLaunchCrashRSI(&stderr, new(guardRSISession), trace, "codex", "NONZERO_EXIT", 1, 0)
	if !ok {
		t.Fatalf("stale recovery launch refused: %s", stderr.String())
	}
	if launches != 1 {
		t.Fatalf("launches=%d, want 1", launches)
	}
	if !strings.Contains(stderr.String(), tag) {
		t.Fatalf("status missing tag: %s", stderr.String())
	}
}

func TestGuardFailureRSIHostWideAtomicClaim(t *testing.T) {
	dir := t.TempDir()
	useRealGuardCrashRSIAdmission(t, dir)
	t.Setenv(guardCrashRSIMarkerEnv, "")
	oldLaunch := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = oldLaunch })

	var launches int
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		launches++
		return nil
	}

	trace := "failure-receipt-trace"
	var stderr1 bytes.Buffer
	session1 := new(guardRSISession)
	ok1 := guardMaybeLaunchFailureRSI(&stderr1, session1, trace, "codex", guardTestContainmentSurvivorError(999))
	if !ok1 {
		t.Fatalf("first failure RSI did not launch: %s", stderr1.String())
	}
	if launches != 1 {
		t.Fatalf("launches=%d, want 1", launches)
	}

	// Second independent guard process with fresh session observing the same failure.
	var stderr2 bytes.Buffer
	session2 := new(guardRSISession)
	ok2 := guardMaybeLaunchFailureRSI(&stderr2, session2, trace, "codex", guardTestContainmentSurvivorError(999))
	if ok2 {
		t.Fatal("second failure RSI should have been refused as duplicate")
	}
	if launches != 1 {
		t.Fatalf("launches=%d, want still 1", launches)
	}
	if !strings.Contains(stderr2.String(), "failure RSI launch refused") || !strings.Contains(stderr2.String(), "duplicate-active") {
		t.Fatalf("stderr missing refusal message: %s", stderr2.String())
	}
}

func TestGuardCrashRSIHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	args := os.Args
	for len(args) > 0 {
		if args[0] == "--" {
			args = args[1:]
			break
		}
		args = args[1:]
	}
	if len(args) < 2 {
		os.Exit(3)
	}
	dir := args[0]
	tag := args[1]
	guardCrashRSIDir = dir
	guardCrashRSIAdmit = admitGuardCrashRSILaunch

	finish, decision, err := guardCrashRSIAdmit(tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "admit error: %v\n", err)
		os.Exit(4)
	}
	if finish == nil {
		fmt.Fprintf(os.Stderr, "refused: %s\n", decision.Outcome)
		os.Exit(2)
	}
	time.Sleep(20 * time.Millisecond)
	if err := finish(true); err != nil {
		fmt.Fprintf(os.Stderr, "finish error: %v\n", err)
		os.Exit(5)
	}
	os.Exit(0)
}

func TestGuardCrashRSIMultiProcessConcurrentClaim(t *testing.T) {
	dir := t.TempDir()
	const workers = 8
	tag := "guard-crash-rsi/multi-process-test"
	start := make(chan struct{})
	type res struct {
		code int
		err  string
	}
	results := make(chan res, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			cmd := exec.Command(os.Args[0], "-test.run=^TestGuardCrashRSIHelperProcess$", "--", dir, tag)
			cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			err := cmd.Run()
			code := 0
			if err != nil {
				var exitErr *exec.ExitError
				if errors.As(err, &exitErr) {
					code = exitErr.ExitCode()
				} else {
					code = -1
				}
			}
			results <- res{code: code, err: stderr.String()}
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	winners := 0
	refused := 0
	for r := range results {
		if r.code == 0 {
			winners++
		} else if r.code == 2 && strings.Contains(r.err, "refused: duplicate-active") {
			refused++
		} else {
			t.Errorf("unexpected helper exit code=%d err=%s", r.code, r.err)
		}
	}
	if winners != 1 || refused != workers-1 {
		t.Fatalf("winners=%d refused=%d, want 1 winner and %d refused", winners, refused, workers-1)
	}

	// Subsequent process within cooldown is also refused.
	cmd := exec.Command(os.Args[0], "-test.run=^TestGuardCrashRSIHelperProcess$", "--", dir, tag)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err := cmd.Run()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 || !strings.Contains(stderr.String(), "refused: duplicate-active") {
		t.Fatalf("subsequent call within cooldown exit=%v stderr=%s, want code 2 refused", err, stderr.String())
	}

	// Different tag admits independently.
	cmdOther := exec.Command(os.Args[0], "-test.run=^TestGuardCrashRSIHelperProcess$", "--", dir, "guard-crash-rsi/different-tag")
	cmdOther.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	if out, err := cmdOther.CombinedOutput(); err != nil {
		t.Fatalf("different tag launch failed: %v output=%s", err, out)
	}
}
