package main

import (
	"bytes"
	"encoding/json"
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
	if os.Getenv("GO_HELPER_MODE") == "crash_rsi_child_witness" {
		runGuardCrashRSIChildWitnessHelper()
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

	var finish func(bool) error
	var decision launchguard.Decision
	var err error
	for attempt := 0; attempt < 5; attempt++ {
		finish, decision, err = guardCrashRSIAdmit(tag)
		if err == nil || !strings.Contains(err.Error(), "used by another process") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "admit error: %v\n", err)
		os.Exit(4)
	}
	if finish == nil {
		fmt.Fprintf(os.Stderr, "refused: %s\n", decision.Outcome)
		os.Exit(2)
	}
	time.Sleep(20 * time.Millisecond)
	var finishErr error
	for attempt := 0; attempt < 5; attempt++ {
		finishErr = finish(true)
		if finishErr == nil || !strings.Contains(finishErr.Error(), "used by another process") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if finishErr != nil {
		fmt.Fprintf(os.Stderr, "finish error: %v\n", finishErr)
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

func runGuardCrashRSIChildWitnessHelper() {
	args := os.Args
	for len(args) > 0 {
		if args[0] == "--" {
			args = args[1:]
			break
		}
		args = args[1:]
	}

	provider := ""
	model := ""
	for i := 0; i < len(args); i++ {
		if args[i] == "-c" && i+1 < len(args) {
			val := args[i+1]
			if strings.HasPrefix(val, "model_provider=") {
				provider = strings.TrimPrefix(val, "model_provider=")
			}
		}
		if args[i] == "--model" && i+1 < len(args) {
			model = args[i+1]
		}
	}

	if provider == "" {
		provider = os.Getenv("FAK_GUARD_CRASH_RSI_PROVIDER")
	}
	if model == "" {
		model = os.Getenv("FAK_GUARD_CRASH_RSI_MODEL")
	}

	hasSecrets := false
	for _, envVar := range os.Environ() {
		upper := strings.ToUpper(envVar)
		if strings.HasPrefix(upper, "OPENAI_API_KEY=") ||
			strings.HasPrefix(upper, "ANTHROPIC_API_KEY=") ||
			strings.HasPrefix(upper, "FAK_SECRET_") {
			hasSecrets = true
			break
		}
	}

	exitCode := 0
	if codeStr := os.Getenv("FAK_WITNESS_EXIT_CODE"); codeStr != "" {
		fmt.Sscanf(codeStr, "%d", &exitCode)
	}

	errMsg := os.Getenv("FAK_WITNESS_ERROR_MESSAGE")
	if errMsg != "" && exitCode != 0 {
		fmt.Fprintln(os.Stderr, errMsg)
	}

	receiptPath := os.Getenv("FAK_WITNESS_RECEIPT")
	if receiptPath != "" {
		receipt := map[string]any{
			"tag":           os.Getenv(guardCrashRSIMarkerEnv),
			"provider":      provider,
			"model":         model,
			"has_secrets":   hasSecrets,
			"exit_code":     exitCode,
			"error_message": errMsg,
			"args":          args,
		}
		data, _ := json.Marshal(receipt)
		_ = os.WriteFile(receiptPath, data, 0o600)
	}

	os.Exit(exitCode)
}

func assertArgContains(t *testing.T, args []string, flag, val string) {
	t.Helper()
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == val {
			return
		}
	}
	t.Fatalf("args %v missing flag %q followed by %q", args, flag, val)
}

func TestGuardCrashRSILaunchCodexInjectsProviderAndModelOverrides(t *testing.T) {
	// Test 1: Default fallback provider ("fak") and default model (guardCodexDefaultModelID)
	reqDefault := guardCrashRSIRequest{
		Agent:     "codex",
		Workspace: t.TempDir(),
		Prompt:    "investigate",
	}
	name, args, err := guardCrashRSICommandArgs(reqDefault)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs failed: %v", err)
	}
	if name != "codex" {
		t.Fatalf("name = %q, want 'codex'", name)
	}
	assertArgContains(t, args, "-c", "model_provider="+guardCodexProviderID)
	assertArgContains(t, args, "--model", guardCodexDefaultModelID)
	if args[len(args)-1] != "investigate" {
		t.Fatalf("last arg = %q, want prompt 'investigate'", args[len(args)-1])
	}

	// Test 2: Explicit provider override
	reqProvider := guardCrashRSIRequest{
		Agent:     "codex",
		Provider:  "custom-provider",
		Workspace: t.TempDir(),
		Prompt:    "investigate",
	}
	_, argsProvider, err := guardCrashRSICommandArgs(reqProvider)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs failed: %v", err)
	}
	assertArgContains(t, argsProvider, "-c", "model_provider=custom-provider")

	// Test 3: Explicit model override
	reqModel := guardCrashRSIRequest{
		Agent:     "codex",
		Model:     "gpt-5.6-sol",
		Workspace: t.TempDir(),
		Prompt:    "investigate",
	}
	_, argsModel, err := guardCrashRSICommandArgs(reqModel)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs failed: %v", err)
	}
	assertArgContains(t, argsModel, "--model", "gpt-5.6-sol")

	// Test 4: Explicit fallback model when model is empty
	reqFallback := guardCrashRSIRequest{
		Agent:         "codex",
		FallbackModel: "fallback-model-x",
		Workspace:     t.TempDir(),
		Prompt:        "investigate",
	}
	_, argsFallback, err := guardCrashRSICommandArgs(reqFallback)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs failed: %v", err)
	}
	assertArgContains(t, argsFallback, "--model", "fallback-model-x")

	// Test 5: Environment variable overrides
	t.Setenv("FAK_GUARD_CRASH_RSI_PROVIDER", "env-provider")
	t.Setenv("FAK_GUARD_CRASH_RSI_MODEL", "env-model")
	reqEnv := guardCrashRSIRequest{
		Agent:     "codex",
		Workspace: t.TempDir(),
		Prompt:    "investigate",
	}
	_, argsEnv, err := guardCrashRSICommandArgs(reqEnv)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs failed: %v", err)
	}
	assertArgContains(t, argsEnv, "-c", "model_provider=env-provider")
	assertArgContains(t, argsEnv, "--model", "env-model")

	// Test 6: Verify launchGuardCrashRSI passes args through to exec.Command
	var capturedName string
	var capturedArgs []string
	oldLookPath, oldCommand := guardCrashRSILookPath, guardCrashRSICommand
	t.Cleanup(func() {
		guardCrashRSILookPath = oldLookPath
		guardCrashRSICommand = oldCommand
	})
	guardCrashRSILookPath = func(name string) (string, error) {
		return "/mock/" + name, nil
	}
	guardCrashRSICommand = func(name string, args ...string) *exec.Cmd {
		capturedName = name
		capturedArgs = args
		return exec.Command("cmd.exe", "/c", "exit 0")
	}

	reqLaunch := guardCrashRSIRequest{
		Tag:       "guard-crash-rsi/test",
		Agent:     "codex",
		Provider:  "test-provider",
		Model:     "test-model",
		Workspace: t.TempDir(),
		Prompt:    "launch prompt",
	}
	_ = launchGuardCrashRSI(reqLaunch)
	if capturedName != "/mock/codex" {
		t.Fatalf("captured executable name = %q, want '/mock/codex'", capturedName)
	}
	assertArgContains(t, capturedArgs, "-c", "model_provider=test-provider")
	assertArgContains(t, capturedArgs, "--model", "test-model")

	// Test 7: Claude agent supports model flag
	reqClaude := guardCrashRSIRequest{
		Agent:     "claude",
		Model:     "claude-haiku-4-5",
		Workspace: t.TempDir(),
		Prompt:    "claude prompt",
	}
	_, argsClaude, err := guardCrashRSICommandArgs(reqClaude)
	if err != nil {
		t.Fatalf("guardCrashRSICommandArgs claude failed: %v", err)
	}
	assertArgContains(t, argsClaude, "--model", "claude-haiku-4-5")
}

func TestGuardCrashRSIChildRouteIdentityAndTerminalFailureWitness(t *testing.T) {
	resetGuardCrashRSITerminalRecords()
	t.Cleanup(resetGuardCrashRSITerminalRecords)

	oldLookPath, oldCommand := guardCrashRSILookPath, guardCrashRSICommand
	t.Cleanup(func() {
		guardCrashRSILookPath = oldLookPath
		guardCrashRSICommand = oldCommand
	})

	guardCrashRSILookPath = func(name string) (string, error) {
		return os.Args[0], nil
	}
	guardCrashRSICommand = func(name string, args ...string) *exec.Cmd {
		testArgs := []string{"-test.run=^TestGuardCrashRSIHelperProcess$", "--"}
		testArgs = append(testArgs, args...)
		return exec.Command(name, testArgs...)
	}

	dir := t.TempDir()
	receiptPath := filepath.Join(dir, "receipt.json")

	t.Setenv("OPENAI_API_KEY", "secret-key-12345")
	t.Setenv("ANTHROPIC_API_KEY", "secret-anthropic-key")

	waitDone := make(chan guardCrashRSITerminalRecord, 1)
	var stderrBuf bytes.Buffer

	req := guardCrashRSIRequest{
		Tag:       "guard-crash-rsi/witness-test-tag",
		Source:    "witness-test-tag",
		Agent:     "codex",
		Class:     "NONZERO_EXIT",
		ExitCode:  1,
		Workspace: dir,
		Provider:  "fak",
		Model:     "gpt-6-astra",
		Prompt:    "investigate crash root cause",
		Stderr:    &stderrBuf,
		Env: []string{
			"GO_WANT_HELPER_PROCESS=1",
			"GO_HELPER_MODE=crash_rsi_child_witness",
			"FAK_WITNESS_RECEIPT=" + receiptPath,
			"FAK_WITNESS_EXIT_CODE=42",
			"FAK_WITNESS_ERROR_MESSAGE=OpenAI error: model 'gpt-5.6-sol' rejected with HTTP 400 Bad Request",
		},
		OnWait: func(rec guardCrashRSITerminalRecord) {
			waitDone <- rec
		},
	}

	err := launchGuardCrashRSI(req)
	if err != nil {
		t.Fatalf("launchGuardCrashRSI failed: %v", err)
	}

	var termRec guardCrashRSITerminalRecord
	select {
	case termRec = <-waitDone:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for child terminal failure record")
	}

	// 1. Verify terminal record captured failure details
	if termRec.ExitCode != 42 {
		t.Fatalf("terminal record exit code = %d, want 42", termRec.ExitCode)
	}
	if termRec.Tag != req.Tag {
		t.Fatalf("terminal record tag = %q, want %q", termRec.Tag, req.Tag)
	}
	if termRec.Provider != "fak" {
		t.Fatalf("terminal record provider = %q, want 'fak'", termRec.Provider)
	}
	if termRec.Model != "gpt-6-astra" {
		t.Fatalf("terminal record model = %q, want 'gpt-6-astra'", termRec.Model)
	}
	if !strings.Contains(termRec.Stderr, "HTTP 400 Bad Request") {
		t.Fatalf("terminal record stderr = %q, want to contain 'HTTP 400 Bad Request'", termRec.Stderr)
	}

	// 2. Verify stderr output logged the terminal failure
	loggedStderr := stderrBuf.String()
	if !strings.Contains(loggedStderr, "exit 42") {
		t.Fatalf("logged stderr missing exit 42: %s", loggedStderr)
	}
	if !strings.Contains(loggedStderr, req.Tag) {
		t.Fatalf("logged stderr missing tag: %s", loggedStderr)
	}
	if !strings.Contains(loggedStderr, "HTTP 400 Bad Request") {
		t.Fatalf("logged stderr missing child error message: %s", loggedStderr)
	}

	// 3. Verify integration receipt written by child
	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("failed to read witness receipt: %v", err)
	}
	var receipt map[string]any
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("failed to parse witness receipt: %v", err)
	}
	if receipt["provider"] != "fak" {
		t.Fatalf("witness receipt provider = %v, want 'fak'", receipt["provider"])
	}
	if receipt["model"] != "gpt-6-astra" {
		t.Fatalf("witness receipt model = %v, want 'gpt-6-astra'", receipt["model"])
	}
	if receipt["has_secrets"] != false {
		t.Fatal("witness receipt indicates child received secrets in environment")
	}
	if receipt["exit_code"] != float64(42) {
		t.Fatalf("witness receipt exit_code = %v, want 42", receipt["exit_code"])
	}

	// 4. Test read-only child baseline with exit code 0
	cleanReceiptPath := filepath.Join(dir, "clean_receipt.json")
	var cleanStderr bytes.Buffer
	cleanDone := make(chan guardCrashRSITerminalRecord, 1)

	cleanReq := guardCrashRSIRequest{
		Tag:       "guard-crash-rsi/clean-baseline",
		Source:    "clean-baseline",
		Agent:     "codex",
		Class:     "NONZERO_EXIT",
		ExitCode:  1,
		Workspace: dir,
		Provider:  "fak",
		Model:     "gpt-6-astra",
		Prompt:    "clean run",
		Stderr:    &cleanStderr,
		Env: []string{
			"GO_WANT_HELPER_PROCESS=1",
			"GO_HELPER_MODE=crash_rsi_child_witness",
			"FAK_WITNESS_RECEIPT=" + cleanReceiptPath,
			"FAK_WITNESS_EXIT_CODE=0",
		},
		OnWait: func(rec guardCrashRSITerminalRecord) {
			cleanDone <- rec
		},
	}

	if err := launchGuardCrashRSI(cleanReq); err != nil {
		t.Fatalf("launchGuardCrashRSI clean run failed: %v", err)
	}

	select {
	case cleanRec := <-cleanDone:
		if cleanRec.ExitCode != 0 {
			t.Fatalf("clean run exit code = %d, want 0", cleanRec.ExitCode)
		}
		if cleanRec.Error != "" {
			t.Fatalf("clean run error = %q, want empty", cleanRec.Error)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for clean run completion")
	}

	if cleanStderr.Len() != 0 {
		t.Fatalf("clean run produced unexpected stderr output: %s", cleanStderr.String())
	}
}
