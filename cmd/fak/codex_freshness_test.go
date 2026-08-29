package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestCodexStartupStatusCapturedRenderReplacesAndClears(t *testing.T) {
	var out bytes.Buffer
	status := newCodexStartupStatus(&out, true)
	status.Start("checking launcher")
	status.Update("updating launcher abc123 -> def456")
	status.Stop()
	want := "\r\x1b[2K⠋ fak codex · checking launcher" +
		"\r\x1b[2K⠙ fak codex · updating launcher abc123 -> def456" +
		"\r\x1b[2K"
	if got := out.String(); got != want {
		t.Fatalf("captured startup render=%q, want %q", got, want)
	}
	if bytes.Contains(out.Bytes(), []byte("\n")) {
		t.Fatalf("transient status leaked a persistent line: %q", out.String())
	}
}

func TestCodexStartupStatusNonInteractiveOnlyReportsWork(t *testing.T) {
	var out bytes.Buffer
	status := newCodexStartupStatus(&out, false)
	status.Start("checking launcher")
	if out.Len() != 0 {
		t.Fatalf("fast admission emitted startup noise: %q", out.String())
	}
	status.Update("updating launcher abc123 -> def456")
	if got := out.String(); got != "fak codex: updating launcher abc123 -> def456\n" {
		t.Fatalf("non-interactive update status=%q", got)
	}
}

func TestRunCodexFreshnessAdmissionFreshIsSilent(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "abc123", TargetCommit: "abc123"})
	defer restore()
	var out bytes.Buffer
	codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(&out, false) }

	args, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
	if code != 0 || stop || !reflect.DeepEqual(args, []string{"--model", "gpt-5"}) {
		t.Fatalf("args=%q code=%d stop=%v", args, code, stop)
	}
	if out.Len() != 0 {
		t.Fatalf("fresh startup emitted %q", out.String())
	}
}

func TestRunCodexFreshnessAdmissionShowsSelfUpdate(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "aaaaaaaaaaaa111", TargetCommit: "bbbbbbbbbbbb222"})
	defer restore()
	var out bytes.Buffer
	codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(&out, false) }
	updated := false
	codexFreshnessUpdate = func(_, _ string) error { updated = true; return nil }
	codexFreshnessReexec = func(_ string, _ []string) error { return nil }

	_, code, stop := runCodexFreshnessAdmission(nil)
	if code != 0 || !stop || !updated {
		t.Fatalf("code=%d stop=%v updated=%v", code, stop, updated)
	}
	want := "fak codex: updating launcher aaaaaaaaaaaa -> bbbbbbbbbbbb at C:\\bin\\fak.exe\n" +
		"fak codex: launching updated build bbbbbbbbbbbb from C:\\bin\\fak.exe\n"
	if got := out.String(); got != want {
		t.Fatalf("update status=%q, want %q", got, want)
	}
}

func TestCodexFreshnessSelfUpdateUsesSupportedFlags(t *testing.T) {
	want := []string{"self-update", "--root", `C:\work\fak`, "--target", `C:\bin\fak.exe`}
	if got := codexFreshnessSelfUpdateArgs(`C:\work\fak`, `C:\bin\fak.exe`); !reflect.DeepEqual(got, want) {
		t.Fatalf("self-update args=%q, want %q", got, want)
	}
}

func TestRunCodexFreshnessAdmissionStalePreservesFilteredArguments(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"})
	defer restore()
	codexFreshnessUpdate = func(_, _ string) error { return nil }
	var got []string
	codexFreshnessReexec = func(_ string, argv []string) error { got = append([]string(nil), argv...); return nil }

	_, code, stop := runCodexFreshnessAdmission([]string{"--freshness-gate=on", "--model", "gpt-5", "resume", "thread-7"})
	want := []string{`C:\bin\fak.exe`, "codex", "--model", "gpt-5", "resume", "thread-7"}
	if code != 0 || !stop || !reflect.DeepEqual(got, want) {
		t.Fatalf("code=%d stop=%v reexec=%q, want %q", code, stop, got, want)
	}
}

func TestRunCodexFreshnessAdmissionFailuresRefuseLaunch(t *testing.T) {
	for _, tc := range []struct {
		name       string
		assessment codexFreshnessAssessment
		updateErr  error
	}{
		{name: "unknown provenance", assessment: codexFreshnessAssessment{Verdict: codexFreshnessUnknown, Detail: "unstamped build"}},
		{name: "update failed", assessment: codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"}, updateErr: errors.New("gate failed")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restore := stubCodexFreshness(t, tc.assessment)
			defer restore()
			codexFreshnessUpdate = func(_, _ string) error { return tc.updateErr }
			_, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
			if code == 0 || !stop {
				t.Fatalf("code=%d stop=%v, want refusal", code, stop)
			}
		})
	}
}

func TestRunCodexFreshnessAdmissionRecursionRefusesSecondUpdate(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"})
	defer restore()
	t.Setenv(codexFreshnessReexecEnv, "1")
	updated := false
	codexFreshnessUpdate = func(_, _ string) error { updated = true; return nil }
	_, code, stop := runCodexFreshnessAdmission(nil)
	if code == 0 || !stop || updated {
		t.Fatalf("code=%d stop=%v updated=%v, want refusal before update", code, stop, updated)
	}
}

func TestRunCodexFreshnessAdmissionEscapeAndNoCheckoutContinue(t *testing.T) {
	t.Run("explicit escape", func(t *testing.T) {
		restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessUnknown})
		defer restore()
		args, code, stop := runCodexFreshnessAdmission([]string{"--freshness-gate", "off", "--model", "gpt-5"})
		if code != 0 || stop || !reflect.DeepEqual(args, []string{"--model", "gpt-5"}) {
			t.Fatalf("args=%q code=%d stop=%v", args, code, stop)
		}
	})
	t.Run("no checkout", func(t *testing.T) {
		restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessUnknown})
		defer restore()
		codexFreshnessResolveCheckout = func() (string, string, error) { return "", "", nil }
		args, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
		if code != 0 || stop || !reflect.DeepEqual(args, []string{"--model", "gpt-5"}) {
			t.Fatalf("args=%q code=%d stop=%v", args, code, stop)
		}
	})
}

func TestIsNotGitRepositoryUsesGitStderr(t *testing.T) {
	if !isNotGitRepository(errors.New("exit status 128"), []byte("fatal: not a git repository")) {
		t.Fatal("git stderr was not classified as no-checkout")
	}
}

func stubCodexFreshness(t *testing.T, assessment codexFreshnessAssessment) func() {
	t.Helper()
	oldExecutable, oldGetwd := codexFreshnessExecutable, codexFreshnessGetwd
	oldInspect, oldUpdate := codexFreshnessInspect, codexFreshnessUpdate
	oldReexec, oldStatus := codexFreshnessReexec, codexFreshnessStatus
	oldResolveContext := codexFreshnessResolveCheckout
	oldNow, oldCacheDir := codexFreshnessNow, codexFreshnessCacheDir
	oldArgs := append([]string(nil), os.Args...)
	codexFreshnessExecutable = func() (string, error) { return `C:\bin\fak.exe`, nil }
	codexFreshnessGetwd = func() (string, error) { return `C:\work\fak`, nil }
	codexFreshnessResolveCheckout = func() (string, string, error) { return `C:\work\fak`, `C:\bin\fak.exe`, nil }
	codexFreshnessNow = func() time.Time { return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC) }
	cacheDir := t.TempDir()
	codexFreshnessCacheDir = func() (string, error) { return cacheDir, nil }
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection { return codexFreshnessInspection{Assessment: assessment} }
	codexFreshnessUpdate = func(_, _ string) error { return errors.New("unexpected update") }
	codexFreshnessReexec = func(_ string, _ []string) error { return errors.New("unexpected reexec") }
	codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(io.Discard, false) }
	os.Args = []string{"fak", "codex"}
	t.Setenv(codexFreshnessReexecEnv, "")
	return func() {
		codexFreshnessExecutable, codexFreshnessGetwd = oldExecutable, oldGetwd
		codexFreshnessInspect, codexFreshnessUpdate = oldInspect, oldUpdate
		codexFreshnessReexec, codexFreshnessStatus = oldReexec, oldStatus
		codexFreshnessResolveCheckout = oldResolveContext
		codexFreshnessNow, codexFreshnessCacheDir = oldNow, oldCacheDir
		os.Args = oldArgs
	}
}

func codexFreshnessTestStatePath(t *testing.T) string {
	t.Helper()
	root, executable, err := codexFreshnessResolveCheckout()
	if err != nil {
		t.Fatal(err)
	}
	path, err := codexFreshnessStatePath(root, executable)
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCodexFreshnessValidLeaseSkipsInspection(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	if err := codexFreshnessWriteLease(statePath+".json", codexFreshnessNow()); err != nil {
		t.Fatal(err)
	}
	called := false
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		called = true
		return codexFreshnessInspection{}
	}
	args, code, stop := runCodexFreshnessAdmission([]string{"--model", "x"})
	if stop || code != 0 || called || !reflect.DeepEqual(args, []string{"--model", "x"}) {
		t.Fatalf("args=%q code=%d stop=%v inspected=%v", args, code, stop, called)
	}
}

func TestCodexFreshnessCheckNowBypassesLeaseAndIsStripped(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	if err := codexFreshnessWriteLease(statePath+".json", codexFreshnessNow()); err != nil {
		t.Fatal(err)
	}
	called := false
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		called = true
		return codexFreshnessInspection{Assessment: codexFreshnessAssessment{Verdict: codexFreshnessFresh}}
	}
	args, code, stop := runCodexFreshnessAdmission([]string{"--freshness-check-now", "--model", "x"})
	if stop || code != 0 || !called || !reflect.DeepEqual(args, []string{"--model", "x"}) {
		t.Fatalf("args=%q code=%d stop=%v inspected=%v", args, code, stop, called)
	}
}

func TestCodexFreshnessLiveClaimUsesCurrentLauncherImmediately(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	claimed, err := codexFreshnessAcquireClaim(statePath+".lock", codexFreshnessNow())
	if err != nil || !claimed {
		t.Fatalf("claim=%v err=%v", claimed, err)
	}
	called := false
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		called = true
		return codexFreshnessInspection{}
	}
	args, code, stop := runCodexFreshnessAdmission([]string{"--model", "x"})
	if stop || code != 0 || called || !reflect.DeepEqual(args, []string{"--model", "x"}) {
		t.Fatalf("args=%q code=%d stop=%v inspected=%v", args, code, stop, called)
	}
}

func TestCodexFreshnessAcquireClaimDoesNotReapAlreadyClaimedStaleMarker(t *testing.T) {
	path := t.TempDir() + string(os.PathSeparator) + "freshness.lock"
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	if err := os.WriteFile(path, []byte("stale-owner\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	stale := now.Add(-codexFreshnessClaimTTL - time.Minute)
	if err := os.Chtimes(path, stale, stale); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	tombstone := codexFreshnessStaleClaimPath(path, raw, info)
	if err := os.Link(path, tombstone); err != nil {
		t.Fatal(err)
	}

	claimed, err := codexFreshnessAcquireClaim(path, now)
	if err != nil {
		t.Fatal(err)
	}
	if claimed {
		t.Fatal("second stale reaper acquired claim")
	}
	if got, err := os.ReadFile(path); err != nil || string(got) != string(raw) {
		t.Fatalf("second stale reaper changed live claim: got=%q err=%v", got, err)
	}
}
func TestCodexFreshnessStaleClaimIsReaped(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	claimPath := statePath + ".lock"
	if err := os.WriteFile(claimPath, []byte("stale\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	stale := codexFreshnessNow().Add(-codexFreshnessClaimTTL - time.Minute)
	if err := os.Chtimes(claimPath, stale, stale); err != nil {
		t.Fatal(err)
	}
	called := false
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		called = true
		return codexFreshnessInspection{Assessment: codexFreshnessAssessment{Verdict: codexFreshnessFresh}}
	}
	_, code, stop := runCodexFreshnessAdmission(nil)
	if stop || code != 0 || !called {
		t.Fatalf("code=%d stop=%v inspected=%v", code, stop, called)
	}
	if _, err := os.Stat(claimPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("claim remains after admission: %v", err)
	}
}

func TestCodexFreshnessFreshInspectionWritesLease(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	_, code, stop := runCodexFreshnessAdmission(nil)
	if stop || code != 0 {
		t.Fatalf("code=%d stop=%v", code, stop)
	}
	if !codexFreshnessLeaseValid(statePath+".json", codexFreshnessNow()) {
		t.Fatal("successful inspection did not persist a valid lease")
	}
}

func TestCodexFreshnessInspectionErrorRemainsFailClosed(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{})
	defer restore()
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		return codexFreshnessInspection{Err: errors.New("inspect failed")}
	}
	args, code, stop := runCodexFreshnessAdmission(nil)
	if !stop || code != 1 || args != nil {
		t.Fatalf("args=%q code=%d stop=%v", args, code, stop)
	}
}

func TestCodexFreshnessLeaseRenewsOnWindowsCompatiblePath(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	statePath := codexFreshnessTestStatePath(t) + ".json"
	first := codexFreshnessNow().Add(-codexFreshnessLeaseTTL - time.Minute)
	if err := codexFreshnessWriteLease(statePath, first); err != nil {
		t.Fatal(err)
	}
	if err := codexFreshnessWriteLease(statePath, codexFreshnessNow()); err != nil {
		t.Fatalf("renew lease: %v", err)
	}
	if !codexFreshnessLeaseValid(statePath, codexFreshnessNow()) {
		t.Fatal("renewed lease is not valid")
	}
}
