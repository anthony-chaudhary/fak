package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const codexFreshnessReexecHelperEnv = "GO_WANT_CODEX_FRESHNESS_REEXEC_HELPER"

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

func TestRunCodexFreshnessAdmissionFreshConsumesInheritedHandoff(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh})
	defer restore()
	t.Setenv(codexFreshnessReexecEnv, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:4242")
	_, code, stop := runCodexFreshnessAdmission(nil)
	if code != 0 || stop || os.Getenv(codexFreshnessReexecEnv) != "" {
		t.Fatalf("code=%d stop=%v marker=%q, want fresh admission to consume one-shot handoff", code, stop, os.Getenv(codexFreshnessReexecEnv))
	}
}

func TestRunCodexFreshnessAdmissionShowsSelfUpdate(t *testing.T) {
	const installed = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "aaaaaaaaaaaa111", TargetCommit: installed})
	defer restore()
	var out bytes.Buffer
	codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(&out, false) }
	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) { updated = true; return installed, nil }
	codexFreshnessReexec = func(_ string, _ []string, _ string) error { return nil }

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

func TestRunCodexFreshnessAdmissionDirtyWithTargetRebuilds(t *testing.T) {
	const (
		running   = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		installed = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	)
	restore := stubCodexFreshness(t, codexFreshnessAssessment{
		Verdict:       codexFreshnessUnknown,
		RunningCommit: running,
		TargetCommit:  installed,
		Detail:        "DIRTY",
	})
	defer restore()

	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) {
		updated = true
		return installed, nil
	}
	var gotTarget string
	codexFreshnessReexec = func(_ string, _ []string, expectedCommit string) error {
		gotTarget = expectedCommit
		return nil
	}

	_, code, stop := runCodexFreshnessAdmission(nil)
	if code != 0 || !stop || !updated || gotTarget != installed {
		t.Fatalf("code=%d stop=%v updated=%v reexec_target=%q, want guarded rebuild and re-exec at %q", code, stop, updated, gotTarget, installed)
	}
}

func TestRunCodexFreshnessAdmissionDirtyWithoutTargetRefuses(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{
		Verdict:       codexFreshnessUnknown,
		RunningCommit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Detail:        "DIRTY",
	})
	defer restore()

	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) {
		updated = true
		return "", nil
	}

	_, code, stop := runCodexFreshnessAdmission(nil)
	if code == 0 || !stop || updated {
		t.Fatalf("code=%d stop=%v updated=%v, want dirty launcher without committed target refused before rebuild", code, stop, updated)
	}
}

func TestRunCodexFreshnessAdmissionDirtyCannotUseInheritedHandoff(t *testing.T) {
	const running = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	restore := stubCodexFreshness(t, codexFreshnessAssessment{
		Verdict:       codexFreshnessUnknown,
		RunningCommit: running,
		TargetCommit:  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Detail:        "DIRTY",
	})
	defer restore()
	codexFreshnessParentPID = func() int { return 4242 }
	t.Setenv(codexFreshnessReexecEnv, running+":4242")

	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) {
		updated = true
		return "", nil
	}

	args, code, stop := runCodexFreshnessAdmission(nil)
	if args != nil || code == 0 || !stop || updated {
		t.Fatalf("args=%q code=%d stop=%v updated=%v, want dirty launcher refused rather than admitted by inherited handoff", args, code, stop, updated)
	}
}

func TestInspectCodexFreshnessDirtyResolvesCommittedTarget(t *testing.T) {
	const target = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	running := binstamp.Stamp{
		Revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Dirty:    true,
		HasVCS:   true,
	}
	run := func(_ context.Context, dir, name string, args ...string) (string, bool) {
		if dir != "/repo" || name != "git" {
			t.Fatalf("runner dir=%q name=%q", dir, name)
		}
		want := []string{"rev-parse", "--verify", "--quiet", "origin/main^{commit}"}
		if !reflect.DeepEqual(args, want) {
			t.Fatalf("runner args=%q, want %q", args, want)
		}
		return target + "\n", true
	}

	got := inspectCodexFreshness("/repo", running, versionskew.Runner(run)).Assessment
	if got.Detail != versionskew.Dirty.String() || got.TargetCommit != target {
		t.Fatalf("assessment=%+v, want dirty verdict with target %q", got, target)
	}
}

func TestInspectCodexFreshnessDirtyWithoutResolvableTargetFailsClosed(t *testing.T) {
	running := binstamp.Stamp{
		Revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Dirty:    true,
		HasVCS:   true,
	}
	tests := []struct {
		name string
		out  string
		ok   bool
	}{
		{name: "missing ref", ok: false},
		{name: "malformed output", out: "not-a-full-commit\n", ok: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := func(context.Context, string, string, ...string) (string, bool) {
				return tt.out, tt.ok
			}
			got := inspectCodexFreshness("/repo", running, versionskew.Runner(run)).Assessment
			if got.Detail != versionskew.Dirty.String() || got.TargetCommit != "" {
				t.Fatalf("assessment=%+v, want dirty verdict without target", got)
			}
		})
	}
}

func TestCodexFreshnessSelfUpdateUsesSupportedFlags(t *testing.T) {
	want := []string{"self-update", "--json", "--root", `C:\work\fak`, "--target", `C:\bin\fak.exe`}
	if got := codexFreshnessSelfUpdateArgs(`C:\work\fak`, `C:\bin\fak.exe`); !reflect.DeepEqual(got, want) {
		t.Fatalf("self-update args=%q, want %q", got, want)
	}
	if got := codexFreshnessSelfUpdateArgs(`C:\work\fak`, `C:\bin\fak.exe`, false); !reflect.DeepEqual(got, want) {
		t.Fatalf("self-update args (verbose=false)=%q, want %q", got, want)
	}
	wantVerbose := []string{"self-update", "--json", "--root", `C:\work\fak`, "--target", `C:\bin\fak.exe`, "--verbose"}
	if got := codexFreshnessSelfUpdateArgs(`C:\work\fak`, `C:\bin\fak.exe`, true); !reflect.DeepEqual(got, wantVerbose) {
		t.Fatalf("self-update args (verbose=true)=%q, want %q", got, wantVerbose)
	}
}

func TestCodexFreshnessAdmissionStopsStatusBeforeUpdate(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"})
	defer restore()

	var order []string
	var out bytes.Buffer
	status := newCodexStartupStatus(&out, true)
	oldStatus := codexFreshnessStatus
	codexFreshnessStatus = func() *codexStartupStatus {
		return status
	}
	defer func() { codexFreshnessStatus = oldStatus }()

	codexFreshnessUpdate = func(_, _ string) (string, error) {
		order = append(order, "update")
		if !strings.Contains(out.String(), "\r\x1b[2K") {
			t.Errorf("status line was not cleared before update")
		}
		return "new", nil
	}
	codexFreshnessReexec = func(_ string, _ []string, _ string) error {
		order = append(order, "reexec")
		return nil
	}

	_, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
	if code != 0 || !stop {
		t.Fatalf("expected code=0, stop=true, got code=%d, stop=%v", code, stop)
	}
	if len(order) != 2 || order[0] != "update" || order[1] != "reexec" {
		t.Fatalf("unexpected order: %v", order)
	}
}

func TestParseCodexFreshnessSettingsVerbose(t *testing.T) {
	// Flag --verbose
	filtered, policy, err := parseCodexFreshnessSettings([]string{"--verbose", "--model", "gpt-5"}, "", "", codexFreshnessConfig{})
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !policy.Verbose {
		t.Fatalf("expected policy.Verbose to be true")
	}
	if !reflect.DeepEqual(filtered, []string{"--verbose", "--model", "gpt-5"}) {
		t.Fatalf("expected --verbose preserved in filtered args, got: %v", filtered)
	}

	// Flag -v
	filtered, policy, err = parseCodexFreshnessSettings([]string{"-v", "--model", "gpt-5"}, "", "", codexFreshnessConfig{})
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !policy.Verbose {
		t.Fatalf("expected policy.Verbose to be true")
	}
	if !reflect.DeepEqual(filtered, []string{"-v", "--model", "gpt-5"}) {
		t.Fatalf("expected -v preserved in filtered args, got: %v", filtered)
	}

	// Env var FAK_SELF_UPDATE_VERBOSE
	t.Setenv("FAK_SELF_UPDATE_VERBOSE", "1")
	_, policy, err = parseCodexFreshnessSettings([]string{"--model", "gpt-5"}, "", "", codexFreshnessConfig{})
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !policy.Verbose {
		t.Fatalf("expected policy.Verbose to be true from env var")
	}
}

func TestRunCodexFreshnessAdmissionStalePreservesFilteredArguments(t *testing.T) {
	const (
		observed  = "0123456789abcdef0123456789abcdef01234567"
		installed = "89abcdef0123456789abcdef0123456789abcdef"
	)
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: observed})
	defer restore()
	codexFreshnessUpdate = func(_, _ string) (string, error) { return installed, nil }
	var got []string
	gotTarget := ""
	codexFreshnessReexec = func(_ string, argv []string, expectedCommit string) error {
		got = append([]string(nil), argv...)
		gotTarget = expectedCommit
		return nil
	}

	_, code, stop := runCodexFreshnessAdmission([]string{"--freshness-gate=on", "--model", "gpt-5", "resume", "thread-7"})
	want := []string{`C:\bin\fak.exe`, "codex", "--model", "gpt-5", "resume", "thread-7"}
	if code != 0 || !stop || !reflect.DeepEqual(got, want) || gotTarget != installed {
		t.Fatalf("code=%d stop=%v reexec=%q target=%q, want %q at transaction commit %q (not pre-update observation %q)", code, stop, got, gotTarget, want, installed, observed)
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
			codexFreshnessUpdate = func(_, _ string) (string, error) { return "", tc.updateErr }
			_, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
			if code == 0 || !stop {
				t.Fatalf("code=%d stop=%v, want refusal", code, stop)
			}
		})
	}
}

func TestRunCodexFreshnessAdmissionPrintsCacheRecoveryExhaustion(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"})
	defer restore()
	const diagnostic = "vet: Go build cache /tmp/fak-cache became unavailable after the one bounded recovery; stop concurrent cache cleanup and rerun fak self-update"
	codexFreshnessUpdate = func(_, _ string) (string, error) {
		return "", errors.New("exit status 1: self-update receipt status is \"gate_failed\": " + diagnostic)
	}

	readEnd, writeEnd, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	originalStderr := os.Stderr
	os.Stderr = writeEnd
	_, code, stop := runCodexFreshnessAdmission(nil)
	_ = writeEnd.Close()
	os.Stderr = originalStderr
	captured, readErr := io.ReadAll(readEnd)
	_ = readEnd.Close()
	if readErr != nil {
		t.Fatal(readErr)
	}
	if code == 0 || !stop {
		t.Fatalf("code=%d stop=%v, want fail-closed freshness refusal", code, stop)
	}
	if got := string(captured); !strings.Contains(got, diagnostic) || !strings.Contains(got, "freshness admission refused") {
		t.Fatalf("freshness stderr=%q, want cache exhaustion and refusal context", got)
	}
}

func TestRunCodexFreshnessAdmissionRecursionRefusesSecondUpdate(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: "old", TargetCommit: "new"})
	defer restore()
	t.Setenv(codexFreshnessReexecEnv, "1")
	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) { updated = true; return "", nil }
	_, code, stop := runCodexFreshnessAdmission(nil)
	if code == 0 || !stop || updated {
		t.Fatalf("code=%d stop=%v updated=%v, want refusal before update", code, stop, updated)
	}
}

func TestRunCodexFreshnessAdmissionReexecRefusesWrongParent(t *testing.T) {
	const selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessBehind, RunningCommit: selected, TargetCommit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"})
	defer restore()
	codexFreshnessParentPID = func() int { return 4243 }
	t.Setenv(codexFreshnessReexecEnv, selected+":4242")
	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) { updated = true; return "", nil }
	_, code, stop := runCodexFreshnessAdmission(nil)
	if code == 0 || !stop || updated {
		t.Fatalf("code=%d stop=%v updated=%v, want parent-mismatched handoff refused", code, stop, updated)
	}
}

func TestRunCodexFreshnessAdmissionReexecAdmitsSelectedCommitAfterRefAdvances(t *testing.T) {
	const (
		selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		advanced = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	)
	restore := stubCodexFreshness(t, codexFreshnessAssessment{
		Verdict:       codexFreshnessBehind,
		RunningCommit: selected,
		TargetCommit:  advanced,
	})
	defer restore()
	codexFreshnessParentPID = func() int { return 4242 }
	t.Setenv(codexFreshnessReexecEnv, selected+":4242")

	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) { updated = true; return "", nil }
	args, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
	if code != 0 || stop || updated || !reflect.DeepEqual(args, []string{"--model", "gpt-5"}) {
		t.Fatalf("args=%q code=%d stop=%v updated=%v, want immediate child at selected A admitted while live tip is B", args, code, stop, updated)
	}
	if got := os.Getenv(codexFreshnessReexecEnv); got != "" {
		t.Fatalf("successful child admission retained reusable handoff marker %q", got)
	}
}

func TestRunCodexFreshnessAdmissionHandoffCannotWeakenOrdinaryAdmission(t *testing.T) {
	const selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	restore := stubCodexFreshness(t, codexFreshnessAssessment{
		Verdict:       codexFreshnessBehind,
		RunningCommit: selected,
		TargetCommit:  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	})
	defer restore()
	updated := false
	codexFreshnessUpdate = func(_, _ string) (string, error) {
		updated = true
		return "", errors.New("ordinary admission observed newer tip")
	}
	_, code, stop := runCodexFreshnessAdmission(nil)
	if code == 0 || !stop || !updated {
		t.Fatalf("code=%d stop=%v updated=%v, want ordinary admission to observe newer tip", code, stop, updated)
	}
}

func TestCodexFreshnessInstalledRevisionRequiresChangedPrimaryReceipt(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	receipt := selfUpdateReceipt{
		Schema:        selfUpdateReceiptSchema,
		SchemaVersion: 1,
		Status:        "updated",
		Changed:       1,
		NewRevision:   stringPtr(revision),
		Targets:       []selfUpdateReceiptTarget{{Role: "primary", Path: `C:\bin\fak.exe`}},
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := codexFreshnessInstalledRevision(data, `C:\bin\fak.exe`); err != nil || got != revision {
		t.Fatalf("revision=%q err=%v, want transaction revision %q", got, err, revision)
	}
	receipt.Changed = 0
	data, _ = json.Marshal(receipt)
	if _, err := codexFreshnessInstalledRevision(data, `C:\bin\fak.exe`); err == nil {
		t.Fatal("unchanged receipt admitted a re-exec identity")
	}
	receipt.Changed = 1
	receipt.Status = "hot-copy-divergent"
	data, _ = json.Marshal(receipt)
	if _, err := codexFreshnessInstalledRevision(data, `C:\bin\fak.exe`); err == nil {
		t.Fatal("non-updated receipt status admitted a re-exec identity")
	}
}

func TestCodexFreshnessInstalledRevisionPreservesFailedReceiptDetail(t *testing.T) {
	const diagnostic = "vet: Go build cache /tmp/fak-cache became unavailable after the one bounded recovery; stop concurrent cache cleanup and rerun fak self-update"
	receipt := selfUpdateReceipt{
		Schema:        selfUpdateReceiptSchema,
		SchemaVersion: 1,
		Status:        "gate_failed",
		Detail:        diagnostic,
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	_, err = codexFreshnessInstalledRevision(data, `C:\bin\fak.exe`)
	if err == nil || !strings.Contains(err.Error(), diagnostic) {
		t.Fatalf("failed receipt error=%v, want actionable cache detail preserved", err)
	}
}

func TestCodexFreshnessInstalledRevisionDivergentReceiptIsActionable(t *testing.T) {
	const diagnostic = "target installed; convergeable hot copies still need repair"
	receipt := selfUpdateReceipt{
		Schema:        selfUpdateReceiptSchema,
		SchemaVersion: 1,
		Status:        "divergent",
		Detail:        diagnostic,
		NextCommand:   "fak self-update --all",
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	_, err = codexFreshnessInstalledRevision(data, `C:\bin\fak.exe`)
	if err == nil {
		t.Fatal("divergent receipt should return an error refusing launch")
	}
	msg := err.Error()
	if strings.Contains(msg, "self-update failed") {
		t.Fatalf("error should not call installation failed: %q", msg)
	}
	if !strings.Contains(msg, "target updated successfully") {
		t.Fatalf("error missing target updated successfully: %q", msg)
	}
	if !strings.Contains(msg, "fak self-update --all") {
		t.Fatalf("error missing repair next command: %q", msg)
	}
}

func TestCodexFreshnessReexecHelperRoundTrip(t *testing.T) {
	const (
		selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		advanced = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	)
	if os.Getenv(codexFreshnessReexecHelperEnv) == "1" {
		codexFreshnessResolveCheckout = func() (string, string, error) { return `C:\work\fak`, os.Args[0], nil }
		codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
			return codexFreshnessInspection{Assessment: codexFreshnessAssessment{
				Verdict:       codexFreshnessBehind,
				RunningCommit: selected,
				TargetCommit:  advanced,
			}}
		}
		updated := false
		codexFreshnessUpdate = func(_, _ string) (string, error) {
			updated = true
			return "", errors.New("helper child unexpectedly tried another update")
		}
		codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(io.Discard, false) }
		args, code, stop := runCodexFreshnessAdmission([]string{"--model", "gpt-5"})
		if code != 0 || stop || updated || !reflect.DeepEqual(args, []string{"--model", "gpt-5"}) {
			t.Fatalf("child args=%q code=%d stop=%v updated=%v, want exact selected child admitted against advanced live ref", args, code, stop, updated)
		}
		if marker := os.Getenv(codexFreshnessReexecEnv); marker != "" {
			t.Fatalf("child retained consumed one-shot marker %q", marker)
		}
		return
	}

	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(codexFreshnessReexecHelperEnv, "1")
	argv := []string{executable, "-test.run=^TestCodexFreshnessReexecHelperRoundTrip$", "-test.count=1"}
	if err := runCodexFreshnessReexec(executable, argv, selected); err != nil {
		t.Fatalf("production reexec helper roundtrip: %v", err)
	}
}

func TestCodexFreshnessReexecEnvironmentScrubsInheritedMarkersCaseInsensitively(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	got := codexFreshnessReexecEnvironment([]string{
		"PATH=keep",
		"fak_codex_freshness_reexec=stale:1",
		"FAK_CODEX_FRESHNESS_REEXEC=duplicate:2",
	}, revision, 4242)
	want := []string{"PATH=keep", codexFreshnessReexecEnv + "=" + revision + ":4242"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reexec env=%q, want scrubbed single marker %q", got, want)
	}
}

func stringPtr(value string) *string { return &value }

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

func TestCodexFreshnessValidLeaseSkipsInspection(t *testing.T) {
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "0123456789abcdef0123456789abcdef01234567", TargetCommit: "0123456789abcdef0123456789abcdef01234567"})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	if err := codexFreshnessWriteReceipt(statePath+".json", codexFreshnessNow(), "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"); err != nil {
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
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "0123456789abcdef0123456789abcdef01234567", TargetCommit: "0123456789abcdef0123456789abcdef01234567"})
	defer restore()
	statePath := codexFreshnessTestStatePath(t)
	if err := codexFreshnessWriteReceipt(statePath+".json", codexFreshnessNow(), "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"); err != nil {
		t.Fatal(err)
	}
	called := false
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
		called = true
		return codexFreshnessInspection{Assessment: codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "0123456789abcdef0123456789abcdef01234567", TargetCommit: "0123456789abcdef0123456789abcdef01234567"}}
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
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "0123456789abcdef0123456789abcdef01234567", TargetCommit: "0123456789abcdef0123456789abcdef01234567"})
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
	restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: "0123456789abcdef0123456789abcdef01234567", TargetCommit: "0123456789abcdef0123456789abcdef01234567"})
	defer restore()
	statePath := codexFreshnessTestStatePath(t) + ".json"
	first := codexFreshnessNow().Add(-codexFreshnessLeaseTTL - time.Minute)
	if err := codexFreshnessWriteReceipt(statePath, first, "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"); err != nil {
		t.Fatal(err)
	}
	if err := codexFreshnessWriteReceipt(statePath, codexFreshnessNow(), "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"); err != nil {
		t.Fatalf("renew lease: %v", err)
	}
	if !codexFreshnessLeaseValid(statePath, codexFreshnessNow()) {
		t.Fatal("renewed lease is not valid")
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
func stubCodexFreshness(t *testing.T, assessment codexFreshnessAssessment) func() {
	t.Helper()
	oldExecutable, oldGetwd := codexFreshnessExecutable, codexFreshnessGetwd
	oldInspect, oldUpdate := codexFreshnessInspect, codexFreshnessUpdate
	oldReexec, oldStatus := codexFreshnessReexec, codexFreshnessStatus
	oldResolveContext, oldParentPID := codexFreshnessResolveCheckout, codexFreshnessParentPID
	oldNow, oldCacheDir := codexFreshnessNow, codexFreshnessCacheDir
	oldRunningCommit, oldUserConfigDir := codexFreshnessRunningCommit, codexFreshnessUserConfigDir
	cacheDir := t.TempDir()
	oldArgs := append([]string(nil), os.Args...)
	codexFreshnessExecutable = func() (string, error) { return `C:\bin\fak.exe`, nil }
	codexFreshnessGetwd = func() (string, error) { return `C:\work\fak`, nil }
	codexFreshnessResolveCheckout = func() (string, string, error) { return `C:\work\fak`, `C:\bin\fak.exe`, nil }
	codexFreshnessNow = func() time.Time { return time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC) }
	codexFreshnessCacheDir = func() (string, error) { return cacheDir, nil }
	codexFreshnessUserConfigDir = func() (string, error) { return filepath.Join(cacheDir, "no-config"), nil }
	codexFreshnessRunningCommit = func() string { return "0123456789abcdef0123456789abcdef01234567" }
	codexFreshnessInspect = func(_, _ string) codexFreshnessInspection { return codexFreshnessInspection{Assessment: assessment} }
	codexFreshnessUpdate = func(_, _ string) (string, error) { return "", errors.New("unexpected update") }
	codexFreshnessReexec = func(_ string, _ []string, _ string) error { return errors.New("unexpected reexec") }
	codexFreshnessStatus = func() *codexStartupStatus { return newCodexStartupStatus(io.Discard, false) }
	os.Args = []string{"fak", "codex"}
	t.Setenv(codexFreshnessReexecEnv, "")
	return func() {
		codexFreshnessExecutable, codexFreshnessGetwd = oldExecutable, oldGetwd
		codexFreshnessInspect, codexFreshnessUpdate = oldInspect, oldUpdate
		codexFreshnessReexec, codexFreshnessStatus = oldReexec, oldStatus
		codexFreshnessResolveCheckout = oldResolveContext
		codexFreshnessParentPID = oldParentPID
		codexFreshnessNow, codexFreshnessCacheDir = oldNow, oldCacheDir
		codexFreshnessRunningCommit, codexFreshnessUserConfigDir = oldRunningCommit, oldUserConfigDir
		os.Args = oldArgs
	}
}

func TestCodexFreshnessPolicyPrecedence(t *testing.T) {
	filtered, policy, err := parseCodexFreshnessSettings([]string{"--freshness-max-age", "15m", "--freshness-force", "--model", "x"}, "2h", "false", codexFreshnessConfig{MaxAge: "3h"})
	if err != nil {
		t.Fatal(err)
	}
	if policy.MaxAge != 15*time.Minute || !policy.Force || !reflect.DeepEqual(filtered, []string{"--model", "x"}) {
		t.Fatalf("filtered=%q policy=%+v", filtered, policy)
	}
	_, policy, err = parseCodexFreshnessSettings(nil, "45m", "", codexFreshnessConfig{MaxAge: "3h"})
	if err != nil || policy.MaxAge != 45*time.Minute || policy.Force {
		t.Fatalf("env policy=%+v err=%v", policy, err)
	}
	_, policy, err = parseCodexFreshnessSettings(nil, "", "", codexFreshnessConfig{})
	if err != nil || policy.MaxAge != codexFreshnessLeaseTTL {
		t.Fatalf("default policy=%+v err=%v", policy, err)
	}
}

func TestCodexFreshnessReceiptRejectsInvalidStates(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	running := "0123456789abcdef0123456789abcdef01234567"
	other := "fedcba9876543210fedcba9876543210fedcba98"
	path := filepath.Join(t.TempDir(), "freshness.json")
	cases := []struct {
		name    string
		payload []byte
		maxAge  time.Duration
	}{
		{name: "missing", maxAge: time.Hour},
		{name: "corrupt", payload: []byte("{oops"), maxAge: time.Hour},
		{name: "expired", payload: freshnessReceiptJSON(t, now.Add(-2*time.Hour), running, running), maxAge: time.Hour},
		{name: "future", payload: freshnessReceiptJSON(t, now.Add(time.Second), running, running), maxAge: time.Hour},
		{name: "running mismatch", payload: freshnessReceiptJSON(t, now, other, other), maxAge: time.Hour},
		{name: "target mismatch", payload: freshnessReceiptJSON(t, now, running, other), maxAge: time.Hour},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_ = os.Remove(path)
			if tc.payload != nil {
				if err := os.WriteFile(path, tc.payload, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if codexFreshnessLeaseValidFor(path, now, tc.maxAge, running) {
				t.Fatal("invalid receipt accepted")
			}
		})
	}
}

func TestCodexFreshnessReceiptPathsCountInspections(t *testing.T) {
	running := "0123456789abcdef0123456789abcdef01234567"
	for _, tc := range []struct {
		name            string
		age             time.Duration
		args            []string
		wantInspections int
	}{
		{name: "fresh local hit", age: time.Minute, wantInspections: 0},
		{name: "expired", age: 2 * time.Hour, args: []string{"--freshness-max-age", "1h"}, wantInspections: 1},
		{name: "forced", age: time.Minute, args: []string{"--freshness-force"}, wantInspections: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restore := stubCodexFreshness(t, codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: running, TargetCommit: running})
			defer restore()
			statePath := codexFreshnessTestStatePath(t) + ".json"
			if err := codexFreshnessWriteReceipt(statePath, codexFreshnessNow().Add(-tc.age), running, running); err != nil {
				t.Fatal(err)
			}
			inspections := 0
			codexFreshnessInspect = func(_, _ string) codexFreshnessInspection {
				inspections++
				return codexFreshnessInspection{Assessment: codexFreshnessAssessment{Verdict: codexFreshnessFresh, RunningCommit: running, TargetCommit: running}}
			}
			args := append(tc.args, "--model", "x")
			started := time.Now()
			filtered, code, stop := runCodexFreshnessAdmission(args)
			if stop || code != 0 || inspections != tc.wantInspections || !reflect.DeepEqual(filtered, []string{"--model", "x"}) {
				t.Fatalf("filtered=%q code=%d stop=%v inspections=%d elapsed=%s", filtered, code, stop, inspections, time.Since(started))
			}
		})
	}
}

func freshnessReceiptJSON(t *testing.T, checkedAt time.Time, running, target string) []byte {
	t.Helper()
	raw, err := json.Marshal(codexFreshnessLease{Schema: codexFreshnessReceiptSchema, CheckedAt: checkedAt, RunningCommit: running, TargetCommit: target})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestCodexFreshnessConfigEnvCLIForcePrecedence(t *testing.T) {
	configForce := true
	filtered, policy, err := parseCodexFreshnessSettings(
		[]string{"--freshness-max-age=10m", "--model", "x"},
		"20m", "false", codexFreshnessConfig{MaxAge: "30m", Force: &configForce},
	)
	if err != nil {
		t.Fatal(err)
	}
	if policy.MaxAge != 10*time.Minute || policy.Force || !reflect.DeepEqual(filtered, []string{"--model", "x"}) {
		t.Fatalf("CLI/env/config precedence: filtered=%q policy=%+v", filtered, policy)
	}
	_, policy, err = parseCodexFreshnessSettings(nil, "", "", codexFreshnessConfig{MaxAge: "30m", Force: &configForce})
	if err != nil || policy.MaxAge != 30*time.Minute || !policy.Force {
		t.Fatalf("config policy=%+v err=%v", policy, err)
	}
}

func TestLoadCodexFreshnessConfig(t *testing.T) {
	old := codexFreshnessUserConfigDir
	dir := t.TempDir()
	codexFreshnessUserConfigDir = func() (string, error) { return dir, nil }
	t.Cleanup(func() { codexFreshnessUserConfigDir = old })
	path := filepath.Join(dir, "fak", "codex-freshness.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"max_age":"25m","force":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := loadCodexFreshnessConfig()
	if err != nil || got.MaxAge != "25m" || got.Force == nil || !*got.Force {
		t.Fatalf("config=%+v err=%v", got, err)
	}
}
