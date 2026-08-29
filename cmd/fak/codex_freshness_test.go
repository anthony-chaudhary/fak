package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
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

func TestCodexFreshnessSelfUpdateUsesSupportedFlags(t *testing.T) {
	want := []string{"self-update", "--json", "--root", `C:\work\fak`, "--target", `C:\bin\fak.exe`}
	if got := codexFreshnessSelfUpdateArgs(`C:\work\fak`, `C:\bin\fak.exe`); !reflect.DeepEqual(got, want) {
		t.Fatalf("self-update args=%q, want %q", got, want)
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

func stubCodexFreshness(t *testing.T, assessment codexFreshnessAssessment) func() {
	t.Helper()
	oldExecutable, oldGetwd := codexFreshnessExecutable, codexFreshnessGetwd
	oldInspect, oldUpdate := codexFreshnessInspect, codexFreshnessUpdate
	oldReexec, oldStatus := codexFreshnessReexec, codexFreshnessStatus
	oldResolveContext, oldParentPID := codexFreshnessResolveCheckout, codexFreshnessParentPID
	oldArgs := append([]string(nil), os.Args...)
	codexFreshnessExecutable = func() (string, error) { return `C:\bin\fak.exe`, nil }
	codexFreshnessGetwd = func() (string, error) { return `C:\work\fak`, nil }
	codexFreshnessResolveCheckout = func() (string, string, error) { return `C:\work\fak`, `C:\bin\fak.exe`, nil }
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
		os.Args = oldArgs
	}
}
