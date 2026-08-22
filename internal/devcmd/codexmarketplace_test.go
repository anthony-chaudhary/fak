package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSweepCodexMarketplaceReclaimsTerminalOwnedCandidate(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := filepath.Join(home, ".tmp", "marketplaces", ".staging", "marketplace-upgrade-owned")
	if err := os.MkdirAll(filepath.Join(candidate, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(candidate, "payload"), []byte("clone"), 0o644); err != nil {
		t.Fatal(err)
	}
	receipt := codexMarketplaceAttemptReceipt{
		Schema:        codexMarketplaceAttemptSchema,
		AttemptID:     "owned",
		Marketplace:   "dos",
		CandidatePath: candidate,
		Status:        codexMarketplaceAttemptFailed,
		StartedAt:     now.Add(-2 * time.Hour),
		FinishedAt:    now.Add(-time.Hour),
	}
	if err := writeCodexMarketplaceAttemptReceipt(candidate, receipt); err != nil {
		t.Fatal(err)
	}
	setTreeModTime(t, candidate, now.Add(-time.Hour))

	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home:           home,
		Now:            now,
		Grace:          20 * time.Minute,
		QuarantineRoot: filepath.Join(home, "quarantine"),
		Processes: func() ([]codexMarketplaceProcess, error) {
			return nil, nil
		},
	}, true)
	if report.Err != "" {
		t.Fatalf("sweep error: %s", report.Err)
	}
	if report.ReclaimedClones != 1 || report.ReclaimedFiles < 2 {
		t.Fatalf("reclaimed clones/files = %d/%d, want 1/at least 2: %+v", report.ReclaimedClones, report.ReclaimedFiles, report)
	}
	if _, err := os.Stat(candidate); !os.IsNotExist(err) {
		t.Fatalf("candidate still exists after reclaim: %v", err)
	}
	second := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		QuarantineRoot: filepath.Join(home, "quarantine"),
		Processes:      func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	if second.ReclaimedClones != 0 || second.RetainedClones != 0 {
		t.Fatalf("second sweep was not idempotent: %+v", second)
	}
}

func TestSweepCodexMarketplaceKeepsMissingReceiptNestedClone(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := filepath.Join(home, ".tmp", "marketplaces", ".staging", "marketplace-upgrade-unowned")
	if err := os.MkdirAll(filepath.Join(candidate, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(candidate, ".git", "config"), []byte("nested repository"), 0o644); err != nil {
		t.Fatal(err)
	}
	setTreeModTime(t, candidate, now.Add(-48*time.Hour))
	processCalls := 0
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) {
			processCalls++
			return nil, nil
		},
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonMissingReceipt)
	if !pathExists(candidate) || report.ReclaimedClones != 0 || processCalls != 0 {
		t.Fatalf("unowned nested clone was touched: %+v", report)
	}
}

func TestSweepCodexMarketplaceKeepsActiveReceiptOwnedClone(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := writeTerminalMarketplaceCandidate(t, home, "active", now, "")
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) {
			return []codexMarketplaceProcess{{PID: 42, CommandLine: `git -C "` + candidate + `" fetch`, ExecutablePath: filepath.Join(candidate, "bin", "helper")}}, nil
		},
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonActive)
	if !pathExists(candidate) {
		t.Fatal("active candidate was removed")
	}
}

func TestSweepCodexMarketplaceKeepsCandidateWhenProcessCensusFails(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := writeTerminalMarketplaceCandidate(t, home, "census-failed", now, "")
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, errors.New("census unavailable") },
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonProcessEnumerationFailed)
	if !pathExists(candidate) {
		t.Fatal("candidate was removed without a process census")
	}
}

func TestSweepCodexMarketplaceKeepsReparseCandidate(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := writeTerminalMarketplaceCandidate(t, home, "reparse", now, "")
	outside := filepath.Join(home, "outside")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(candidate, "escape")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	setTreeModTime(t, candidate, now.Add(-time.Hour))
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonReparse)
	if !pathExists(candidate) {
		t.Fatal("reparse candidate was removed")
	}
}

func TestSweepCodexMarketplaceKeepsContainmentEscapeReceipt(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	outside := filepath.Join(home, "outside-clone")
	candidate := writeTerminalMarketplaceCandidate(t, home, "escape", now, outside)
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonReceiptMismatch)
	if !pathExists(candidate) {
		t.Fatal("receipt path escape candidate was removed")
	}
}

func TestSweepCodexMarketplaceKeepsContainmentEscapeRoot(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	outsideHome := t.TempDir()
	outsideCandidate := writeTerminalMarketplaceCandidate(t, outsideHome, "outside", now, "")
	staging := filepath.Join(home, ".tmp", "marketplaces", ".staging")
	if err := os.MkdirAll(filepath.Dir(staging), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Dir(outsideCandidate), staging); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute,
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	if report.Err == "" || !strings.Contains(report.Err, "reparse point") {
		t.Fatalf("containment escape root was not refused: %+v", report)
	}
	if !pathExists(outsideCandidate) {
		t.Fatal("candidate reachable only through escaped staging root was removed")
	}
}

func TestSweepCodexMarketplaceRollsBackWhenQuarantineBecomesReferenced(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	quarantine := filepath.Join(home, "quarantine")
	candidate := writeTerminalMarketplaceCandidate(t, home, "recheck", now, "")
	calls := 0
	report := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute, QuarantineRoot: quarantine,
		Processes: func() ([]codexMarketplaceProcess, error) {
			calls++
			if calls < 3 {
				return nil, nil
			}
			matches, _ := filepath.Glob(filepath.Join(quarantine, "run-*", filepath.Base(candidate)))
			if len(matches) != 1 {
				return nil, errors.New("quarantine destination not observable")
			}
			return []codexMarketplaceProcess{{PID: 77, ExecutablePath: filepath.Join(matches[0], "bin", "helper")}}, nil
		},
	}, true)
	assertMarketplaceEntryReason(t, report, candidate, codexMarketplaceReasonPostMoveActive)
	if !pathExists(candidate) {
		t.Fatal("post-move referenced candidate was not restored")
	}
}

func TestSweepCodexMarketplaceInterruptedCleanupRecoversOnNextRun(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	candidate := writeTerminalMarketplaceCandidate(t, home, "interrupted", now, "")
	blockedQuarantine := filepath.Join(home, "not-a-directory")
	if err := os.WriteFile(blockedQuarantine, []byte("blocked"), 0o644); err != nil {
		t.Fatal(err)
	}
	blocked := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute, QuarantineRoot: blockedQuarantine,
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	assertMarketplaceEntryReason(t, blocked, candidate, codexMarketplaceReasonQuarantined)
	if !pathExists(candidate) {
		t.Fatal("failed cleanup lost the candidate instead of retaining it")
	}

	recovered := sweepCodexMarketplace(codexMarketplaceOptions{
		Home: home, Now: now, Grace: 20 * time.Minute, QuarantineRoot: filepath.Join(home, "quarantine"),
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, true)
	if recovered.ReclaimedClones != 1 || pathExists(candidate) {
		t.Fatalf("next recovery did not reclaim the exact retained candidate: %+v", recovered)
	}
}

func TestUpgradeCodexMarketplacePromotesVerifiedRevisionAndRemovesStage(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "codex-home")
	source := filepath.Join(root, "source")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	runGitTest(t, source, "init", "--initial-branch=main")
	runGitTest(t, source, "config", "user.email", "fixture@example.invalid")
	runGitTest(t, source, "config", "user.name", "fixture")
	writeMarketplacePayload(t, source, "v1")
	runGitTest(t, source, "add", "payload.txt")
	runGitTest(t, source, "commit", "-m", "v1")
	firstRevision := runGitTest(t, source, "rev-parse", "HEAD")

	destination := filepath.Join(home, ".tmp", "marketplaces", "dos")
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		t.Fatal(err)
	}
	runCommandTest(t, "git", "clone", source, destination)
	install := codexMarketplaceInstallReceipt{SourceType: "git", Source: source, RefName: "main", Revision: firstRevision}
	if err := writeJSONAtomic(filepath.Join(destination, ".codex-marketplace-install.json"), install); err != nil {
		t.Fatal(err)
	}
	writeMarketplacePayload(t, source, "v2")
	runGitTest(t, source, "add", "payload.txt")
	runGitTest(t, source, "commit", "-m", "v2")
	wantRevision := runGitTest(t, source, "rev-parse", "HEAD")

	attempt, err := upgradeCodexMarketplace(codexMarketplaceOptions{
		Home: home, QuarantineRoot: filepath.Join(root, "quarantine"),
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
	}, "dos")
	if err != nil {
		t.Fatal(err)
	}
	if attempt.Status != codexMarketplaceAttemptPromoted || attempt.Revision != wantRevision {
		t.Fatalf("attempt = %+v, want promoted %s", attempt, wantRevision)
	}
	if pathExists(attempt.CandidatePath) {
		t.Fatalf("successful promotion retained exact staging clone %s", attempt.CandidatePath)
	}
	if got := strings.TrimSpace(string(mustReadFile(t, filepath.Join(destination, "payload.txt")))); got != "v2" {
		t.Fatalf("promoted payload = %q, want v2", got)
	}
	installed := runGitTest(t, destination, "rev-parse", "HEAD")
	if installed != wantRevision {
		t.Fatalf("promoted HEAD = %s, want %s", installed, wantRevision)
	}
	terminal, readErr := readCodexMarketplaceAttemptReceipt(filepath.Join(home, ".tmp", "marketplaces", ".fak-upgrade-receipts", attempt.AttemptID+".json"))
	if readErr != nil || terminal.Status != codexMarketplaceAttemptPromoted {
		t.Fatalf("terminal receipt = %+v, err=%v", terminal, readErr)
	}
}

func TestUpgradeCodexMarketplaceFailureKeepsBoundedReceiptNotClone(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "codex-home")
	destination := filepath.Join(home, ".tmp", "marketplaces", "dos")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		t.Fatal(err)
	}
	install := codexMarketplaceInstallReceipt{SourceType: "git", Source: "fixture-source", RefName: "main", Revision: "old"}
	if err := writeJSONAtomic(filepath.Join(destination, ".codex-marketplace-install.json"), install); err != nil {
		t.Fatal(err)
	}
	longDetail := strings.Repeat("diagnostic", 1000)
	attempt, err := upgradeCodexMarketplace(codexMarketplaceOptions{
		Home: home, QuarantineRoot: filepath.Join(root, "quarantine"),
		Processes: func() ([]codexMarketplaceProcess, error) { return nil, nil },
		RunGit:    func(string, ...string) (string, error) { return "", errors.New(longDetail) },
	}, "dos")
	if err == nil || attempt.Status != codexMarketplaceAttemptFailed {
		t.Fatalf("attempt=%+v err=%v, want bounded failure", attempt, err)
	}
	if len(attempt.Detail) != codexMarketplaceDetailLimit || attempt.DiagnosticBytes != codexMarketplaceDetailLimit {
		t.Fatalf("diagnostic bytes=%d len=%d, want %d", attempt.DiagnosticBytes, len(attempt.Detail), codexMarketplaceDetailLimit)
	}
	if pathExists(attempt.CandidatePath) {
		t.Fatalf("failed attempt retained full clone at %s", attempt.CandidatePath)
	}
	terminal, readErr := readCodexMarketplaceAttemptReceipt(filepath.Join(home, ".tmp", "marketplaces", ".fak-upgrade-receipts", attempt.AttemptID+".json"))
	if readErr != nil || terminal.Status != codexMarketplaceAttemptFailed || len(terminal.Detail) != codexMarketplaceDetailLimit {
		t.Fatalf("terminal receipt=%+v err=%v", terminal, readErr)
	}
}

func TestCodexMarketplaceReportHumanAndJSONAccounting(t *testing.T) {
	report := codexMarketplaceReport{Schema: codexMarketplaceReportSchema, DryRun: true, Entries: []codexMarketplaceEntry{
		{Reason: codexMarketplaceReasonMissingReceipt, Files: 5, Bytes: 50},
		{Reason: codexMarketplaceReasonEligible, Files: 3, Bytes: 30},
		{Reason: codexMarketplaceReasonReclaimed, Files: 2, Bytes: 20},
	}}
	foldCodexMarketplaceReport(&report)
	var human bytes.Buffer
	writeCodexMarketplaceReport(&human, report, false)
	for _, want := range []string{"retained 2 clone(s)", "reclaimed 1", "retained 8 file(s)/80 byte(s)", "missing-terminal-receipt"} {
		if !strings.Contains(human.String(), want) {
			t.Fatalf("human report missing %q:\n%s", want, human.String())
		}
	}
	var machine bytes.Buffer
	writeCodexMarketplaceReport(&machine, report, true)
	var decoded codexMarketplaceReport
	if err := json.Unmarshal(machine.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.RetainedClones != 2 || decoded.ReclaimedClones != 1 || decoded.SkipReasons[codexMarketplaceReasonMissingReceipt] != 1 {
		t.Fatalf("JSON accounting drifted: %+v", decoded)
	}
}

func TestCodexMarketplaceReferenceMatchUsesPathBoundaries(t *testing.T) {
	candidate := filepath.Join(string(filepath.Separator)+"tmp", "marketplace-upgrade-12")
	if codexMarketplaceFieldReferencesPath("worker "+candidate+"3", candidate) {
		t.Fatal("prefix collision counted as an exact path reference")
	}
	if !codexMarketplaceFieldReferencesPath("worker --cwd="+candidate+string(filepath.Separator)+"repo", candidate) {
		t.Fatal("exact child path was not counted as a reference")
	}
	pids := codexMarketplaceReferencingPIDs([]codexMarketplaceProcess{{PID: 9, ExecutablePath: filepath.Join(candidate, "bin", "helper")}}, candidate)
	if len(pids) != 1 || pids[0] != 9 {
		t.Fatalf("executable-path reference PIDs = %v, want [9]", pids)
	}
}

func writeTerminalMarketplaceCandidate(t *testing.T, home, suffix string, now time.Time, receiptPath string) string {
	t.Helper()
	candidate := filepath.Join(home, ".tmp", "marketplaces", ".staging", codexMarketplaceStagePrefix+suffix)
	if err := os.MkdirAll(filepath.Join(candidate, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(candidate, ".git", "config"), []byte("nested repository"), 0o644); err != nil {
		t.Fatal(err)
	}
	if receiptPath == "" {
		receiptPath = candidate
	}
	receipt := codexMarketplaceAttemptReceipt{
		Schema: codexMarketplaceAttemptSchema, AttemptID: codexMarketplaceStagePrefix + suffix,
		Marketplace: "dos", CandidatePath: receiptPath, Status: codexMarketplaceAttemptFailed,
		StartedAt: now.Add(-2 * time.Hour), FinishedAt: now.Add(-time.Hour),
	}
	if err := writeCodexMarketplaceAttemptReceipt(candidate, receipt); err != nil {
		t.Fatal(err)
	}
	setTreeModTime(t, candidate, now.Add(-time.Hour))
	return candidate
}

func assertMarketplaceEntryReason(t *testing.T, report codexMarketplaceReport, candidate, want string) {
	t.Helper()
	for _, entry := range report.Entries {
		if sameCanonicalPath(entry.Path, candidate) {
			if entry.Reason != want {
				t.Fatalf("candidate reason = %q, want %q: %+v", entry.Reason, want, report)
			}
			return
		}
	}
	t.Fatalf("candidate %s absent from report: %+v", candidate, report)
}

func writeMarketplacePayload(t *testing.T, root, value string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, "payload.txt"), []byte(value+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func runGitTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	commandArgs := append([]string{"-C", dir}, args...)
	return runCommandTest(t, "git", commandArgs...)
}

func runCommandTest(t *testing.T, name string, args ...string) string {
	t.Helper()
	cmd := exec.Command(name, args...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %v: %v\n%s", name, args, err, b)
	}
	return strings.TrimSpace(string(b))
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func setTreeModTime(t *testing.T, root string, when time.Time) {
	t.Helper()
	err := filepath.Walk(root, func(path string, _ os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		return os.Chtimes(path, when, when)
	})
	if err != nil {
		t.Fatal(err)
	}
}
