package tempartifact

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPreviewSelectsOnlyOldUnreferencedAllowedFiles(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 21, 12, 0, 0, 0, time.UTC)
	eligible := makeArtifact(t, root, "fak-eligible.zip", "zip", now.Add(-7*time.Hour))
	prefix := makeArtifact(t, root, "fak-prefix.tar", "tar", now.Add(-8*time.Hour))
	fresh := makeArtifact(t, root, "fak-fresh.exe", "fresh", now.Add(-time.Hour))
	active := makeArtifact(t, root, "fak-active.exe", "active", now.Add(-9*time.Hour))
	rejected := makeArtifact(t, root, "fak-rejected.txt", "text", now.Add(-10*time.Hour))
	unrelated := makeArtifact(t, root, "other.zip", "other", now.Add(-10*time.Hour))
	if err := os.Mkdir(filepath.Join(root, "fak-directory.zip"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := makeArtifact(t, root, "target.bin", "target", now.Add(-10*time.Hour))
	link := filepath.Join(root, "fak-link.zip")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}

	report, err := Run(context.Background(), Config{
		Root:   root,
		MinAge: 6 * time.Hour,
		Now:    func() time.Time { return now },
		Inspect: func(_ context.Context, paths []string) Inspection {
			references := map[string]bool{pathKey(active): true}
			prefixOnly := referencesFromProcessRecords([]processRecord{{
				CommandLine: `runner.exe "` + prefix + `.suffix"`,
			}}, paths)
			for path := range prefixOnly {
				references[path] = true
			}
			return Inspection{Complete: true, References: references}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Mode != "preview" || report.Schema != Schema {
		t.Fatalf("unexpected report header: %+v", report)
	}
	assertItemReason(t, report, eligible, ReasonEligible)
	assertItemReason(t, report, prefix, ReasonEligible)
	assertItemReason(t, report, fresh, ReasonFresh)
	assertItemReason(t, report, active, ReasonActiveReference)
	assertItemReason(t, report, link, ReasonReparsePoint)
	if findItem(report, rejected) != nil || findItem(report, unrelated) != nil {
		t.Fatalf("rejected or unrelated file entered report: %+v", report.Items)
	}
	if report.Summary.EligibleCount != 2 || report.Summary.ReapedCount != 0 {
		t.Fatalf("summary = %+v, want two eligible and none reaped", report.Summary)
	}
	for _, path := range []string{eligible, prefix, fresh, active, rejected, unrelated, link} {
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("preview changed %q: %v", path, err)
		}
	}
}

func TestApplyReapsOnlyTheExactEligibleFile(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	old := makeArtifact(t, root, "fak-old.zip", "old", now.Add(-7*time.Hour))
	fresh := makeArtifact(t, root, "fak-fresh.zip", "fresh", now.Add(-time.Hour))
	unrelated := makeArtifact(t, root, "keep.zip", "keep", now.Add(-9*time.Hour))

	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: completeInspection,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertItemReason(t, report, old, ReasonReaped)
	if report.Summary.EligibleCount != 1 || report.Summary.ReapedBytes != int64(len("old")) {
		t.Fatalf("summary = %+v", report.Summary)
	}
	if _, err := os.Stat(old); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("old artifact still exists: %v", err)
	}
	for _, path := range []string{fresh, unrelated} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("sibling %q was changed: %v", path, err)
		}
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "fak-maintenance-quarantine-") {
			t.Fatalf("empty quarantine residue remains: %s", entry.Name())
		}
	}
}

func TestApplyRechecksReferenceBeforeMove(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	path := makeArtifact(t, root, "fak-active-late.exe", "active", now.Add(-7*time.Hour))
	referenced := false
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		BeforeMove: func(string) { referenced = true },
		Inspect: func(_ context.Context, _ []string) Inspection {
			refs := map[string]bool{}
			if referenced {
				refs[pathKey(path)] = true
			}
			return Inspection{Complete: true, References: refs}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertItemReason(t, report, path, ReasonActiveReference)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("newly referenced artifact was not preserved: %v", err)
	}
}

func TestApplyPreservesChangedSinceScan(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	path := makeArtifact(t, root, "fak-changing.tar", "before", now.Add(-7*time.Hour))
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: completeInspection,
		BeforeMove: func(path string) {
			if err := os.WriteFile(path, []byte("changed-after-scan"), 0o600); err != nil {
				t.Fatal(err)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertItemReason(t, report, path, ReasonChangedSinceScan)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("changed artifact was not preserved: %v", err)
	}
}

func TestApplyPreservesPostMoveReferenceInQuarantine(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	source := makeArtifact(t, root, "fak-post-move.zip", "payload", now.Add(-7*time.Hour))
	var referenced string
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		AfterMove: func(_, destination string) { referenced = destination },
		Inspect: func(_ context.Context, _ []string) Inspection {
			refs := map[string]bool{}
			if referenced != "" {
				refs[pathKey(referenced)] = true
			}
			return Inspection{Complete: true, References: refs}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	item := assertItemReason(t, report, source, ReasonPostMoveReference)
	if item.QuarantinePath == "" {
		t.Fatal("post-move reference did not report its quarantine path")
	}
	if _, err := os.Stat(item.QuarantinePath); err != nil {
		t.Fatalf("referenced quarantine artifact was not preserved: %v", err)
	}
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("source still exists after quarantine move: %v", err)
	}
}

func TestApplyPostMoveIdentityChangeIsTypedAndPreserved(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	source := makeArtifact(t, root, "fak-post-move-change.tar", "before", now.Add(-7*time.Hour))
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: completeInspection,
		AfterMove: func(_, destination string) {
			if err := os.WriteFile(destination, []byte("changed-after-move"), 0o600); err != nil {
				t.Fatal(err)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	item := assertItemReason(t, report, source, ReasonPostMoveRecheckFailed)
	if item.QuarantinePath == "" {
		t.Fatal("changed post-move artifact has no quarantine path")
	}
	if _, err := os.Stat(item.QuarantinePath); err != nil {
		t.Fatalf("changed post-move artifact was not preserved: %v", err)
	}
}

func TestApplyPostMoveInspectionFailurePreservesQuarantine(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	source := makeArtifact(t, root, "fak-post-move-uninspected.zip", "payload", now.Add(-7*time.Hour))
	calls := 0
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: func(context.Context, []string) Inspection {
			calls++
			if calls == 3 {
				return Inspection{Reason: ReasonInspectionUnavailable, References: map[string]bool{}}
			}
			return Inspection{Complete: true, References: map[string]bool{}}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	item := assertItemReason(t, report, source, ReasonPostMoveInspectUnavailable)
	if _, err := os.Stat(item.QuarantinePath); err != nil {
		t.Fatalf("uninspected quarantine artifact was not preserved: %v", err)
	}
}

func TestApplyMoveFailureIsTypedAndPreservesSource(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	path := makeArtifact(t, root, "fak-move-fails.exe", "payload", now.Add(-7*time.Hour))
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: completeInspection,
		Rename:  func(string, string) error { return errors.New("move refused") },
	})
	if err != nil {
		t.Fatal(err)
	}
	assertItemReason(t, report, path, ReasonMoveFailed)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("failed move did not preserve source: %v", err)
	}
}

func TestApplyDeleteFailureIsTypedAndPreservesQuarantine(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	source := makeArtifact(t, root, "fak-delete-fails.exe", "payload", now.Add(-7*time.Hour))
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: completeInspection,
		Remove:  func(string) error { return errors.New("delete refused") },
	})
	if err != nil {
		t.Fatal(err)
	}
	item := assertItemReason(t, report, source, ReasonDeleteFailed)
	if _, err := os.Stat(item.QuarantinePath); err != nil {
		t.Fatalf("delete failure did not preserve quarantine artifact: %v", err)
	}
}

func TestInspectionFailurePreservesCandidate(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	path := makeArtifact(t, root, "fak-uninspected.zip", "payload", now.Add(-7*time.Hour))
	report, err := Run(context.Background(), Config{
		Root: root, MinAge: 6 * time.Hour, Now: func() time.Time { return now }, Apply: true,
		Inspect: func(context.Context, []string) Inspection {
			return Inspection{Reason: ReasonInspectionUnavailable, References: map[string]bool{}}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertItemReason(t, report, path, ReasonInspectionUnavailable)
	if report.Inspection != ReasonInspectionUnavailable {
		t.Fatalf("inspection = %q", report.Inspection)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("uninspected artifact was not preserved: %v", err)
	}
}

func TestExactProcessPathDoesNotMatchPrefixCollision(t *testing.T) {
	candidate := `C:\Temp\fak-old.exe`
	records := []processRecord{
		{ExecutablePath: candidate + `.bak`},
		{CommandLine: `runner.exe "` + candidate + `.suffix"`},
	}
	if refs := referencesFromProcessRecords(records, []string{candidate}); refs[pathKey(candidate)] {
		t.Fatalf("prefix collision matched exact candidate: %+v", refs)
	}
	records = append(records, processRecord{CommandLine: `runner.exe --artifact="` + candidate + `"`})
	if refs := referencesFromProcessRecords(records, []string{candidate}); !refs[pathKey(candidate)] {
		t.Fatalf("exact --flag path was not matched: %+v", refs)
	}
}

func completeInspection(_ context.Context, _ []string) Inspection {
	return Inspection{Complete: true, References: map[string]bool{}}
}

func makeArtifact(t *testing.T, root, name, contents string, modTime time.Time) string {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, modTime, modTime); err != nil {
		t.Fatal(err)
	}
	return filepath.Clean(path)
}

func findItem(report Report, path string) *Item {
	for index := range report.Items {
		if report.Items[index].Path == path {
			return &report.Items[index]
		}
	}
	return nil
}

func assertItemReason(t *testing.T, report Report, path, reason string) *Item {
	t.Helper()
	item := findItem(report, path)
	if item == nil {
		t.Fatalf("report has no item for %q: %+v", path, report.Items)
	}
	if item.Reason != reason {
		t.Fatalf("reason for %q = %q, want %q", path, item.Reason, reason)
	}
	return item
}
