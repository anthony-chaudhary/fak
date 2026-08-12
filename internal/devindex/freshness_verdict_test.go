package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// TestFreshnessReportUnknownWhenADetectorCannotRun is the #5962 witness for the
// self-index probe: a detector whose source could not be read must move the verdict to
// unknown, never leave it at fresh.
//
// The tree below is deliberately clean -- every leaf declared, every doc link resolving
// -- so the drift-only fold is empty. Before this change that empty slice WAS the answer
// ("an empty slice means the index agrees with reality"), even though the verb detector
// never read cmd/fak/main.go and the llms.txt detector never read llms.txt. The report
// keeps the drift list empty (an unread source is not evidence of drift) and says
// unknown instead.
func TestFreshnessReportUnknownWhenADetectorCannotRun(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "g.go", "package gateway\n")
	mustWrite(t, root, "README.md", "# readme\n")
	mustWrite(t, root, "INDEX.md", "# INDEX\n- [Readme](README.md) — fine.\n")
	// No cmd/fak/main.go and no llms.txt: two detectors have nothing to read.
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	rep := c.CheckFreshnessReport()
	if len(rep.Drifts) != 0 {
		t.Fatalf("an unread source is not drift, got %+v", rep.Drifts)
	}
	if rep.Verdict() != VerdictUnknown {
		t.Fatalf("verdict = %q, want %q: two detectors never ran", rep.Verdict(), VerdictUnknown)
	}
	if rep.Fresh() {
		t.Fatal("a report with an unchecked detector must not read as fresh")
	}
	got := map[DriftKind]string{}
	for _, u := range rep.Unchecked {
		if u.Reason == "" {
			t.Errorf("unchecked %+v carries no reason", u)
		}
		got[u.Detector] = u.Source
	}
	for detector, source := range map[DriftKind]string{
		DriftUnknownVerb:  "cmd/fak/main.go",
		DriftDeadLLMSLink: "llms.txt",
	} {
		if got[detector] != source {
			t.Errorf("detector %q unchecked source = %q, want %q", detector, got[detector], source)
		}
	}
	// The drift-only projection stays byte-compatible for the callers that only count
	// proven drift, which is exactly why it cannot be the surface a verdict comes from.
	if drift := c.CheckFreshness(); len(drift) != 0 {
		t.Errorf("CheckFreshness = %+v, want the unchanged drift-only view", drift)
	}
}

// TestFreshnessReportFreshOnlyWhenEveryDetectorRan: the fresh verdict is still
// reachable, and it requires every source to have been read.
func TestFreshnessReportFreshOnlyWhenEveryDetectorRan(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\ncmd = [\"cmd/**\"]\n")
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "g.go", "package gateway\n")
	mustWrite(t, root, "README.md", "# readme\n")
	mustWrite(t, root, "INDEX.md", "# INDEX\n- [Readme](README.md) — fine.\n")
	mustWrite(t, root, "llms.txt", "# llms\nSee [Readme](README.md).\n")
	mustMkdir(t, root, "docs", "notes")
	mustMkdir(t, root, "cmd", "fak")
	mustWrite(t, filepath.Join(root, "cmd", "fak"), "main.go",
		"package main\n\nimport \"os\"\n\nfunc main() {\n\tswitch os.Args[1] {\n\tcase \"index\":\n\t\tcmdIndex()\n\tdefault:\n\t\tusage()\n\t}\n}\n")
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	rep := c.CheckFreshnessReport()
	if !rep.Fresh() || rep.Verdict() != VerdictFresh {
		t.Fatalf("verdict = %q with unchecked %+v, want fresh: every source is readable", rep.Verdict(), rep.Unchecked)
	}

	// Drift outranks unknown: an undeclared leaf beside an unreadable source is still
	// the actionable answer, because it was actually proven.
	mustMkdir(t, root, "internal", "orphan")
	mustWrite(t, filepath.Join(root, "internal", "orphan"), "orphan.go", "package orphan\n")
	if err := os.Remove(filepath.Join(root, "llms.txt")); err != nil {
		t.Fatal(err)
	}
	c2, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rep2 := c2.CheckFreshnessReport()
	if rep2.Verdict() != VerdictStale {
		t.Errorf("verdict = %q, want %q: proven drift outranks an unchecked detector", rep2.Verdict(), VerdictStale)
	}
	if len(rep2.Unchecked) == 0 {
		t.Error("the stale verdict must still carry the detector that never ran")
	}
}

// TestFreshnessReportUnknownWhenTheDocMapCouldNotBeRead covers the seam Load hides: a
// missing INDEX.md degrades to an EMPTY doc map, so DeadDocLinks returns nothing for a
// map it never had. Both detectors that depend on those bytes must report unchecked.
func TestFreshnessReportUnknownWhenTheDocMapCouldNotBeRead(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "g.go", "package gateway\n")
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rep := c.CheckFreshnessReport()
	if rep.Fresh() {
		t.Fatal("a tree with no INDEX.md was never link-checked and must not read as fresh")
	}
	seen := map[DriftKind]bool{}
	for _, u := range rep.Unchecked {
		if u.Source == "INDEX.md" {
			seen[u.Detector] = true
		}
	}
	for _, detector := range []DriftKind{DriftDeadDocLink, DriftOrphanNote} {
		if !seen[detector] {
			t.Errorf("detector %q did not report INDEX.md unchecked; unchecked = %+v", detector, rep.Unchecked)
		}
	}
}
