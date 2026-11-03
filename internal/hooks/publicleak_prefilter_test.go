package hooks

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestPublicLeakTreeTimingRecordsPhases(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init", "-q")
	writeFile(t, root, "clean.txt", "ordinary source\n")
	gitFixture(t, root, "add", "clean.txt")
	got, err := AuditPublicLeakTree(root)
	if err != nil {
		t.Fatal(err)
	}
	if !got.OK || got.Count != 0 || got.Timing.TrackedPaths != 1 || got.Timing.EnumerationNS < 0 || got.Timing.ScanNS < 0 {
		t.Fatalf("invalid phase receipt: %+v", got)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPublicLeakLiteralPrefilterParity(t *testing.T) {
	t.Parallel()
	lines := []string{"", "ordinary source line", "\xff\x00", strings.Repeat("abc123 ", 10000),
		"xoxb-" + strings.Repeat("1", 8) + "-" + strings.Repeat("2", 8) + "-" + strings.Repeat("a", 16),
		"a@b.iam." + "gserviceaccount.com", "DG" + "X123", "lab-dg" + "x1", "/home/" + "alice/x", "/home/" + "agent/x",
		`C:\` + `uſerſ\` + `alice\x`, `C:\` + `Users\` + `alice\x`, "/Users/" + "alice/x"}
	for _, file := range []string{"docs/a.md", "docs/research/inventory/corpus.jsonl", "internal/hooks/hooks_test.go"} {
		for _, line := range lines {
			for _, rx := range auditRegexes {
				want := rx.re.MatchString(line)
				switch rx.label {
				case operatorHomePathLabel:
					want = operatorHomePathLeaks(line, strings.ReplaceAll(file, "\\", "/"))
				case privateGPUAliasLabel:
					if hardwareScrubberSelfTest[strings.ReplaceAll(file, "\\", "/")] {
						want = false
					}
				}
				if got := auditRegexMatches(rx.re, rx.label, line, file); got != want {
					t.Fatalf("prefilter changed %s for file %q line %q: got %v want %v", rx.label, file, line, got, want)
				}
			}
		}
	}
}
