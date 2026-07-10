package safecommit

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestCommitCleanSkillDocumentsEveryRefusalReason binds the commit-clean SKILL.md
// (the mechanization of the repo's "commit clean by default" mantra) to
// safecommit's closed refusal vocabulary: every reason fak commit can report to an
// operator must appear in the skill's refusal table, so the doc cannot silently
// drift as new reasons are added. It guards the historical gap where SYMLINK_ESCAPE
// and MERGE_IN_PROGRESS were emitted by the executor but missing from the skill.
func TestCommitCleanSkillDocumentsEveryRefusalReason(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed; cannot locate the repo root")
	}
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	skillPath := filepath.Join(repoRoot, ".claude", "skills", "commit-clean", "SKILL.md")
	data, err := os.ReadFile(skillPath)
	if err != nil {
		t.Fatalf("read commit-clean SKILL.md at %s: %v", skillPath, err)
	}
	skill := string(data)
	for _, reason := range RefusalReasons() {
		if !strings.Contains(skill, reason) {
			t.Errorf("commit-clean SKILL.md does not document refusal reason %q; add a row to the refusal-vocabulary table", reason)
		}
	}
}
