package safecommit

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
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

// TestRefusalExitCodeClassifiesEveryReason asserts every closed-vocabulary refusal reason
// has an explicit exit-code classification. It guards the gap where a newly-added reason
// would fall through cmd/fak's commitExitCode default to the halt-class exit 1 and wedge a
// loop that treats exit 1 as "stop, human review" when the refusal was actually a blocked
// pre-commit attempt.
func TestRefusalExitCodeClassifiesEveryReason(t *testing.T) {
	for _, reason := range RefusalReasons() {
		code, ok := RefusalExitCode(reason)
		if !ok {
			t.Errorf("refusal reason %q is not classified by RefusalExitCode; add it to the contention, refused or post-commit case", reason)
			continue
		}
		if code != ExitLockBusy && code != ExitRefused && code != ExitPostCommitFailure {
			t.Errorf("refusal reason %q classified as exit %d, want %d (contention), %d (refused) or %d (post-commit)",
				reason, code, ExitLockBusy, ExitRefused, ExitPostCommitFailure)
		}
	}
}

// TestCommitCleanSkillDocumentsTheExitCodeClass binds the published exit-code contract to
// its documentation: the skill's "Exit codes" section must list each refusal reason under
// the code fak commit actually returns for it. The #5505 W4 split only helps a caller if
// the doc it reads agrees with the binary — the pre-split doc said exit 3 was "safe to
// retry" while listing OFF_TRUNK and CORE_SELF_MODIFY under it, which is precisely how a
// lander came to burn a retry budget on a refusal that could never clear.
func TestCommitCleanSkillDocumentsTheExitCodeClass(t *testing.T) {
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
	_, section, found := strings.Cut(string(data), "## Exit codes")
	if !found {
		t.Fatalf("commit-clean SKILL.md has no '## Exit codes' section to bind the wire contract to")
	}
	// One bullet per published code: "- **3** — ... Reasons: `A`, `B`."
	bullet := regexp.MustCompile(`(?m)^- \*\*(\d+)\*\*(.*)$`)
	byCode := map[string]string{}
	for _, m := range bullet.FindAllStringSubmatch(section, -1) {
		byCode[m[1]] = m[2]
	}
	for _, code := range []int{ExitLockBusy, ExitRefused, ExitPostCommitFailure} {
		if _, ok := byCode[strconv.Itoa(code)]; !ok {
			t.Errorf("commit-clean SKILL.md 'Exit codes' has no bullet for exit %d; add '- **%d** — ...'", code, code)
		}
	}
	for _, reason := range RefusalReasons() {
		code, ok := RefusalExitCode(reason)
		if !ok {
			continue // already reported by TestRefusalExitCodeClassifiesEveryReason
		}
		for docCode, line := range byCode {
			mentions := strings.Contains(line, "`"+reason+"`")
			switch {
			case docCode == strconv.Itoa(code) && !mentions:
				t.Errorf("refusal reason %q exits %d but the SKILL.md exit-%s bullet does not list it", reason, code, docCode)
			case docCode != strconv.Itoa(code) && mentions:
				t.Errorf("refusal reason %q exits %d but the SKILL.md exit-%s bullet lists it — the doc contradicts the binary", reason, code, docCode)
			}
		}
	}
}
