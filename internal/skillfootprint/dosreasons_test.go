package skillfootprint

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestSkillDescBudgetReasonsRegisteredInDosToml is #5444's structural half: both
// members of this gate's closed refusal vocabulary must be declared in the workspace
// dos.toml [reasons] table, so `dos_check_reason` resolves each token as known and
// refusable and `dos_refuse_reasons` lists it. The Go constants above are the
// producer of record; this test fails loudly if a token is added or renamed in Go
// without a matching [reasons.<TOKEN>] registration — the drift that otherwise
// surfaces at runtime as an UNCLASSIFIED free-text reason.
//
// It binds BOTH halves of "known, refusable": the table must exist AND declare
// refusal = true, and cite issue #5444 in the block so the registration stays
// traceable to this gate. A header alone (left at refusal = false, or copied from a
// sibling without the provenance) would resolve the token as non-refusable — a
// silent regression a header-presence check could not catch.
func TestSkillDescBudgetReasonsRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	for _, token := range []string{ReasonSkillDescBudgetExceeded, ReasonSkillDescBudgetStale} {
		header := "[reasons." + token + "]"
		if !strings.Contains(content, header) {
			t.Errorf("refusal token %q has no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED drift)", token, header)
			continue
		}
		block := dosReasonBlock(content, header)
		if !reasonFieldTrue(block, "refusal") {
			t.Errorf("refusal token %q is registered but not marked refusal = true — dos_check_reason would resolve it as non-refusable", token)
		}
		if !strings.Contains(block, "issue #5444") {
			t.Errorf("refusal token %q registration does not cite issue #5444 in its table — the gate provenance is unbound", token)
		}
		if !strings.Contains(block, "SkillDescriptionBudgetBytes") {
			t.Errorf("refusal token %q registration does not name the constant an author raises to clear it — the fix is not actionable", token)
		}
	}
}

// dosReasonBlock returns the text of the [reasons.<TOKEN>] table named by header:
// from the header line up to (but excluding) the next top-level [section] or EOF.
// This scopes the binding assertions to a single reason's fields rather than
// matching a sibling table's refusal/summary by accident.
func dosReasonBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// reasonFieldTrue reports whether block contains a `field = true` line, tolerant of
// the aligned whitespace the dos.toml [reasons] tables use (e.g. "refusal  = true").
func reasonFieldTrue(block, field string) bool {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") && strings.TrimSpace(rest[1:]) == "true" {
			return true
		}
	}
	return false
}

// readRepoDosToml reads the repo-root dos.toml located relative to this package's
// own source path, so the lookup is independent of the test's working directory.
func readRepoDosToml(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRootForTest(t), "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}

// TestBaselineDocPinsTheCommittedCeiling closes the loop #5444 asks for on the doc:
// the committed baseline note must carry the SAME number the constant does, plus the
// command that regenerates it. A doc that quietly falls behind the constant is how a
// ratchet stops being reviewable — the reviewer reads the doc, not the source.
func TestBaselineDocPinsTheCommittedCeiling(t *testing.T) {
	path := filepath.Join(repoRootForTest(t), "docs", "context-budget", "skill-description-floor.md")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read the committed baseline doc: %v", err)
	}
	doc := string(b)
	for _, want := range []string{
		strconv.Itoa(SkillDescriptionBudgetBytes),
		"fak skill footprint",
		ReasonSkillDescBudgetExceeded,
		ReasonSkillDescBudgetStale,
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("docs/context-budget/skill-description-floor.md does not carry %q — the baseline drifted from the constant", want)
		}
	}
}
