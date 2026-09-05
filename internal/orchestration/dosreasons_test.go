package orchestration

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestUltracodeBudgetReasonsRegisteredInDosToml binds the closed budget vocabulary
// in ultracode_envelope.go to the workspace dos.toml [reasons] table (issue #11469).
// A token existing only in Go resolves as known=false through dos_check_reason and
// UNCLASSIFIED at runtime — the drift a closed vocabulary exists to prevent.
//
// It binds "known, refusable, actionable": each table must exist, declare category =
// "OPERATOR_GATE", refusal = true, non-empty summary and see_also, and a fix that
// names the owning floor: internal/orchestration.
func TestUltracodeBudgetReasonsRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	tokens := []string{
		UltracodeBudgetReasonTokenOverrun,
		UltracodeBudgetReasonWallOverrun,
		UltracodeBudgetReasonIncomplete,
	}
	for _, token := range tokens {
		header := "[reasons." + token + "]"
		if !strings.Contains(content, header) {
			t.Errorf("refusal reason %q has no %s table in dos.toml — dos_check_reason would return known=false", token, header)
			continue
		}
		block := dosReasonBlock(content, header)
		if category := strings.TrimSpace(reasonFieldValue(block, "category")); category != `"OPERATOR_GATE"` {
			t.Errorf("refusal reason %q category = %q, want %q", token, category, `"OPERATOR_GATE"`)
		}
		if !reasonFieldTrue(block, "refusal") {
			t.Errorf("refusal reason %q is registered but not marked refusal = true", token)
		}
		for _, field := range []string{"summary", "fix", "see_also"} {
			if !reasonFieldSet(block, field) {
				t.Errorf("refusal reason %q registration has no non-empty %s", token, field)
			}
		}
		fix := reasonFieldValue(block, "fix")
		if !strings.Contains(fix, "internal/orchestration") {
			t.Errorf("refusal reason %q fix does not name the floor 'internal/orchestration': %s", token, fix)
		}
		if !strings.Contains(block, "#11469") {
			t.Errorf("refusal reason %q registration does not cite issue #11469", token)
		}
	}
}

// TestUltracodeBudgetReasonsRegisteredInRecoveryPlans ensures that all three
// envelope tokens are registered in cmd/fak/recover.go's emittedRecoveryReasons
// and recoveryPlans() (via treeRecoveryPlans), guaranteeing actionable dry-run
// recovery guidance without live-agent execution failure (issue #11469).
func TestUltracodeBudgetReasonsRegisteredInRecoveryPlans(t *testing.T) {
	recoverSrc := readRepoFile(t, filepath.Join("cmd", "fak", "recover.go"))
	tokens := []string{
		UltracodeBudgetReasonTokenOverrun,
		UltracodeBudgetReasonWallOverrun,
		UltracodeBudgetReasonIncomplete,
	}
	for _, token := range tokens {
		if !strings.Contains(recoverSrc, `"`+token+`"`) {
			t.Errorf("refusal reason %q not found in cmd/fak/recover.go emittedRecoveryReasons", token)
		}
		if !strings.Contains(recoverSrc, `"`+token+`": {`) {
			t.Errorf("refusal reason %q not found in cmd/fak/recover.go treeRecoveryPlans/recoveryPlans", token)
		}
	}
}

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

func reasonFieldTrue(block, field string) bool {
	return strings.TrimSpace(reasonFieldValue(block, field)) == "true"
}

func reasonFieldSet(block, field string) bool {
	v := strings.TrimSpace(reasonFieldValue(block, field))
	return v != "" && v != `""` && v != "[]"
}

func reasonFieldValue(block, field string) string {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") {
			return strings.TrimSpace(rest[1:])
		}
	}
	return ""
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..")
}

func readRepoDosToml(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRoot(t), "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}

func readRepoFile(t *testing.T, relPath string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRoot(t), relPath))
	if err != nil {
		t.Fatalf("read repo file %s: %v", relPath, err)
	}
	return string(b)
}
