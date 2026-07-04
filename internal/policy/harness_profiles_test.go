package policy

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// realFloor reads the committed embedded guard floor the lint is meant to
// police, the same file cmd/fak embeds via go:embed.
func realFloor(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read embedded guard floor: %v", err)
	}
	return b
}

func problemsContain(problems []string, want string) bool {
	for _, p := range problems {
		if strings.Contains(p, want) {
			return true
		}
	}
	return false
}

// TestLintHarnessProfilesCurrentFloorIsClean is the headline gate (#2595): the
// real embedded guard floor must satisfy every harness-profile invariant. A
// regression here means the floor drifted from the profile contract — a shell
// alias lost an inherited rule, a harness's required tool fell off the allow
// list, or an unclassified shell-program name slipped in. It also proves the
// deny_regex values in harness-profiles.json match the floor verbatim.
func TestLintHarnessProfilesCurrentFloorIsClean(t *testing.T) {
	for _, p := range LintHarnessProfiles(realFloor(t)) {
		t.Errorf("harness-profile lint defect: %s", p)
	}
}

// TestLintHarnessProfilesProfilesAreValidJSON guards the data source itself.
func TestLintHarnessProfilesProfilesAreValidJSON(t *testing.T) {
	var prof harnessProfilesDoc
	if err := json.Unmarshal(harnessProfilesJSON, &prof); err != nil {
		t.Fatalf("harness-profiles.json is not valid JSON: %v", err)
	}
	if len(prof.ShellRuleSets) == 0 || len(prof.ShellAliases) == 0 || len(prof.Harnesses) == 0 {
		t.Fatalf("harness-profiles.json is empty in a required section: %+v", prof)
	}
}

// TestLintHarnessProfilesSyntheticShellAliasFails is the witness for the first
// acceptance criterion: adding a shell-like alias to the allow list WITHOUT its
// inherited dangerous-command denies must fail the lint (and thus CI). This is
// the exact regression the issue names — the dangerous mirror-image of the
// Codex/fakc coverage bug.
func TestLintHarnessProfilesSyntheticShellAliasFails(t *testing.T) {
	var m Manifest
	if err := json.Unmarshal(realFloor(t), &m); err != nil {
		t.Fatal(err)
	}
	m.Allow = append(m.Allow, "zsh") // a new shell alias, no arg rules
	mod, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	problems := LintHarnessProfiles(mod)
	if !problemsContain(problems, "zsh") {
		t.Fatalf("expected the lint to flag the unclassified shell alias \"zsh\"; got %v", problems)
	}
}

// TestLintHarnessProfilesDeclaredAliasLosingRulesFails proves a DECLARED shell
// alias whose arg rules were stripped is caught even though it stays on the
// allow list — so weakening an admitted shell's danger floor fails CI, not just
// admitting a new one without rules.
func TestLintHarnessProfilesDeclaredAliasLosingRulesFails(t *testing.T) {
	var m Manifest
	if err := json.Unmarshal(realFloor(t), &m); err != nil {
		t.Fatal(err)
	}
	var kept []ArgRule
	for _, r := range m.ArgRules {
		if !strings.EqualFold(r.Tool, "shell_command") {
			kept = append(kept, r)
		}
	}
	m.ArgRules = kept
	mod, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	problems := LintHarnessProfiles(mod)
	hit := false
	for _, p := range problems {
		if strings.Contains(p, "shell_command") && strings.Contains(p, "missing") {
			hit = true
		}
	}
	if !hit {
		t.Fatalf("expected the lint to flag shell_command missing inherited rules; got %v", problems)
	}
}

// TestLintHarnessProfilesHarnessMissingRequiredToolFails is the witness for the
// second acceptance criterion: a first-class harness profile that names a
// required host-plumbing tool the floor does not cover must fail the lint — so
// a new harness cannot ship an advertised launcher whose first tool call is
// DEFAULT_DENY.
func TestLintHarnessProfilesHarnessMissingRequiredToolFails(t *testing.T) {
	var prof harnessProfilesDoc
	if err := json.Unmarshal(harnessProfilesJSON, &prof); err != nil {
		t.Fatal(err)
	}
	prof.Harnesses = append(prof.Harnesses, harnessProfileDecl{
		Name:          "future-ide",
		RequiredTools: []string{"future_ide_planner"}, // not on the floor
	})
	profJSON, err := json.Marshal(prof)
	if err != nil {
		t.Fatal(err)
	}
	problems := lintHarnessProfilesDoc(realFloor(t), profJSON)
	if !problemsContain(problems, "future_ide_planner") {
		t.Fatalf("expected the lint to flag the uncovered required tool; got %v", problems)
	}
}
