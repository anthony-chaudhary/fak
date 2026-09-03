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
func TestCodexHarnessRequiresPTYContinuationTools(t *testing.T) {
	var profiles harnessProfilesDoc
	err := json.Unmarshal(harnessProfilesJSON, &profiles)
	if err != nil {
		t.Fatal(err)
	}
	var required map[string]bool
	for _, harness := range profiles.Harnesses {
		if harness.Name != "codex" {
			continue
		}
		required = make(map[string]bool, len(harness.RequiredTools))
		for _, tool := range harness.RequiredTools {
			required[tool] = true
		}
	}
	if required == nil {
		t.Fatal("codex harness profile missing")
	}
	for _, tool := range []string{"write_stdin", "functions.write_stdin"} {
		if !required[tool] {
			t.Errorf("Codex PTY continuation tool %q is off the built-in floor", tool)
		}
	}
}
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

func TestCapturedCodexToolSchemaCurrentAndLegacyAliasesPass(t *testing.T) {
	var captured capturedToolSchemaDoc
	if err := json.Unmarshal(codexToolSchemaJSON, &captured); err != nil {
		t.Fatalf("codex-tool-schema.json is not valid JSON: %v", err)
	}
	if captured.Schema != "fak-captured-tool-schema/v1" || captured.Provenance.Product != "codex-cli" || captured.Provenance.Version != "0.148.0" || //boundarylint:ignore CHANGE_DETECTOR_TEST captured fixture provenance must name its source release
		captured.Provenance.CapturedOn == "" || !captured.Provenance.Sanitized {
		t.Fatalf("unexpected capture provenance: %+v", captured)
	}
	want := map[string]struct {
		source string
		arg    string
	}{
		"exec_command":            {source: "captured", arg: "cmd"},
		"shell_command":           {source: "committed_legacy", arg: "command"},
		"functions.shell_command": {source: "committed_legacy", arg: "command"},
	}
	for _, executor := range captured.ShellExecutors {
		expected, ok := want[executor.Name]
		if !ok {
			t.Errorf("unexpected captured shell executor %q", executor.Name)
			continue
		}
		if executor.Source != expected.source || executor.CommandArg != expected.arg {
			t.Errorf("captured shell executor %q = source %q arg %q, want source %q arg %q", executor.Name, executor.Source, executor.CommandArg, expected.source, expected.arg)
		}
		delete(want, executor.Name)
	}
	if len(want) != 0 {
		t.Fatalf("captured schema is missing shell executors: %v", want)
	}
	if problems := LintHarnessProfiles(realFloor(t)); len(problems) != 0 {
		t.Fatalf("current and legacy captured shell schemas do not match the guard floor: %v", problems)
	}
}

func TestLintHarnessProfilesCapturedCodexSchemaMutationsFail(t *testing.T) {
	var baseProfiles harnessProfilesDoc
	if err := json.Unmarshal(harnessProfilesJSON, &baseProfiles); err != nil {
		t.Fatal(err)
	}
	var baseSchema capturedToolSchemaDoc
	if err := json.Unmarshal(codexToolSchemaJSON, &baseSchema); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name    string
		mutate  func(*harnessProfilesDoc, *capturedToolSchemaDoc)
		problem string
	}{
		{
			name: "missing exec_command",
			mutate: func(_ *harnessProfilesDoc, schema *capturedToolSchemaDoc) {
				var kept []capturedShellExecutor
				for _, executor := range schema.ShellExecutors {
					if executor.Name != "exec_command" {
						kept = append(kept, executor)
					}
				}
				schema.ShellExecutors = kept
			},
			problem: `Codex shell alias "exec_command" is missing`,
		},
		{
			name: "cmd command drift",
			mutate: func(profiles *harnessProfilesDoc, _ *capturedToolSchemaDoc) {
				for i := range profiles.ShellAliases {
					if profiles.ShellAliases[i].Name == "exec_command" {
						profiles.ShellAliases[i].Arg = "command"
					}
				}
			},
			problem: `command argument is "cmd"`,
		},
		{
			name: "missing posix danger dialect",
			mutate: func(profiles *harnessProfilesDoc, _ *capturedToolSchemaDoc) {
				removeAliasInheritance(profiles, "exec_command", "posix_shell")
			},
			problem: `missing required danger class "posix_shell"`,
		},
		{
			name: "missing windows danger dialect",
			mutate: func(profiles *harnessProfilesDoc, _ *capturedToolSchemaDoc) {
				removeAliasInheritance(profiles, "exec_command", "windows_shell")
			},
			problem: `missing required danger class "windows_shell"`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			profiles := baseProfiles
			profiles.ShellAliases = append([]harnessShellAlias(nil), baseProfiles.ShellAliases...)
			for i := range profiles.ShellAliases {
				profiles.ShellAliases[i].Inherits = append([]string(nil), baseProfiles.ShellAliases[i].Inherits...)
			}
			schema := baseSchema
			schema.ShellExecutors = append([]capturedShellExecutor(nil), baseSchema.ShellExecutors...)
			tc.mutate(&profiles, &schema)
			profilesJSON, err := json.Marshal(profiles)
			if err != nil {
				t.Fatal(err)
			}
			schemaJSON, err := json.Marshal(schema)
			if err != nil {
				t.Fatal(err)
			}
			problems := lintHarnessProfilesInputs(realFloor(t), profilesJSON, schemaJSON)
			if !problemsContain(problems, tc.problem) {
				t.Fatalf("expected problem containing %q; got %v", tc.problem, problems)
			}
		})
	}
}

func removeAliasInheritance(profiles *harnessProfilesDoc, aliasName, class string) {
	for i := range profiles.ShellAliases {
		if profiles.ShellAliases[i].Name != aliasName {
			continue
		}
		var kept []string
		for _, inherited := range profiles.ShellAliases[i].Inherits {
			if inherited != class {
				kept = append(kept, inherited)
			}
		}
		profiles.ShellAliases[i].Inherits = kept
	}
}

func TestLintHarnessProfilesFloorArgumentDriftFails(t *testing.T) {
	var floor Manifest
	if err := json.Unmarshal(realFloor(t), &floor); err != nil {
		t.Fatal(err)
	}
	for i := range floor.ArgRules {
		if floor.ArgRules[i].Tool == "exec_command" {
			floor.ArgRules[i].Arg = "command"
		}
	}
	floorJSON, err := json.Marshal(floor)
	if err != nil {
		t.Fatal(err)
	}
	problems := LintHarnessProfiles(floorJSON)
	if !problemsContain(problems, `exec_command" argument "cmd" is missing`) {
		t.Fatalf("expected the lint to bind exec_command danger rules to cmd; got %v", problems)
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

func TestShellDangerRulesForNamespacedShells(t *testing.T) {
	cases := []struct {
		name string
		sets string
	}{
		{"opencode.bash", "posix_shell"},
		{"mcp__x__bash", "posix_shell"},
		{"mcp__x__pwsh", "windows_shell"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rules, ok := ShellDangerRulesFor(tc.name)
			if !ok || len(rules) == 0 {
				t.Fatalf("ShellDangerRulesFor(%q) = %d, %v; want inherited rules", tc.name, len(rules), ok)
			}
			if got := strings.Join(ShellDangerRuleSetsFor(tc.name), ","); got != tc.sets {
				t.Fatalf("rule sets = %q, want %q", got, tc.sets)
			}
			for _, rule := range rules {
				if rule.Tool != tc.name || rule.Arg != "command" || rule.DenyRegex == "" {
					t.Fatalf("bad inherited rule: %+v", rule)
				}
			}
		})
	}
	if rules, ok := ShellDangerRulesFor("mcp__x__search"); ok || len(rules) != 0 {
		t.Fatalf("non-shell inherited rules = %+v, %v; want none", rules, ok)
	}
}

func TestShellDangerRulesForAliasArgumentOverride(t *testing.T) {
	cases := []struct {
		tool string
		arg  string
	}{
		{tool: "exec_command", arg: "cmd"},
		{tool: "shell_command", arg: "command"},
		{tool: "functions.shell_command", arg: "command"},
	}
	for _, tc := range cases {
		t.Run(tc.tool, func(t *testing.T) {
			rules, ok := ShellDangerRulesFor(tc.tool)
			if !ok || len(rules) == 0 {
				t.Fatalf("ShellDangerRulesFor(%q) = %d, %v; want inherited rules", tc.tool, len(rules), ok)
			}
			for _, rule := range rules {
				if rule.Tool != tc.tool || rule.Arg != tc.arg || rule.DenyRegex == "" {
					t.Fatalf("bad inherited rule: %+v", rule)
				}
			}
		})
	}
}

func TestHarnessToolSchemaDriftMutationCoverage(t *testing.T) {
	floorBytes := realFloor(t)
	var floor struct {
		Allow []string `json:"allow"`
	}
	if err := json.Unmarshal(floorBytes, &floor); err != nil {
		t.Fatal(err)
	}

	for _, omittedTool := range []string{"write_stdin", "functions.write_stdin", "update_plan", "exec_command"} {
		t.Run("omit_"+omittedTool, func(t *testing.T) {
			var filtered []string
			for _, a := range floor.Allow {
				if a != omittedTool {
					filtered = append(filtered, a)
				}
			}
			mutated, err := json.Marshal(map[string]any{"allow": filtered})
			if err != nil {
				t.Fatal(err)
			}
			problems := LintHarnessProfiles(mutated)
			if !problemsContain(problems, omittedTool) {
				t.Fatalf("omitting tool %q was not caught: %v", omittedTool, problems)
			}
		})
	}

	t.Run("unclassified_tool_requires_review", func(t *testing.T) {
		var schema capturedToolSchemaDoc
		if err := json.Unmarshal(codexToolSchemaJSON, &schema); err != nil {
			t.Fatal(err)
		}
		schema.HostTools = append(schema.HostTools, capturedHostTool{
			Name:           "dangerous_remote_eval",
			Classification: "unreviewed",
		})
		schemaJSON, err := json.Marshal(schema)
		if err != nil {
			t.Fatal(err)
		}
		problems := lintHarnessProfilesInputs(floorBytes, harnessProfilesJSON, schemaJSON)
		if !problemsContain(problems, "review required") || !problemsContain(problems, "dangerous_remote_eval") {
			t.Fatalf("unclassified tool was not flagged for review: %v", problems)
		}
	})
}
