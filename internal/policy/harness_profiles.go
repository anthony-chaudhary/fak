package policy

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

//go:embed harness-profiles.json
var harnessProfilesJSON []byte

// guard-floor profile lint (issue #2595).
//
// The embedded guard floor (cmd/fak/guard-default-policy.json) admits a growing
// set of harness tool aliases — Claude Code's PascalCase names, Codex's
// snake_case planning/shell names, OpenCode's lowercase names, MCP client verbs.
// Hand-maintaining that floor invites two regressions the postmortem
// (docs/notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md) names directly:
//
//   - a new host-plumbing name (Codex's update_plan) lands off-floor and a
//     guarded /goal turn loops on DEFAULT_DENY before it can do anything;
//   - a new shell-like name (shell_command, functions.shell_command, a future
//     harness's run terminal) lands ON the floor without the rm -rf / sudo /
//     curl|sh / out-of-tree-write denies, weakening the danger floor while
//     fixing compatibility.
//
// harness-profiles.json is the single description of (a) the shared
// dangerous-command rule sets every shell-like alias must inherit, (b) the
// shell-like aliases and which sets they inherit, and (c) the first-class
// harness profiles whose required host-plumbing tools the floor must cover.
// LintHarnessProfiles compares a floor manifest against it and returns one
// human-readable defect string per problem (nil/empty = clean). It is the CI
// gate: a shell alias missing an inherited rule, a harness whose required tool
// is off-floor, or an unclassified shell-program name on the allow list each
// fail the build.

// harnessRuleEntry is one deny pattern inside a named shell-danger rule set.
type harnessRuleEntry struct {
	DenyRegex string `json:"deny_regex"`
}

// harnessRuleSet bundles the deny patterns a shell-like tool must carry on a
// given argument field to count as inheriting that danger class.
type harnessRuleSet struct {
	Arg   string             `json:"arg"`
	Rules []harnessRuleEntry `json:"rules"`
}

// harnessShellAlias declares one shell-like tool alias and the named rule sets
// its argument field must inherit, so an admitted shell never bypasses the
// shared danger floor.
type harnessShellAlias struct {
	Name     string   `json:"name"`
	Inherits []string `json:"inherits"`
}

// harnessProfileDecl is one first-class harness (Claude Code, Codex, OpenCode,
// MCP client) whose required host-plumbing tools the embedded floor must cover.
type harnessProfileDecl struct {
	Name          string   `json:"name"`
	RequiredTools []string `json:"required_tools"`
}

type harnessProfilesDoc struct {
	ShellRuleSets map[string]harnessRuleSet `json:"shell_rule_sets"`
	ShellAliases  []harnessShellAlias       `json:"shell_aliases"`
	Harnesses     []harnessProfileDecl      `json:"harnesses"`
}

// normalizeDangerRegex collapses a leading inline case-insensitive flag so a
// case-insensitive pattern (which matches a SUPERSET) satisfies its
// case-sensitive twin. The cross-platform shell_command alias carries the
// (?i) terraform-destroy rule; the posix rule set's canonical terraform-destroy
// rule is case-sensitive. A stricter rule always stands in for a weaker one, so
// the normalized comparison treats them as equivalent.
func normalizeDangerRegex(s string) string {
	return strings.TrimPrefix(s, "(?i)")
}

// shellProgramNames are tool-name segments that are unambiguously a shell
// program; an allow-listed tool whose final segment matches one must carry
// danger rules. Kept tight (exact match, not substring) to avoid false
// positives on orchestration names like BashOutput or KillShell.
var shellProgramNames = map[string]bool{
	"bash": true, "powershell": true, "pwsh": true, "sh": true,
	"zsh": true, "fish": true, "ksh": true, "csh": true, "tcsh": true,
}

func looksShellLike(tool string) bool {
	last := tool
	if i := strings.LastIndex(tool, "."); i >= 0 {
		last = tool[i+1:]
	}
	return shellProgramNames[strings.ToLower(last)]
}

// LintHarnessProfiles compares a guard-floor manifest against the committed
// harness-profile data source and returns one human-readable defect string per
// problem (nil/empty = clean).
func LintHarnessProfiles(floorJSON []byte) []string {
	return lintHarnessProfilesDoc(floorJSON, harnessProfilesJSON)
}

// lintHarnessProfilesDoc is the pure, testable core: both inputs are parameters
// so the witness tests can drive it with synthetic floors/profiles without
// touching the committed artifacts.
func lintHarnessProfilesDoc(floorJSON, profilesJSON []byte) []string {
	var floor Manifest
	var prof harnessProfilesDoc
	if err := json.Unmarshal(floorJSON, &floor); err != nil {
		return []string{fmt.Sprintf("guard floor is not valid JSON: %v", err)}
	}
	if err := json.Unmarshal(profilesJSON, &prof); err != nil {
		return []string{fmt.Sprintf("harness profiles are not valid JSON: %v", err)}
	}

	var problems []string
	allowSet := make(map[string]bool, len(floor.Allow))
	for _, a := range floor.Allow {
		allowSet[a] = true
	}
	// arg rules indexed by lowercase tool name -> set of normalized deny
	// patterns. The adjudicator indexes arg predicates case-insensitively
	// (decide.go: indexArgPredicates keys on strings.ToLower), so a rule
	// authored for "Bash" covers "bash"; the lint mirrors that or it would
	// false-report OpenCode's lowercase shell.
	rulesByTool := map[string]map[string]bool{}
	for _, r := range floor.ArgRules {
		if r.DenyRegex == "" {
			continue
		}
		k := strings.ToLower(r.Tool)
		if rulesByTool[k] == nil {
			rulesByTool[k] = map[string]bool{}
		}
		rulesByTool[k][normalizeDangerRegex(r.DenyRegex)] = true
	}

	allowed := func(tool string) bool {
		if allowSet[tool] {
			return true
		}
		for _, p := range floor.AllowPrefix {
			if strings.HasPrefix(tool, p) {
				return true
			}
		}
		return false
	}

	// (1) every declared shell alias carries its inherited danger-rule set.
	for _, a := range prof.ShellAliases {
		if !allowed(a.Name) {
			problems = append(problems, fmt.Sprintf(
				"shell alias %q is declared in the harness profiles but is NOT on the guard floor allow list", a.Name))
			continue
		}
		expected := map[string]bool{}
		for _, setName := range a.Inherits {
			rs, ok := prof.ShellRuleSets[setName]
			if !ok {
				problems = append(problems, fmt.Sprintf(
					"shell alias %q inherits unknown rule set %q", a.Name, setName))
				continue
			}
			for _, rule := range rs.Rules {
				if rule.DenyRegex != "" {
					expected[normalizeDangerRegex(rule.DenyRegex)] = true
				}
			}
		}
		actual := rulesByTool[strings.ToLower(a.Name)]
		for pat := range expected {
			if !actual[pat] {
				problems = append(problems, fmt.Sprintf(
					"shell alias %q is missing inherited dangerous-command deny rule: %s", a.Name, pat))
			}
		}
	}

	// (2) every first-class harness's required tools are covered by the floor;
	//     also collect the declared-name set for the heuristic backstop below.
	declared := make(map[string]bool)
	for _, a := range prof.ShellAliases {
		declared[a.Name] = true
	}
	for _, h := range prof.Harnesses {
		for _, tool := range h.RequiredTools {
			declared[tool] = true
			if !allowed(tool) {
				problems = append(problems, fmt.Sprintf(
					"harness %q required tool %q is not on the guard floor (would DEFAULT_DENY on first call)", h.Name, tool))
			}
		}
	}

	// (3) backstop: an allow-listed, undeclared tool whose name IS a shell
	//     program must carry danger rules — catches a new shell alias added to
	//     the allow list without classification or inherited rules.
	for _, tool := range floor.Allow {
		if declared[tool] || !looksShellLike(tool) {
			continue
		}
		if len(rulesByTool[strings.ToLower(tool)]) == 0 {
			problems = append(problems, fmt.Sprintf(
				"allow-listed tool %q is a shell-program name but carries no dangerous-command deny rules; classify it in the harness profiles and add the inherited rule set", tool))
		}
	}

	return problems
}
