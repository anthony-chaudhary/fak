package policy

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

//go:embed harness-profiles.json
var harnessProfilesJSON []byte

//go:embed testdata/codex-tool-schema.json
var codexToolSchemaJSON []byte

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
	Arg      string   `json:"arg,omitempty"`
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

type capturedToolSchemaProvenance struct {
	Product    string `json:"product"`
	Version    string `json:"version"`
	CapturedOn string `json:"captured_on"`
	Sanitized  bool   `json:"sanitized"`
}

type capturedToolInputSchema struct {
	Type       string                     `json:"type"`
	Properties map[string]json.RawMessage `json:"properties"`
	Required   []string                   `json:"required"`
}

type capturedShellExecutor struct {
	Name                  string                  `json:"name"`
	Source                string                  `json:"source"`
	CommandArg            string                  `json:"command_arg"`
	InputSchema           capturedToolInputSchema `json:"input_schema"`
	RequiredDangerClasses []string                `json:"required_danger_classes"`
}

type capturedHostTool struct {
	Name           string `json:"name"`
	Classification string `json:"classification"`
}

type capturedToolSchemaDoc struct {
	Schema         string                       `json:"schema"`
	Provenance     capturedToolSchemaProvenance `json:"provenance"`
	ShellExecutors []capturedShellExecutor      `json:"shell_executors"`
	HostTools      []capturedHostTool           `json:"host_tools,omitempty"`
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

func shellToolSegment(tool string) string {
	last := tool
	if i := strings.LastIndex(last, "."); i >= 0 {
		last = last[i+1:]
	}
	if i := strings.LastIndex(last, "__"); i >= 0 {
		last = last[i+2:]
	}
	return strings.ToLower(last)
}

func looksShellLike(tool string) bool {
	return shellProgramNames[shellToolSegment(tool)]
}

func shellAliasFor(prof harnessProfilesDoc, tool string) (harnessShellAlias, bool) {
	for _, alias := range prof.ShellAliases {
		if strings.EqualFold(alias.Name, tool) {
			return alias, true
		}
	}
	return harnessShellAlias{}, false
}

func shellRuleArg(alias harnessShellAlias, set harnessRuleSet) string {
	if alias.Arg != "" {
		return alias.Arg
	}
	return set.Arg
}

// ShellDangerRuleSetsFor returns the inherited danger classes for a shell-like
// tool name. It recognizes committed aliases plus namespaced dot and MCP `__`
// forms whose final segment is an unambiguous shell program.
func ShellDangerRuleSetsFor(tool string) []string {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return nil
	}
	var prof harnessProfilesDoc
	if err := json.Unmarshal(harnessProfilesJSON, &prof); err != nil {
		return nil // embedded data is CI-linted; fail closed at its callers
	}
	for _, alias := range prof.ShellAliases {
		if strings.EqualFold(alias.Name, tool) {
			return append([]string(nil), alias.Inherits...)
		}
	}
	switch shellToolSegment(tool) {
	case "powershell", "pwsh":
		return []string{"windows_shell"}
	case "bash", "sh", "zsh", "fish", "ksh", "csh", "tcsh":
		return []string{"posix_shell"}
	default:
		return nil
	}
}

// ShellDangerRulesFor materializes the shared danger classes for tool as
// manifest rules. The returned rules are fresh values and can be compiled into
// a live Runtime without mutating the embedded profile data.
func ShellDangerRulesFor(tool string) ([]ArgRule, bool) {
	var prof harnessProfilesDoc
	if err := json.Unmarshal(harnessProfilesJSON, &prof); err != nil {
		return nil, false
	}
	sets := ShellDangerRuleSetsFor(tool)
	if len(sets) == 0 {
		return nil, false
	}
	alias, declared := shellAliasFor(prof, tool)
	var out []ArgRule
	for _, setName := range sets {
		set, ok := prof.ShellRuleSets[setName]
		if !ok {
			return nil, false
		}
		arg := set.Arg
		if declared {
			arg = shellRuleArg(alias, set)
		}
		for _, entry := range set.Rules {
			out = append(out, ArgRule{Tool: tool, Arg: arg, DenyRegex: entry.DenyRegex})
		}
	}
	return out, len(out) > 0
}

// AttachShellDangerRules compiles and appends the inherited danger rules for a
// newly admitted shell-like tool. It returns the rule-set names attached.
func AttachShellDangerRules(rt *Runtime, tool string) []string {
	if rt == nil {
		return nil
	}
	rules, ok := ShellDangerRulesFor(tool)
	if !ok {
		return nil
	}
	preds, err := compileArgRules(rules)
	if err != nil {
		return nil // embedded patterns are compile-tested by the profile lint
	}
	for _, candidate := range preds {
		duplicate := false
		for _, existing := range rt.Adjudicator.ArgPredicates {
			if strings.EqualFold(existing.Tool, candidate.Tool) && existing.Arg == candidate.Arg && existing.Kind == candidate.Kind && existing.Re != nil && candidate.Re != nil && existing.Re.String() == candidate.Re.String() {
				duplicate = true
				break
			}
		}
		if !duplicate {
			rt.Adjudicator.ArgPredicates = append(rt.Adjudicator.ArgPredicates, candidate)
		}
	}
	return ShellDangerRuleSetsFor(tool)
}

// LintHarnessProfiles compares a guard-floor manifest against the committed
// harness-profile data source and returns one human-readable defect string per
// problem (nil/empty = clean).
func LintHarnessProfiles(floorJSON []byte) []string {
	return lintHarnessProfilesInputs(floorJSON, harnessProfilesJSON, codexToolSchemaJSON)
}

// lintHarnessProfilesDoc is the pure, testable core: both inputs are parameters
// so the witness tests can drive it with synthetic floors/profiles without
// touching the committed artifacts.
func lintHarnessProfilesDoc(floorJSON, profilesJSON []byte) []string {
	return lintHarnessProfilesInputs(floorJSON, profilesJSON, nil)
}

func lintHarnessProfilesInputs(floorJSON, profilesJSON, schemaJSON []byte) []string {
	var floor Manifest
	var prof harnessProfilesDoc
	var captured capturedToolSchemaDoc
	if err := json.Unmarshal(floorJSON, &floor); err != nil {
		return []string{fmt.Sprintf("guard floor is not valid JSON: %v", err)}
	}
	if err := json.Unmarshal(profilesJSON, &prof); err != nil {
		return []string{fmt.Sprintf("harness profiles are not valid JSON: %v", err)}
	}
	if len(schemaJSON) != 0 {
		if err := json.Unmarshal(schemaJSON, &captured); err != nil {
			return []string{fmt.Sprintf("captured Codex tool schema is not valid JSON: %v", err)}
		}
	}

	var problems []string
	allowSet := make(map[string]bool, len(floor.Allow))
	for _, a := range floor.Allow {
		allowSet[a] = true
	}
	// Arg rules are indexed by lowercase tool name and exact argument key. The
	// adjudicator folds tool names but argument keys are schema-defined, so the
	// lint must preserve that same identity or cmd/command drift stays hidden.
	rulesByToolArg := map[string]map[string]map[string]bool{}
	for _, r := range floor.ArgRules {
		if r.DenyRegex == "" {
			continue
		}
		tool := strings.ToLower(r.Tool)
		if rulesByToolArg[tool] == nil {
			rulesByToolArg[tool] = map[string]map[string]bool{}
		}
		if rulesByToolArg[tool][r.Arg] == nil {
			rulesByToolArg[tool][r.Arg] = map[string]bool{}
		}
		rulesByToolArg[tool][r.Arg][normalizeDangerRegex(r.DenyRegex)] = true
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
		for _, setName := range a.Inherits {
			rs, ok := prof.ShellRuleSets[setName]
			if !ok {
				problems = append(problems, fmt.Sprintf(
					"shell alias %q inherits unknown rule set %q", a.Name, setName))
				continue
			}
			arg := shellRuleArg(a, rs)
			actual := rulesByToolArg[strings.ToLower(a.Name)][arg]
			for _, rule := range rs.Rules {
				if rule.DenyRegex != "" {
					pat := normalizeDangerRegex(rule.DenyRegex)
					if !actual[pat] {
						problems = append(problems, fmt.Sprintf(
							"shell alias %q argument %q is missing inherited dangerous-command deny rule: %s", a.Name, arg, pat))
					}
				}
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
		if len(rulesByToolArg[strings.ToLower(tool)]) == 0 {
			problems = append(problems, fmt.Sprintf(
				"allow-listed tool %q is a shell-program name but carries no dangerous-command deny rules; classify it in the harness profiles and add the inherited rule set", tool))
		}
	}

	if len(schemaJSON) != 0 {
		problems = append(problems, lintCapturedToolSchema(captured, prof, allowed)...)
	}

	return problems
}

func lintCapturedToolSchema(captured capturedToolSchemaDoc, prof harnessProfilesDoc, allowed func(string) bool) []string {
	var problems []string
	if captured.Schema == "" || captured.Provenance.Product == "" || captured.Provenance.Version == "" || captured.Provenance.CapturedOn == "" || !captured.Provenance.Sanitized {
		problems = append(problems, "captured Codex tool schema is missing sanitized capture provenance")
	}

	aliases := make(map[string]harnessShellAlias, len(prof.ShellAliases))
	for _, alias := range prof.ShellAliases {
		aliases[alias.Name] = alias
	}
	codexRequired := map[string]bool{}
	for _, harness := range prof.Harnesses {
		if harness.Name != "codex" {
			continue
		}
		for _, tool := range harness.RequiredTools {
			codexRequired[tool] = true
		}
	}

	capturedByName := make(map[string]capturedShellExecutor, len(captured.ShellExecutors))
	for _, executor := range captured.ShellExecutors {
		if _, duplicate := capturedByName[executor.Name]; duplicate {
			problems = append(problems, fmt.Sprintf("captured Codex tool schema repeats shell executor %q", executor.Name))
			continue
		}
		capturedByName[executor.Name] = executor
		alias, ok := aliases[executor.Name]
		if !ok {
			problems = append(problems, fmt.Sprintf("captured Codex shell executor %q is not declared as a shell alias", executor.Name))
			continue
		}
		if !allowed(executor.Name) {
			problems = append(problems, fmt.Sprintf("captured Codex shell executor %q is not on the guard floor", executor.Name))
		}
		if executor.CommandArg == "" {
			problems = append(problems, fmt.Sprintf("captured Codex shell executor %q has no command-bearing argument", executor.Name))
		} else {
			if _, ok := executor.InputSchema.Properties[executor.CommandArg]; !ok {
				problems = append(problems, fmt.Sprintf("captured Codex shell executor %q command argument %q is absent from its input schema", executor.Name, executor.CommandArg))
			}
			if !stringSliceContains(executor.InputSchema.Required, executor.CommandArg) {
				problems = append(problems, fmt.Sprintf("captured Codex shell executor %q command argument %q is not required by its input schema", executor.Name, executor.CommandArg))
			}
		}
		for _, class := range executor.RequiredDangerClasses {
			set, exists := prof.ShellRuleSets[class]
			if !exists {
				problems = append(problems, fmt.Sprintf("captured Codex shell executor %q requires unknown danger class %q", executor.Name, class))
				continue
			}
			if !stringSliceContains(alias.Inherits, class) {
				problems = append(problems, fmt.Sprintf("captured Codex shell executor %q is missing required danger class %q", executor.Name, class))
			}
			if got := shellRuleArg(alias, set); executor.CommandArg != "" && got != executor.CommandArg {
				problems = append(problems, fmt.Sprintf("captured Codex shell executor %q command argument is %q but danger class %q binds %q", executor.Name, executor.CommandArg, class, got))
			}
		}
	}

	for tool := range codexRequired {
		if _, shellAlias := aliases[tool]; shellAlias {
			if _, captured := capturedByName[tool]; !captured {
				problems = append(problems, fmt.Sprintf("Codex shell alias %q is missing from the captured tool schema", tool))
			}
		}
	}

	for _, hostTool := range captured.HostTools {
		if hostTool.Classification != "host_plumbing" {
			problems = append(problems, fmt.Sprintf("harness %q (version %s): tool %q is unclassified (%q); review required",
				captured.Provenance.Product, captured.Provenance.Version, hostTool.Name, hostTool.Classification))
			continue
		}
		if !allowed(hostTool.Name) {
			problems = append(problems, fmt.Sprintf("harness %q (version %s): missing required host plumbing tool %q; remediation: admit it in cmd/fak/guard-default-policy.json and harness-profiles.json",
				captured.Provenance.Product, captured.Provenance.Version, hostTool.Name))
		}
		if !codexRequired[hostTool.Name] {
			problems = append(problems, fmt.Sprintf("harness %q (version %s): tool %q is classified as required host plumbing but missing from harness-profiles.json; remediation: add to required_tools",
				captured.Provenance.Product, captured.Provenance.Version, hostTool.Name))
		}
	}
	return problems
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
