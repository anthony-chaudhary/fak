package adjudicator

import (
	"context"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ReasonTestTamperRefused is the closed refusal reason code emitted when a tool
// proposal attempts to write or edit gating test suites while running under an
// implementation lane (#10923).
const ReasonTestTamperRefused abi.ReasonCode = 1100

// ReasonTestTamperRefusedName is the stable name registered for ReasonTestTamperRefused.
const ReasonTestTamperRefusedName = "TEST_TAMPER_REFUSED"

// Reason constants for test immunity phase evaluations (#11050).
const (
	ReasonImplementationLockedInRedPhase = "IMPLEMENTATION_LOCKED_IN_RED_PHASE"
	ReasonTestImmuneInGreenPhase         = "TEST_IMMUNE_IN_GREEN_PHASE"
)

// Affordance constants for test immunity phase evaluations.
const (
	AffordanceImplementationLockedInRedPhase = "In Phase Red, write a test reproducing the defect before editing implementation code."
	AffordanceTestImmuneInGreenPhase         = "Reproduction test is frozen during green phase. Modify implementation files to pass the test."
)

func init() {
	abi.RegisterReason(ReasonTestTamperRefused, ReasonTestTamperRefusedName)
}

type laneContextKey struct{}

// ContextWithLane returns a derived context carrying the designated lane name.
func ContextWithLane(ctx context.Context, lane string) context.Context {
	return context.WithValue(ctx, laneContextKey{}, lane)
}

// LaneFromContext retrieves the lane name from the context, or "" if unset.
func LaneFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(laneContextKey{}).(string); ok {
		return v
	}
	return ""
}

// resolveLane determines the active lane from ToolCall metadata, arguments, context,
// active attestation, or policy configuration.
func resolveLane(ctx context.Context, p Policy, c *abi.ToolCall, args map[string]any, a *Adjudicator) string {
	if c != nil && c.Meta != nil {
		for _, k := range []string{"lane", "session_lane", "work_lane", "task_lane", "lane_type", "lane_kind", "lane_role"} {
			if v := strings.TrimSpace(c.Meta[k]); v != "" {
				return v
			}
		}
	}
	if args != nil {
		if v, ok := args["lane"].(string); ok && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	if l := LaneFromContext(ctx); strings.TrimSpace(l) != "" {
		return strings.TrimSpace(l)
	}
	if a != nil {
		if s := a.devEdit.Load(); s != nil && s.att.Lane != "" {
			return s.att.Lane
		}
	}
	if strings.TrimSpace(p.Lane) != "" {
		return strings.TrimSpace(p.Lane)
	}
	return ""
}

// IsTestLane reports whether lane is explicitly designated as a test lane.
func IsTestLane(lane string, p Policy) bool {
	trimmed := strings.TrimSpace(lane)
	if trimmed == "" {
		return false
	}
	for _, l := range p.TestLanes {
		if strings.EqualFold(trimmed, l) {
			return true
		}
	}
	for _, l := range p.ExemptLanes {
		if strings.EqualFold(trimmed, l) {
			return true
		}
	}
	lower := strings.ToLower(trimmed)
	switch lower {
	case "test", "tests", "testing", "qa", "eval", "evaluation",
		"benchmark", "benchmarks", "redteam", "test_lane", "test-lane",
		"testsuite", "test_suite", "test-suite":
		return true
	}
	if strings.HasPrefix(lower, "test/") || strings.HasPrefix(lower, "tests/") ||
		strings.HasPrefix(lower, "test-") || strings.HasPrefix(lower, "test_") ||
		strings.HasSuffix(lower, "-test") || strings.HasSuffix(lower, "_test") ||
		strings.HasSuffix(lower, "-tests") || strings.HasSuffix(lower, "_tests") ||
		strings.HasSuffix(lower, "-qa") || strings.HasSuffix(lower, "_qa") {
		return true
	}
	return false
}

// isTestImmunityExempt reports whether the current request is exempt from the
// test-immunity gate (e.g. explicitly disabled, lane designated as test, or red phase).
func isTestImmunityExempt(ctx context.Context, p Policy, c *abi.ToolCall, args map[string]any, a *Adjudicator) bool {
	if p.DisableTestImmunity {
		return true
	}
	if c != nil && c.Meta != nil {
		mode := strings.ToLower(strings.TrimSpace(c.Meta["test_immunity"]))
		if mode == "disabled" || mode == "off" || mode == "false" || mode == "exempt" {
			return true
		}
		if strings.EqualFold(c.Meta["lane_type"], "test") ||
			strings.EqualFold(c.Meta["lane_kind"], "test") ||
			strings.EqualFold(c.Meta["lane_role"], "test") {
			return true
		}
		if isRedPhase(c.Meta["workflow_phase"]) {
			return true
		}
	}
	lane := resolveLane(ctx, p, c, args, a)
	return IsTestLane(lane, p)
}

// IsTestImmunityTarget reports whether raw path refers to a gating test suite,
// fixture, or test configuration.
func IsTestImmunityTarget(raw string) (bool, string) {
	clean := filepath.ToSlash(strings.TrimSpace(raw))
	if clean == "" {
		return false, ""
	}
	clean = strings.Trim(clean, `"'`+"`")
	clean = filepath.ToSlash(filepath.Clean(clean))
	clean = strings.TrimPrefix(clean, "./")
	cleanLower := strings.ToLower(clean)
	base := filepath.Base(clean)
	baseLower := strings.ToLower(base)

	// 1. Go test files (*_test.go)
	if strings.HasSuffix(baseLower, "_test.go") {
		return true, "test suite: " + clean
	}

	// 2. Test data directory / fixtures (testdata/**)
	if cleanLower == "testdata" || strings.HasPrefix(cleanLower, "testdata/") || strings.Contains(cleanLower, "/testdata/") {
		return true, "test fixture: " + clean
	}

	// 3. Testing fixtures directories and golden files
	if cleanLower == "fixtures" || strings.HasPrefix(cleanLower, "fixtures/") || strings.Contains(cleanLower, "/fixtures/") ||
		strings.Contains(cleanLower, "test_fixtures") || strings.Contains(cleanLower, "test-fixtures") ||
		strings.HasSuffix(baseLower, ".golden") || strings.Contains(baseLower, ".fixture.") || strings.HasSuffix(baseLower, ".fixture") {
		return true, "test fixture: " + clean
	}

	// 4. Testing configs
	if isTestConfigFile(baseLower) {
		return true, "test configuration: " + clean
	}

	// 5. Test files in other languages commonly used in suites
	if strings.HasSuffix(baseLower, "_test.py") || (strings.HasPrefix(baseLower, "test_") && strings.HasSuffix(baseLower, ".py")) ||
		strings.HasSuffix(baseLower, ".test.js") || strings.HasSuffix(baseLower, ".test.ts") ||
		strings.HasSuffix(baseLower, ".spec.js") || strings.HasSuffix(baseLower, ".spec.ts") ||
		strings.HasSuffix(baseLower, ".test.jsx") || strings.HasSuffix(baseLower, ".test.tsx") ||
		strings.HasSuffix(baseLower, ".spec.jsx") || strings.HasSuffix(baseLower, ".spec.tsx") {
		return true, "test suite: " + clean
	}

	return false, ""
}

func isTestConfigFile(baseLower string) bool {
	switch baseLower {
	case "pytest.ini", "tox.ini", ".coveragerc", "conftest.py":
		return true
	}
	if strings.HasPrefix(baseLower, "jest.config.") ||
		strings.HasPrefix(baseLower, "vitest.config.") ||
		strings.HasPrefix(baseLower, "playwright.config.") ||
		strings.HasPrefix(baseLower, "cypress.config.") ||
		strings.HasPrefix(baseLower, ".mocharc.") ||
		strings.HasPrefix(baseLower, "karma.conf.") {
		return true
	}
	if strings.HasSuffix(baseLower, ".test.json") ||
		strings.HasSuffix(baseLower, ".test.toml") ||
		strings.HasSuffix(baseLower, ".test.yaml") ||
		strings.HasSuffix(baseLower, ".test.yml") {
		return true
	}
	return false
}

func isWriteOrEditTool(lowerTool string) bool {
	if writeShapedLower(lowerTool) {
		return true
	}
	for _, w := range []string{"write", "edit", "delete", "patch", "put", "modify", "create", "rm", "unlink", "truncate"} {
		if strings.Contains(lowerTool, w) {
			return true
		}
	}
	return false
}

var psWriteCmdlets = []string{"set-content", "out-file", "remove-item", "add-content", "new-item"}

func powerShellWriteTargets(cmd string) []string {
	lc := strings.ToLower(cmd)
	hasCmdlet := false
	for _, c := range psWriteCmdlets {
		if strings.Contains(lc, c) {
			hasCmdlet = true
			break
		}
	}
	if !hasCmdlet {
		return nil
	}
	var targets []string
	for _, line := range strings.Split(cmd, "\n") {
		for _, stmt := range strings.Split(line, ";") {
			stmt = strings.TrimSpace(stmt)
			if stmt == "" {
				continue
			}
			stmtLower := strings.ToLower(stmt)
			for _, c := range psWriteCmdlets {
				if idx := strings.Index(stmtLower, c); idx >= 0 {
					rest := strings.TrimSpace(stmt[idx+len(c):])
					tokens := strings.Fields(rest)
					for i := 0; i < len(tokens); i++ {
						tok := tokens[i]
						tokLower := strings.ToLower(tok)
						if tokLower == "-path" || tokLower == "-filepath" || tokLower == "-literalpath" {
							if i+1 < len(tokens) {
								targets = append(targets, strings.Trim(tokens[i+1], `"'`+"`"))
								i++
							}
							continue
						}
						if strings.HasPrefix(tok, "-") {
							continue
						}
						targets = append(targets, strings.Trim(tok, `"'`+"`"))
						break
					}
				}
			}
		}
	}
	return targets
}

func extractWriteTargets(tool, lowerTool string, args map[string]any, p Policy) []string {
	if len(args) == 0 {
		return nil
	}
	var targets []string
	seen := make(map[string]bool)
	addTarget := func(raw string) {
		trimmed := strings.TrimSpace(raw)
		trimmed = strings.Trim(trimmed, `"'`+"`")
		if trimmed == "" || isNullSink(trimmed) {
			return
		}
		clean := filepath.ToSlash(filepath.Clean(trimmed))
		if clean != "" && !seen[clean] {
			seen[clean] = true
			targets = append(targets, clean)
		}
	}

	if isWriteOrEditTool(lowerTool) {
		for _, k := range []string{
			"filePath", "file_path", "filepath",
			"path", "file", "target", "filename",
			"destination", "dest", "newPath", "new_path",
		} {
			if v, ok := args[k]; ok {
				if s, ok := v.(string); ok && s != "" {
					addTarget(s)
				}
			}
		}
		for _, k := range []string{"paths", "files", "targets"} {
			if v, ok := args[k]; ok {
				switch sl := v.(type) {
				case []string:
					for _, s := range sl {
						addTarget(s)
					}
				case []any:
					for _, item := range sl {
						if s, ok := item.(string); ok {
							addTarget(s)
						}
					}
				}
			}
		}
	}

	if cmd, ok := commandArg(args); ok && cmd != "" {
		for _, t := range commandWriteTargetsWithSpecs(cmd, p.InlineEval) {
			addTarget(t)
		}
		for _, t := range powerShellWriteTargets(cmd) {
			addTarget(t)
		}
	}

	return targets
}

// testImmunityVerdict is the test-immunity gate (#10923): it refuses write, edit, or delete
// proposals targeting gating test suites (*_test.go, testdata/**, testing fixtures/configs)
// when running under an implementation lane (or any non-test lane).
func (a *Adjudicator) testImmunityVerdict(ctx context.Context, p Policy, c *abi.ToolCall, lowerTool string, args map[string]any) (abi.Verdict, bool) {
	if isTestImmunityExempt(ctx, p, c, args, a) {
		return abi.Verdict{}, false
	}
	targets := extractWriteTargets(c.Tool, lowerTool, args, p)
	if len(targets) == 0 {
		return abi.Verdict{}, false
	}

	for _, target := range targets {
		if isTest, desc := IsTestImmunityTarget(target); isTest {
			lane := resolveLane(ctx, p, c, args, a)
			if lane == "" {
				lane = "implementation"
			}
			return p.soften(abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  ReasonTestTamperRefused,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: target},
				Meta: map[string]string{
					"rule":        "test_immunity",
					"target":      target,
					"lane":        lane,
					"description": desc,
				},
			}, nil), true
		}
	}

	return abi.Verdict{}, false
}

// IsTestPath detects whether filePath represents a test file:
// - Go test files (_test.go)
// - TypeScript / JavaScript test files (.test.ts, .spec.ts, .test.tsx, .spec.tsx, .test.js, .spec.js, etc.)
// - Python test files (test_*.py, *_test.py)
// - Java test files (*Test.java)
// - Files or paths located inside tests/ or testdata/ directories
func IsTestPath(filePath string) bool {
	trimmed := strings.TrimSpace(filePath)
	if trimmed == "" {
		return false
	}

	normalized := strings.ReplaceAll(trimmed, "\\", "/")
	clean := path.Clean(normalized)
	if clean == "." || clean == "/" || clean == "" {
		return false
	}

	parts := strings.Split(clean, "/")
	for _, part := range parts {
		low := strings.ToLower(part)
		if low == "tests" || low == "testdata" {
			return true
		}
	}

	base := path.Base(clean)

	// Go test files
	if strings.HasSuffix(base, "_test.go") {
		return true
	}

	// TypeScript / JavaScript test files
	if strings.HasSuffix(base, ".test.ts") || strings.HasSuffix(base, ".spec.ts") ||
		strings.HasSuffix(base, ".test.tsx") || strings.HasSuffix(base, ".spec.tsx") ||
		strings.HasSuffix(base, ".test.js") || strings.HasSuffix(base, ".spec.js") ||
		strings.HasSuffix(base, ".test.jsx") || strings.HasSuffix(base, ".spec.jsx") {
		return true
	}

	// Python test files
	if strings.HasSuffix(base, ".py") {
		if strings.HasPrefix(base, "test_") || strings.HasSuffix(base, "_test.py") {
			return true
		}
	}

	// Java test files
	if strings.HasSuffix(base, "Test.java") || strings.HasSuffix(base, "Tests.java") {
		return true
	}

	return false
}

func isRedPhase(workflowPhase string) bool {
	switch strings.ToLower(strings.TrimSpace(workflowPhase)) {
	case "red_reproduction", "phase_red", "repro_authoring":
		return true
	default:
		return false
	}
}

func isGreenPhase(workflowPhase string) bool {
	switch strings.ToLower(strings.TrimSpace(workflowPhase)) {
	case "green_implementation", "phase_green", "fix_implementation":
		return true
	default:
		return false
	}
}

// CheckTestImmunityPhase evaluates whether filePath can be modified during workflowPhase.
func CheckTestImmunityPhase(filePath string, workflowPhase string) (allowed bool, reason string, affordance string) {
	isTest := IsTestPath(filePath)

	if isRedPhase(workflowPhase) {
		if isTest {
			return true, "", ""
		}
		return false, ReasonImplementationLockedInRedPhase, AffordanceImplementationLockedInRedPhase
	}

	if isGreenPhase(workflowPhase) {
		if !isTest {
			return true, "", ""
		}
		return false, ReasonTestImmuneInGreenPhase, AffordanceTestImmuneInGreenPhase
	}

	return true, "", ""
}
