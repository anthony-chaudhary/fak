// Package taskfixture provides deterministic coding tasks for agent benchmarks.
package taskfixture

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

type Fixture struct {
	ID, Family, Prompt, TargetFile, BrokenSource, FixedSource, VisibleTest, OracleTest string
	Workflow                                                                           Workflow
}

type Workflow struct {
	Kind              string   `json:"kind"`
	Area              string   `json:"area,omitempty"`
	PriorTaskID       string   `json:"prior_task_id,omitempty"`
	PriorStateSHA256  string   `json:"prior_state_sha256,omitempty"`
	PriorSteps        []string `json:"prior_steps,omitempty"`
	PriorBeforeSource string   `json:"prior_before_source,omitempty"`
	PriorAfterSource  string   `json:"prior_after_source,omitempty"`
	PriorDiff         string   `json:"prior_diff,omitempty"`
	PriorTestSource   string   `json:"prior_test_source,omitempty"`
	RequiredSteps     []string `json:"required_steps"`
}

func Cases(includeHeldout bool) []Fixture {
	cases := []Fixture{
		retryFixture("retry-delay-base250", 250, 1000, 1, 500, retryBaseOracle),
		retryFixture("retry-delay-capped1000", 100, 1000, 9, 1000, retryCappedOracle),
		labelFixture("labels-trim-lower", "  QUEUED  ", "queued", labelTrimOracle),
		labelFixture("labels-blank", " \\t\\n", "", labelBlankOracle),
		requirementsFixture("requirements-empty", "map[string]bool{\"read\": true}", "nil", requirementsEmptyOracle),
		requirementsFixture("requirements-named", "map[string]bool{\"read\": true, \"write\": true}", "[]string{\"read\", \"write\"}", requirementsNamedOracle),
	}
	if includeHeldout {
		cases = append(cases,
			retryFixture("retry-delay-large-cap", 7, 20, 1, 14, retryLargeCapOracle),
			labelFixture("labels-unicode-space", "\\u2003Ready\\u2003", "ready", labelUnicodeOracle),
		)
	}
	return cases
}

func workflowForID(id string) Workflow {
	switch id {
	case "retry-delay-base250", "retry-delay-capped1000":
		return Workflow{Kind: "behavior_fix", Area: "retry", RequiredSteps: []string{"inspect", "edit", "test_pass"}}
	case "labels-trim-lower":
		return labelFollowupWorkflow()
	case "labels-blank":
		return labelFollowupWorkflow()
	case "requirements-empty":
		return Workflow{Kind: "tdd", Area: "requirements", RequiredSteps: []string{"inspect", "test_fail", "edit", "test_pass"}}
	case "requirements-named":
		return Workflow{Kind: "stale_reread", Area: "requirements", RequiredSteps: []string{"inspect", "edit", "reread", "test_pass"}}
	default:
		return Workflow{Kind: "behavior_fix", Area: "heldout", RequiredSteps: []string{"inspect", "edit", "test_pass"}}
	}
}

func labelFollowupWorkflow() Workflow {
	diff := `{"old_string":"return label","new_string":"return strings.ToLower(label)"}`
	return Workflow{
		Kind: "shared_area_followup", Area: "labels",
		PriorTaskID: "frozen-label-lowercase-baseline", PriorStateSHA256: sourceDigest(labelBroken),
		PriorSteps: []string{"inspect", "edit", "test_pass"}, PriorBeforeSource: labelPriorBefore,
		PriorAfterSource: labelBroken, PriorDiff: diff, PriorTestSource: labelPriorTest,
		RequiredSteps: []string{"inspect", "edit", "test_pass"},
	}
}

func workflowPrompt(workflow Workflow) string {
	steps := make([]string, len(workflow.RequiredSteps))
	for index, step := range workflow.RequiredSteps {
		steps[index] = strings.ReplaceAll(step, "_", " ")
	}
	prompt := "Implement the requested behavior without changing tests. Follow this exact evidenced workflow: " + strings.Join(steps, ", then ") + ". Edit only the original target function body; preserve the package, imports, function name and signature, and every other declaration byte-for-byte."
	if workflow.PriorTaskID != "" {
		prompt += " This is a follow-up in area " + workflow.Area + " built on supplied frozen prehistory " + workflow.PriorTaskID + " (" + strings.Join(workflow.PriorSteps, ", then ") + ") at prior-state SHA-256 " + workflow.PriorStateSHA256 + "; this prehistory is benchmark input, not work executed in this run."
		prompt += "\n\nFrozen prior after-source:\n```go\n" + workflow.PriorAfterSource + "```\nFrozen prior edit record:\n```json\n" + workflow.PriorDiff + "\n```\nFrozen prior test record:\n```go\n" + workflow.PriorTestSource + "```"
	}
	return prompt
}

func sourceDigest(source string) string {
	sum := sha256.Sum256([]byte(source))
	return hex.EncodeToString(sum[:])
}

const retryBroken = `package fixture

func RetryDelay(attempt, base, cap int) int {
	delay := attempt * base
	if delay > cap { return cap }
	return delay
}
`
const retryFixed = `package fixture

func RetryDelay(attempt, base, cap int) int {
	delay := (attempt + 1) * base
	if delay > cap { return cap }
	return delay
}
`

func retryFixture(id string, base, cap, attempt, want int, oracle string) Fixture {
	workflow := workflowForID(id)
	return Fixture{ID: id, Family: "retry-delay", Prompt: workflowPrompt(workflow), TargetFile: "retry.go", Workflow: workflow,
		BrokenSource: retryBroken, FixedSource: retryFixed,
		VisibleTest: "package fixture\nimport \"testing\"\nfunc TestRetryDelay(t *testing.T) {\n\tif got := RetryDelay(" + decimal(attempt) + ", " + decimal(base) + ", " + decimal(cap) + "); got != " + decimal(want) + " { t.Fatalf(\"got %d\", got) }\n}\n",
		OracleTest:  oracle}
}

const labelBroken = `package fixture

import "strings"

func NormalizeLabel(label string) string { return strings.ToLower(label) }
`
const labelPriorBefore = `package fixture

import "strings"

func NormalizeLabel(label string) string { return label }
`
const labelPriorTest = `package fixture
import "testing"
func TestPriorLowercase(t *testing.T) {
	if got := NormalizeLabel("READY"); got != "ready" { t.Fatalf("got %q", got) }
}
`
const labelFixed = `package fixture

import "strings"

func NormalizeLabel(label string) string { return strings.ToLower(strings.TrimSpace(label)) }
`

func labelFixture(id, input, want, oracle string) Fixture {
	workflow := workflowForID(id)
	return Fixture{ID: id, Family: "labels", Prompt: workflowPrompt(workflow), TargetFile: "labels.go", Workflow: workflow,
		BrokenSource: labelBroken, FixedSource: labelFixed,
		VisibleTest: "package fixture\nimport \"testing\"\nfunc TestNormalizeLabel(t *testing.T) {\n\tif got := NormalizeLabel(\"" + input + "\"); got != \"" + want + "\" { t.Fatalf(\"got %q\", got) }\n}\n",
		OracleTest:  oracle}
}

const requirementsBroken = `package fixture

func ContainsAll(have map[string]bool, requirements []string) bool {
	all := false
	for _, requirement := range requirements { all = all && have[requirement] }
	return all
}
`
const requirementsFixed = `package fixture

func ContainsAll(have map[string]bool, requirements []string) bool {
	all := true
	for _, requirement := range requirements { all = all && have[requirement] }
	return all
}
`

func requirementsFixture(id, have, requirements, oracle string) Fixture {
	workflow := workflowForID(id)
	return Fixture{ID: id, Family: "requirements", Prompt: workflowPrompt(workflow), TargetFile: "requirements.go", Workflow: workflow,
		BrokenSource: requirementsBroken, FixedSource: requirementsFixed,
		VisibleTest: "package fixture\nimport \"testing\"\nfunc TestContainsAll(t *testing.T) {\n\tif !ContainsAll(" + have + ", " + requirements + ") { t.Fatal(\"requirements rejected\") }\n}\n",
		OracleTest:  oracle}
}

func decimal(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	i := len(digits)
	for value > 0 {
		i--
		digits[i] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[i:])
}

const retryBaseOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	for _, tc := range []struct{ attempt, want int }{{0,250},{1,500},{4,1000}} {
		if got := RetryDelay(tc.attempt, 250, 1000); got != tc.want { t.Fatalf("attempt %d: got %d want %d", tc.attempt, got, tc.want) }
	}
}
`
const retryCappedOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	for _, tc := range []struct{ attempt, want int }{{0,100},{9,1000},{12,1000}} {
		if got := RetryDelay(tc.attempt, 100, 1000); got != tc.want { t.Fatalf("attempt %d: got %d want %d", tc.attempt, got, tc.want) }
	}
}
`
const labelTrimOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	if got := NormalizeLabel("  Ready-To-Go \n"); got != "ready-to-go" { t.Fatalf("got %q", got) }
}
`
const labelBlankOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	if got := NormalizeLabel(" \t\n"); got != "" { t.Fatalf("blank got %q", got) }
	if got := NormalizeLabel("ready"); got != "ready" { t.Fatalf("stable got %q", got) }
}
`
const requirementsEmptyOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	if !ContainsAll(map[string]bool{"read":true}, nil) { t.Fatal("empty requirements must pass") }
}
`
const requirementsNamedOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	have := map[string]bool{"read":true,"write":true}
	if !ContainsAll(have, []string{"read","write"}) { t.Fatal("present requirements rejected") }
	if ContainsAll(have, []string{"read","delete"}) { t.Fatal("missing requirement accepted") }
}
`
const retryLargeCapOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	for _, tc := range []struct{ attempt, want int }{{0,7},{1,14},{2,20}} {
		if got := RetryDelay(tc.attempt, 7, 20); got != tc.want { t.Fatalf("attempt %d: got %d want %d", tc.attempt, got, tc.want) }
	}
}
`
const labelUnicodeOracle = `package fixture
import "testing"
func TestOracle(t *testing.T) {
	if got := NormalizeLabel("\u2003MiXeD\u2003"); got != "mixed" { t.Fatalf("got %q", got) }
}
`
