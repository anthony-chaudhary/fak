package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func validDispatchableIssueBody(lane string, paths []string, steps int) string {
	var pathLines strings.Builder
	for _, p := range paths {
		pathLines.WriteString("- " + p + "\n")
	}
	return `## Working spine
Implement kernel optimizations

## Current state
Kernel runs slowly on AMD APU.

## Why this is next
Unblocks kernel performance milestones.

## Parent context
#100

## Core through-line
Change -> real seam -> observable outcome -> witness.

## Gold-plating boundary
Do not add unrelated polish.

## Done condition / witness
Defect resolved and tests pass.
Witness: go test ./internal/compute/...

## Definition of done
- [ ] kernel optimization implemented
- [ ] tests pass

## Acceptance gate
All unit tests pass.

## Closure binding
Resolving commit cites issue and carries (fak ` + lane + `).

## Lane
` + lane + `

## Likely files
` + pathLines.String() + `
## Expected steps
` + strconv.Itoa(steps) + `

- Centrality: Core
- P1 Context: advanced - captures context once
- P2 Net value: preserved - no regression
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: advanced - real path

## Work estimate
Estimate: 2 points

## Overall completion contribution
Contribution: 2/8 points

## Completion standard
demo
`
}

func validSerialIssueBody() string {
	return `## Working spine
Update ABI frozen types

## Current state
ABI types need alignment with vDSO.

## Why this is next
Unblocks ABI frozen sync.

## Parent context
#100

## Core through-line
Change -> real seam -> observable outcome -> witness.

## Gold-plating boundary
Do not add unrelated polish.

## Done condition / witness
Defect resolved and tests pass.
Witness: go test ./internal/abi/...

## Definition of done
- [ ] frozen types aligned
- [ ] tests pass

## Acceptance gate
All unit tests pass.

## Closure binding
Resolving commit cites issue and carries (fak abi).

## Lane
abi

## Likely files
- internal/abi/types.go

## Expected steps
3

- Centrality: Core
- P1 Context: advanced - captures context once
- P2 Net value: preserved - no regression
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: advanced - real path

## Work estimate
Estimate: 2 points

## Overall completion contribution
Contribution: 2/8 points

## Completion standard
demo
`
}

func TestIssueDiscoverabilityParallelWaveDispatchable(t *testing.T) {
	body := validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "feat(compute): optimize kernel",
		"--body", body,
		"--json",
	})
	if code != 0 {
		t.Fatalf("code = %d, want 0; stdout:\n%s\nstderr:\n%s", code, stdout.String(), stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v\nstdout:\n%s", err, stdout.String())
	}
	if !res.OK {
		t.Fatalf("res.OK = false, want true")
	}
	if res.Schema != DiscoverabilitySchema {
		t.Fatalf("schema = %q, want %q", res.Schema, DiscoverabilitySchema)
	}
	if res.Total != 1 || res.Dispatchable != 1 || res.Parallel != 1 {
		t.Fatalf("counts: total=%d dispatchable=%d parallel=%d", res.Total, res.Dispatchable, res.Parallel)
	}
	if len(res.Issues) != 1 {
		t.Fatalf("expected 1 issue, got %d", len(res.Issues))
	}
	iss := res.Issues[0]
	if !iss.Dispatchable {
		t.Fatalf("issue dispatchable = false, want true")
	}
	if iss.Placement != PlacementParallelWave {
		t.Fatalf("placement = %q, want %q", iss.Placement, PlacementParallelWave)
	}
	if iss.Lane != "compute" {
		t.Fatalf("lane = %q, want compute", iss.Lane)
	}
}

func TestIssueDiscoverabilitySerialWaveDispatchable(t *testing.T) {
	body := validSerialIssueBody()
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "fix(abi): align frozen types",
		"--body", body,
		"--json",
	})
	if code != 0 {
		t.Fatalf("code = %d, want 0; stderr:\n%s", code, stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v\nstdout:\n%s", err, stdout.String())
	}
	if !res.OK || res.Serial != 1 || res.Dispatchable != 1 {
		t.Fatalf("res: OK=%v serial=%d dispatchable=%d", res.OK, res.Serial, res.Dispatchable)
	}
	iss := res.Issues[0]
	if !iss.Dispatchable || iss.Placement != PlacementSerialWave || !iss.IsSerial {
		t.Fatalf("issue = %+v", iss)
	}
}

func TestIssueDiscoverabilitySubdivideQueueOversizedSteps(t *testing.T) {
	body := `## Working spine
Giant epic

## Parent context
#100

## Core through-line
Change -> real seam -> observable outcome -> witness.

## Gold-plating boundary
Do not add unrelated polish.

## Done condition / witness
Defect resolved and tests pass.
Witness: go test ./internal/compute/...

## Lane
compute

## Likely files
- internal/compute/kernel.go

## Expected steps
20

- Centrality: Core
- P1 Context: advanced - captures context once
- P2 Net value: preserved - no regression
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: advanced - real path
`
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "feat(compute): giant epic",
		"--body", body,
		"--json",
	})
	if code != 3 {
		t.Fatalf("code = %d, want 3; stderr:\n%s", code, stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v\nstdout:\n%s", err, stdout.String())
	}
	if res.OK {
		t.Fatalf("res.OK = true, want false")
	}
	if res.Subdivide != 1 {
		t.Fatalf("subdivide count = %d, want 1", res.Subdivide)
	}
	iss := res.Issues[0]
	if iss.Placement != PlacementSubdivideQueue || iss.Dispatchable {
		t.Fatalf("iss = %+v", iss)
	}
	if !iss.IsSubdivide {
		t.Fatalf("is_subdivide = false, want true")
	}
	foundStepReason := false
	for _, r := range iss.Reasons {
		if strings.Contains(r, "expected steps (20) exceeds limit of 15") {
			foundStepReason = true
			break
		}
	}
	if !foundStepReason {
		t.Fatalf("reasons missing expected steps limit: %v", iss.Reasons)
	}
	foundDecomposeAction := false
	for _, a := range iss.RepairActions {
		if strings.Contains(a, "decompose oversized issue") {
			foundDecomposeAction = true
			break
		}
	}
	if !foundDecomposeAction {
		t.Fatalf("repair actions missing decompose: %v", iss.RepairActions)
	}
}

func TestIssueDiscoverabilitySubdivideQueueMultipleInternalPackages(t *testing.T) {
	body := `## Working spine
Monolith refactor

## Parent context
#100

## Core through-line
Change -> real seam -> outcome -> witness.

## Gold-plating boundary
No extras.

## Done condition / witness
Tests pass.
Witness: go test ./...

## Lane
compute

## Likely files
- internal/compute/kernel.go
- internal/model/kv.go
- internal/ctxmmu/mmu.go

## Expected steps
4

- Centrality: Core
- P1 Context: advanced - captures context once
- P2 Net value: preserved - no regression
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: advanced - real path
`
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "feat: cross-cutting monolith refactor",
		"--body", body,
		"--json",
	})
	if code != 3 {
		t.Fatalf("code = %d, want 3; stderr:\n%s", code, stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	if res.Subdivide != 1 {
		t.Fatalf("subdivide count = %d, want 1", res.Subdivide)
	}
	iss := res.Issues[0]
	if len(iss.InternalPackages) != 3 {
		t.Fatalf("internal packages = %v, want 3", iss.InternalPackages)
	}
	foundPkgReason := false
	for _, r := range iss.Reasons {
		if strings.Contains(r, "touches 3 distinct internal packages") {
			foundPkgReason = true
			break
		}
	}
	if !foundPkgReason {
		t.Fatalf("reasons missing 3 packages reason: %v", iss.Reasons)
	}
}

func TestIssueDiscoverabilityTriageQueueMissingSections(t *testing.T) {
	body := `## Core through-line
Change -> seam -> outcome -> witness.

## Gold-plating boundary
No extras.
`
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "feat: incomplete draft",
		"--body", body,
	})
	if code != 3 {
		t.Fatalf("code = %d, want 3", code)
	}
	out := stdout.String()
	if !strings.Contains(out, "Status: NOT DISPATCHABLE") {
		t.Fatalf("output missing NOT DISPATCHABLE:\n%s", out)
	}
	if !strings.Contains(out, "Placement: triage_queue") {
		t.Fatalf("output missing triage_queue placement:\n%s", out)
	}
	if !strings.Contains(out, "Repair Actions:") {
		t.Fatalf("output missing Repair Actions:\n%s", out)
	}
}

func TestIssueDiscoverabilityTriageQueueEmptyTitle(t *testing.T) {
	body := validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)
	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverability(&stdout, &stderr, []string{
		"--title", "   ",
		"--body", body,
		"--json",
	})
	if code != 3 {
		t.Fatalf("code = %d, want 3", code)
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	if res.Triage != 1 {
		t.Fatalf("triage count = %d, want 1", res.Triage)
	}
	iss := res.Issues[0]
	foundEmptyTitle := false
	for _, r := range iss.Reasons {
		if r == "issue title is empty" {
			foundEmptyTitle = true
			break
		}
	}
	if !foundEmptyTitle {
		t.Fatalf("reasons missing empty title: %v", iss.Reasons)
	}
}

func TestIssueDiscoverabilityFileAndFromIssues(t *testing.T) {
	dir := t.TempDir()

	// 1. Markdown file via --body-file / --file
	mdPath := filepath.Join(dir, "draft.md")
	mdContent := "# feat(compute): speedup kernel\n\n" + validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)
	if err := os.WriteFile(mdPath, []byte(mdContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := runIssueDiscoverability(&stdout, &stderr, []string{"--file", mdPath, "--json"}); code != 0 {
		t.Fatalf("file markdown: code=%d stderr=%s", code, stderr.String())
	}
	var res1 IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res1); err != nil {
		t.Fatal(err)
	}
	if !res1.OK || res1.Issues[0].Title != "feat(compute): speedup kernel" {
		t.Fatalf("res1 = %+v", res1)
	}

	// 2. Batch JSON via --from-issues
	jsonPath := filepath.Join(dir, "issues.json")
	issuesPayload := `[
		{
			"number": 101,
			"title": "feat(compute): optimize kernel",
			"body": ` + strconvQuote(validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)) + `
		},
		{
			"number": 102,
			"title": "feat: giant epic",
			"body": "## Working spine\nBig\n## Lane\ncompute\n## Expected steps\n20\n"
		}
	]`
	if err := os.WriteFile(jsonPath, []byte(issuesPayload), 0o644); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code := runIssueDiscoverability(&stdout, &stderr, []string{"--from-issues", jsonPath, "--json"})
	if code != 3 {
		t.Fatalf("from-issues batch code = %d, want 3", code)
	}
	var res2 IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res2); err != nil {
		t.Fatal(err)
	}
	if res2.Total != 2 || res2.Dispatchable != 1 || res2.Subdivide != 1 {
		t.Fatalf("batch counts = %+v", res2)
	}
}

func TestIssueDiscoverabilityIssueFlagWithMockRunner(t *testing.T) {
	issueViewJSON := `{
		"number": 42,
		"title": "feat(compute): speedup kernel",
		"body": ` + strconvQuote(validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)) + `,
		"url": "https://github.com/anthony-chaudhary/fak/issues/42",
		"labels": [{"name": "enhancement"}, {"name": "class:dev"}]
	}`

	runner := func(args []string) (string, string, bool) {
		// verify args
		joined := strings.Join(args, " ")
		if !strings.Contains(joined, "issue view 42") {
			return "", "bad args: " + joined, false
		}
		return issueViewJSON, "", true
	}

	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverabilityWith(&stdout, &stderr, []string{"--issue", "42", "--json"}, runner)
	if code != 0 {
		t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.Issues[0].Number != 42 {
		t.Fatalf("res = %+v", res)
	}
}

func TestIssueDiscoverabilityIssueFlagFetchError(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "", "could not find issue", false
	}

	var stdout, stderr bytes.Buffer
	code := runIssueDiscoverabilityWith(&stdout, &stderr, []string{"--issue", "999"}, runner)
	if code != 1 {
		t.Fatalf("code = %d, want 1; stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "could not find issue") {
		t.Fatalf("stderr missing fetch error message: %s", stderr.String())
	}
}

func TestIssueDiscoverabilityFlagsValidation(t *testing.T) {
	cases := [][]string{
		{}, // missing input
		{"--issue", "1", "--from-issues", "foo.json"}, // conflicting
		{"--body", "a", "--body-file", "b.md"},        // conflicting
		{"--extra", "arg"},                            // unknown flag
	}
	for _, c := range cases {
		var stdout, stderr bytes.Buffer
		code := runIssueDiscoverability(&stdout, &stderr, c)
		if code != 2 {
			t.Fatalf("argv=%v code=%d, want 2", c, code)
		}
	}
}

func TestIssueDiscoverabilityRoutedViaRunIssue(t *testing.T) {
	body := validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 4)
	var stdout, stderr bytes.Buffer
	code := RunIssue(&stdout, &stderr, []string{
		"discoverability",
		"--title", "feat(compute): test routing",
		"--body", body,
		"--json",
	})
	if code != 0 {
		t.Fatalf("RunIssue discoverability code = %d, want 0; stderr: %s", code, stderr.String())
	}
	var res IssueDiscoverabilityAuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.Issues[0].Title != "feat(compute): test routing" {
		t.Fatalf("res = %+v", res)
	}

	// Also audit-discoverability alias
	stdout.Reset()
	stderr.Reset()
	codeAlias := RunIssue(&stdout, &stderr, []string{
		"audit-discoverability",
		"--title", "feat(compute): test routing",
		"--body", body,
		"--json",
	})
	if codeAlias != 0 {
		t.Fatalf("RunIssue audit-discoverability code = %d, want 0; stderr: %s", codeAlias, stderr.String())
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
