package issuepolicy

import (
	"fmt"
	"strings"
	"testing"
)

func TestMarkdownSectionsCanonicalAliases(t *testing.T) {
	body := strings.Join([]string{
		"## Motivation",
		"Needs improvement.",
		"## Requirements",
		"Must handle edge cases.",
		"## Acceptance criteria",
		"All tests pass.",
		"## Files",
		"- internal/gateway/server.go",
		"## Value frame",
		"- Today: Slow performance",
		"- Done when: 2x faster",
		"- Witness: go test ./cmd/fak",
	}, "\n")

	sections := markdownSections(body)

	// Check canonical mappings
	if got := sections["current state"]; got != "Needs improvement." {
		t.Fatalf("current state = %q, want %q", got, "Needs improvement.")
	}
	if got := sections["motivation"]; got != "Needs improvement." {
		t.Fatalf("motivation = %q, want %q", got, "Needs improvement.")
	}
	if got := sections["scope"]; got != "Must handle edge cases." {
		t.Fatalf("scope = %q, want %q", got, "Must handle edge cases.")
	}
	if got := sections["requirements"]; got != "Must handle edge cases." {
		t.Fatalf("requirements = %q, want %q", got, "Must handle edge cases.")
	}
	if got := sections["done condition"]; got != "All tests pass." {
		t.Fatalf("done condition = %q, want %q", got, "All tests pass.")
	}
	if got := sections["acceptance criteria"]; got != "All tests pass." {
		t.Fatalf("acceptance criteria = %q, want %q", got, "All tests pass.")
	}
	if got := sections["likely files"]; got != "- internal/gateway/server.go" {
		t.Fatalf("likely files = %q, want %q", got, "- internal/gateway/server.go")
	}
	if got := sections["files"]; got != "- internal/gateway/server.go" {
		t.Fatalf("files = %q, want %q", got, "- internal/gateway/server.go")
	}
	if !strings.Contains(sections["value"], "Today: Slow performance") {
		t.Fatalf("value section missing content: %q", sections["value"])
	}
	if !strings.Contains(sections["value frame"], "Today: Slow performance") {
		t.Fatalf("value frame section missing content: %q", sections["value frame"])
	}
}

func TestExtractBulletedField(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		fieldName string
		want      string
	}{
		{
			name:      "witness with colon inside bold",
			text:      "- **Witness:** go test ./cmd/fak",
			fieldName: "witness",
			want:      "go test ./cmd/fak",
		},
		{
			name:      "witness with asterisk bullet",
			text:      "* Witness: go test ./cmd/fak",
			fieldName: "witness",
			want:      "go test ./cmd/fak",
		},
		{
			name:      "today bullet",
			text:      "- Today: feature broken",
			fieldName: "today",
			want:      "feature broken",
		},
		{
			name:      "done when bold",
			text:      "- **Done when:** criteria met",
			fieldName: "done when",
			want:      "criteria met",
		},
		{
			name:      "acceptance criteria bullet",
			text:      "- Acceptance criteria: all pass",
			fieldName: "acceptance criteria",
			want:      "all pass",
		},
		{
			name:      "colon outside bold",
			text:      "* **Witness**: cmd line",
			fieldName: "witness",
			want:      "cmd line",
		},
		{
			name:      "bold enclosing key and colon",
			text:      "- **Done when: criteria met**",
			fieldName: "done when",
			want:      "criteria met",
		},
		{
			name:      "backticks in value",
			text:      "- **Witness:** `go test ./...`",
			fieldName: "witness",
			want:      "go test ./...",
		},
		{
			name:      "indented bullet",
			text:      "  - **Witness:** cmd",
			fieldName: "witness",
			want:      "cmd",
		},
		{
			name:      "missing field",
			text:      "- Other: foo",
			fieldName: "witness",
			want:      "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := extractBulletedField(tc.text, tc.fieldName)
			if got != tc.want {
				t.Fatalf("extractBulletedField(%q, %q) = %q, want %q", tc.text, tc.fieldName, got, tc.want)
			}
		})
	}
}

func TestIssueDraftPathsSplitsWordsInListItems(t *testing.T) {
	section := strings.Join([]string{
		"- internal/pkg/file.go - description of file",
		"- `internal/pkg/other.go` (main entrypoint)",
		"- internal/pkg/third.go: additional handler",
	}, "\n")

	paths := issueDraftPaths(section)
	want := []string{
		"internal/pkg/file.go",
		"internal/pkg/other.go",
		"internal/pkg/third.go",
	}

	if len(paths) != len(want) {
		t.Fatalf("paths = %+v, want %+v", paths, want)
	}
	for i := range want {
		if paths[i] != want[i] {
			t.Fatalf("paths[%d] = %q, want %q", i, paths[i], want[i])
		}
	}
}

func TestExtractBodyGoPathsFallback(t *testing.T) {
	body := `Please inspect ` + "`internal/gateway/proxy.go`" + ` and "internal/gateway/server.go" for fixes.`
	paths := extractBodyGoPaths(body)
	want := []string{
		"internal/gateway/proxy.go",
		"internal/gateway/server.go",
	}
	if len(paths) != len(want) {
		t.Fatalf("paths = %+v, want %+v", paths, want)
	}
	for i := range want {
		if paths[i] != want[i] {
			t.Fatalf("paths[%d] = %q, want %q", i, paths[i], want[i])
		}
	}

	// Verify CandidateFromIssueDraft falls back to body Go paths
	draft := IssueDraft{
		Title: "gateway: fix proxy",
		Body: strings.Join([]string{
			"## Current state",
			"The proxy fails.",
			"## Scope",
			"Fix the proxy.",
			"## Done condition",
			"Proxy passes.",
			"## Witness",
			"go test ./internal/gateway/...",
			"## Details",
			"See `internal/gateway/proxy.go` for details.",
		}, "\n"),
	}
	c := CandidateFromIssueDraft(draft)
	if len(c.Paths) != 1 || c.Paths[0] != "internal/gateway/proxy.go" {
		t.Fatalf("c.Paths = %+v, want [internal/gateway/proxy.go]", c.Paths)
	}
}

func TestTitleLaneInference(t *testing.T) {
	// 1. Conventional commit feat(agent): ...
	c1 := CandidateFromIssueDraft(IssueDraft{
		Title: "feat(agent): support new commands",
		Body:  "## Scope\nAgent work.",
	})
	if c1.Lane != "agent" {
		t.Fatalf("lane = %q, want agent", c1.Lane)
	}

	// 2. Bracket style [model] ...
	c2 := CandidateFromIssueDraft(IssueDraft{
		Title: "[model] load gguf weights",
		Body:  "## Scope\nModel work.",
	})
	if c2.Lane != "model" {
		t.Fatalf("lane = %q, want model", c2.Lane)
	}

	// 3. Infer from first internal/<pkg> path when title has no prefix
	c3 := CandidateFromIssueDraft(IssueDraft{
		Title: "refactor proxy logic",
		Body: strings.Join([]string{
			"## Files",
			"- internal/gateway/proxy.go",
		}, "\n"),
	})
	if c3.Lane != "gateway" {
		t.Fatalf("lane = %q, want gateway", c3.Lane)
	}
}

func TestBulletedWitnessAndTodayInValueFrame(t *testing.T) {
	draft := IssueDraft{
		Number: 2001,
		Title:  "feat(gateway): optimize proxy throughput",
		Body: strings.Join([]string{
			"## Value frame",
			"- **Today:** Proxy throughput is 100 req/s.",
			"- **Done when:** Proxy throughput exceeds 200 req/s.",
			"- **Witness:** `go test ./internal/gateway/... -bench=.`",
			"## Scope",
			"Optimize the buffer pooling in `internal/gateway/buffer.go`.",
		}, "\n"),
	}

	c := CandidateFromIssueDraft(draft)
	if c.CurrentState != "Proxy throughput is 100 req/s." {
		t.Fatalf("CurrentState = %q, want %q", c.CurrentState, "Proxy throughput is 100 req/s.")
	}
	if c.DoneCondition != "Proxy throughput exceeds 200 req/s." {
		t.Fatalf("DoneCondition = %q, want %q", c.DoneCondition, "Proxy throughput exceeds 200 req/s.")
	}
	if c.Witness != "go test ./internal/gateway/... -bench=." {
		t.Fatalf("Witness = %q, want %q", c.Witness, "go test ./internal/gateway/... -bench=.")
	}
	if c.Lane != "gateway" {
		t.Fatalf("Lane = %q, want gateway", c.Lane)
	}
	if len(c.Paths) != 1 || c.Paths[0] != "internal/gateway/buffer.go" {
		t.Fatalf("Paths = %+v, want [internal/gateway/buffer.go]", c.Paths)
	}

	review := ReviewIssueDraft(draft, Options{})
	if !review.OK || review.Dispatchability != Dispatchable {
		t.Fatalf("review = %+v, want OK and dispatchable", review)
	}
}

func TestFiledIssueCeremonyFieldDefaulting(t *testing.T) {
	draft := IssueDraft{
		Number: 3001,
		Title:  "feat(model): parse tensor metadata",
		Body: strings.Join([]string{
			"## Motivation",
			"Model loading requires tensor metadata.",
			"## Requirements",
			"Parse metadata headers in `internal/model/tensor.go`.",
			"## Acceptance criteria",
			"Tensor metadata parsed without errors.",
			"## Witness",
			"go test ./internal/model/...",
		}, "\n"),
	}

	c := CandidateFromIssueDraft(draft)

	// Verify ceremony fields were defaulted
	if c.ParentRef != "#3001" {
		t.Fatalf("ParentRef = %q, want #3001", c.ParentRef)
	}
	if c.WhyNow != "Unblocks #3001" {
		t.Fatalf("WhyNow = %q, want Unblocks #3001", c.WhyNow)
	}
	if c.WorkingSpine != draft.Title {
		t.Fatalf("WorkingSpine = %q, want %q", c.WorkingSpine, draft.Title)
	}
	if c.OutOfScope != "Not specified (defer gold-plating)" {
		t.Fatalf("OutOfScope = %q, want Not specified (defer gold-plating)", c.OutOfScope)
	}
	if c.AcceptanceGate != c.Witness {
		t.Fatalf("AcceptanceGate = %q, want %q", c.AcceptanceGate, c.Witness)
	}
	wantClosure := fmt.Sprintf("Resolving commit cites #3001 and carries `(fak model)`")
	if c.ClosureBinding != wantClosure {
		t.Fatalf("ClosureBinding = %q, want %q", c.ClosureBinding, wantClosure)
	}

	review := ReviewIssueDraft(draft, Options{})
	if !review.OK || review.Dispatchability != Dispatchable {
		t.Fatalf("review = %+v, want OK and dispatchable", review)
	}
}

func TestVagueIssue1441RemainsFlagged(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 1441,
		Title:  "make it better",
		Body: strings.Join([]string{
			"### Current state",
			"The feature exists.",
			"### In scope",
			"Improve things.",
		}, "\n"),
	}, Options{})
	if review.OK || review.Dispatchability != TriageOnly {
		t.Fatalf("review = %+v, want triage-only incomplete issue", review)
	}
	for _, want := range []string{
		"parent_ref",
		"why_now",
		"working_spine",
		"out_of_scope",
		"done_condition",
		"witness",
		"acceptance_gate",
		"closure_binding",
	} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent from %+v", want, review.MissingFields)
		}
	}
	if !has(review.Reasons, ReasonUnrouted) {
		t.Fatalf("unrouted reason absent: %+v", review.Reasons)
	}
}

func TestCanonicalHeadingFuzzyMatching(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		// scope
		{"core requirements", "scope"},
		{"technical proposal", "scope"},
		{"key deliverables", "scope"},
		{"execution plan", "scope"},
		{"technical spec", "scope"},
		// done condition
		{"done condition", "done condition"},
		{"acceptance criteria", "done condition"},
		{"done when", "done condition"},
		{"project dod", "done condition"},
		// current state
		{"current state", "current state"},
		{"today's behavior", "current state"},
		{"system baseline", "current state"},
		{"motivation", "current state"},
		// likely files
		{"path hint", "likely files"},
		{"touched paths", "likely files"},
		{"source files", "likely files"},
		{"likely file", "likely files"},
		{"file scope", "likely files"},
		// value
		{"value frame", "value"},
		{"problem frame", "value"},
		{"frame", "value"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			got := canonicalHeading(tc.input)
			if got != tc.want {
				t.Fatalf("canonicalHeading(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestMarkdownSectionsProblemPopulatesCurrentState(t *testing.T) {
	// 1. ## Problem populates current state when current state is empty
	body1 := strings.Join([]string{
		"## Problem",
		"Memory leak in buffer pool.",
		"## Scope",
		"Fix buffer pool leaks.",
	}, "\n")
	sections1 := markdownSections(body1)
	if got := sections1["current state"]; got != "Memory leak in buffer pool." {
		t.Fatalf("current state = %q, want %q", got, "Memory leak in buffer pool.")
	}
	if got := sections1["problem"]; got != "Memory leak in buffer pool." {
		t.Fatalf("problem = %q, want %q", got, "Memory leak in buffer pool.")
	}

	// 2. ## Current state is not overwritten by ## Problem
	body2 := strings.Join([]string{
		"## Current state",
		"Existing state description.",
		"## Problem",
		"Problem description.",
	}, "\n")
	sections2 := markdownSections(body2)
	if got := sections2["current state"]; got != "Existing state description." {
		t.Fatalf("current state = %q, want %q", got, "Existing state description.")
	}
}

func TestCandidateFieldFallbacks(t *testing.T) {
	// 1. InScope fallback to BetterBecause
	d1 := IssueDraft{
		Title: "gateway: optimize buffer pool",
		Body: strings.Join([]string{
			"## Better because",
			"Reduces allocation latency by 50%.",
		}, "\n"),
	}
	c1 := CandidateFromIssueDraft(d1)
	if c1.InScope != "Reduces allocation latency by 50%." {
		t.Fatalf("c1.InScope = %q, want %q", c1.InScope, "Reduces allocation latency by 50%.")
	}

	// 2. InScope fallback to Title for fix(...)
	d2 := IssueDraft{
		Title: "fix(gateway): buffer pool memory leak",
		Body: strings.Join([]string{
			"## Current state",
			"Buffer pool leaks on restart.",
		}, "\n"),
	}
	c2 := CandidateFromIssueDraft(d2)
	if c2.InScope != "fix(gateway): buffer pool memory leak" {
		t.Fatalf("c2.InScope = %q, want %q", c2.InScope, "fix(gateway): buffer pool memory leak")
	}

	// 3. InScope fallback to Title for bug label
	d3 := IssueDraft{
		Title:  "buffer pool memory leak",
		Labels: []IssueLabel{{Name: "bug"}},
		Body: strings.Join([]string{
			"## Current state",
			"Buffer pool leaks on restart.",
		}, "\n"),
	}
	c3 := CandidateFromIssueDraft(d3)
	if c3.InScope != "buffer pool memory leak" {
		t.Fatalf("c3.InScope = %q, want %q", c3.InScope, "buffer pool memory leak")
	}

	// 4. DoneCondition fallback to BetterBecause when Witness is present
	d4 := IssueDraft{
		Title: "gateway: buffer optimization",
		Body: strings.Join([]string{
			"## Better because",
			"Throughput increases 2x.",
			"## Witness",
			"go test ./internal/gateway/...",
		}, "\n"),
	}
	c4 := CandidateFromIssueDraft(d4)
	if c4.DoneCondition != "Throughput increases 2x." {
		t.Fatalf("c4.DoneCondition = %q, want %q", c4.DoneCondition, "Throughput increases 2x.")
	}

	// 5. DoneCondition fallback to Defect resolved... when title starts with fix(
	d5 := IssueDraft{
		Title: "fix(gateway): resolve deadlock",
		Body: strings.Join([]string{
			"## Current state",
			"Deadlock occurs on shutdown.",
			"## Witness",
			"go test ./internal/gateway -run TestShutdown",
		}, "\n"),
	}
	c5 := CandidateFromIssueDraft(d5)
	wantDone5 := "Defect resolved and witness passes: go test ./internal/gateway -run TestShutdown"
	if c5.DoneCondition != wantDone5 {
		t.Fatalf("c5.DoneCondition = %q, want %q", c5.DoneCondition, wantDone5)
	}

	// 6. DoneCondition fallback to Defect resolved... when issue has bug label
	d6 := IssueDraft{
		Title:  "gateway: deadlock on shutdown",
		Labels: []IssueLabel{{Name: "bug"}},
		Body: strings.Join([]string{
			"## Current state",
			"Deadlock occurs on shutdown.",
			"## Witness",
			"go test ./internal/gateway -run TestShutdown",
		}, "\n"),
	}
	c6 := CandidateFromIssueDraft(d6)
	if c6.DoneCondition != wantDone5 {
		t.Fatalf("c6.DoneCondition = %q, want %q", c6.DoneCondition, wantDone5)
	}

	// 7. CurrentState fallback to section("Problem")
	d7 := IssueDraft{
		Title: "gateway: fix leak",
		Body: strings.Join([]string{
			"## Problem",
			"Memory leak in gateway server.",
			"## Scope",
			"Fix the leak.",
		}, "\n"),
	}
	c7 := CandidateFromIssueDraft(d7)
	if c7.CurrentState != "Memory leak in gateway server." {
		t.Fatalf("c7.CurrentState = %q, want %q", c7.CurrentState, "Memory leak in gateway server.")
	}

	// 8. CurrentState fallback to extractBulletedField(..., "problem")
	d8 := IssueDraft{
		Title: "gateway: fix leak",
		Body: strings.Join([]string{
			"## Value",
			"- Problem: Latency spikes on heavy load.",
			"## Scope",
			"Fix the spike.",
		}, "\n"),
	}
	c8 := CandidateFromIssueDraft(d8)
	if c8.CurrentState != "Latency spikes on heavy load." {
		t.Fatalf("c8.CurrentState = %q, want %q", c8.CurrentState, "Latency spikes on heavy load.")
	}
}
