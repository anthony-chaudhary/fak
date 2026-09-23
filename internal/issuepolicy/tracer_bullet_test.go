package issuepolicy

import (
	"strings"
	"testing"
)

const horizontalTracerBulletBody = `## Working spine
Define request and response structs plus an interface for future callers.

## Current state
The package has no shared request or response types.

## Why this is next
Later tickets will connect the types to production code.

## Parent context
#100

## Core through-line
Add inert request and response structs and a mock-only interface test.

## Gold-plating boundary
Do not connect the types to a command, engine, or output yet.

## Done condition / witness
The new types compile and their mock test passes.
Witness: go test ./internal/widget/...

## Definition of done
- [ ] request and response structs exist
- [ ] mock interface test passes

## Acceptance gate
go test ./internal/widget/...

## Closure binding
Resolving commit cites the issue and carries (fak widget).

## Lane
widget

## Likely files
- internal/widget/widget_types.go
- internal/widget/interface.go

## Expected steps
3

- Centrality: Core
- P1 Context: preserved - no runtime path changes
- P2 Net value: preserved - types compile
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: preserved - no production entrypoint changes

## Work estimate
Estimate: 2 points

## Overall completion contribution
Contribution: 2/8 points

## Completion standard
demo
`

const verticalTracerBulletBody = `## Working spine
CLI submit command -> widget engine Execute -> JSON receipt written to stdout.

## Current state
The CLI cannot submit widget work to the engine or return a durable receipt.

## Why this is next
Users need one working end-to-end widget path before adding more operations.

## Parent context
#100

## Core through-line
Parse the CLI trigger, call the widget engine Execute method, and emit its receipt as JSON.

## Gold-plating boundary
Do not add additional widget operations.

## Done condition / witness
The CLI invokes the real engine and emits a verifiable JSON receipt.
Witness: go test ./internal/widget/...

## Definition of done
- [ ] CLI trigger reaches the engine
- [ ] engine result is emitted as a JSON receipt

## Acceptance gate
go test ./internal/widget/...

## Closure binding
Resolving commit cites the issue and carries (fak widget).

## Lane
widget

## Likely files
- internal/widget/submit.go
- internal/widget/engine.go
- internal/widget/receipt.go

## Expected steps
4

- Centrality: Core
- P1 Context: advanced - one command carries the request
- P2 Net value: advanced - a real request produces a receipt
- P3 Adaptation: N/A - no adaptive surface
- P4 Operations: advanced - the production entrypoint exposes the path

## Work estimate
Estimate: 3 points

## Overall completion contribution
Contribution: 3/8 points

## Completion standard
demo
`

func TestDiscoverability_TracerBulletEnforcement(t *testing.T) {
	negatedSpineBody := strings.Replace(verticalTracerBulletBody,
		"CLI submit command -> widget engine Execute -> JSON receipt written to stdout.",
		"Do not connect the CLI input trigger to engine execution or emit an output receipt.", 1)
	negatedSpineBody = strings.Replace(negatedSpineBody,
		"Parse the CLI trigger, call the widget engine Execute method, and emit its receipt as JSON.",
		"Do not connect input to the engine and do not produce output or a receipt.", 1)

	dilutedTypeOnlyBody := strings.Replace(verticalTracerBulletBody,
		"- internal/widget/submit.go\n- internal/widget/engine.go\n- internal/widget/receipt.go",
		"- internal/widget/widget_types.go\n- internal/widget/interface.go\n- internal/widget/widget_types_test.go\n- internal/widget/README.md", 1)
	typeOnlyWithDocBody := strings.Replace(verticalTracerBulletBody,
		"- internal/widget/submit.go\n- internal/widget/engine.go\n- internal/widget/receipt.go",
		"- internal/widget/widget_types.go\n- internal/widget/interface.go\n- internal/widget/doc.go", 1)
	typeOnlyWithSchemaBody := strings.Replace(verticalTracerBulletBody,
		"- internal/widget/submit.go\n- internal/widget/engine.go\n- internal/widget/receipt.go",
		"- internal/widget/widget_types.go\n- internal/widget/interface.go\n- internal/widget/schema.go\n- internal/widget/schema_generator.go", 1)
	policyNegationBody := strings.Replace(verticalTracerBulletBody,
		"CLI submit command -> widget engine Execute -> JSON receipt written to stdout.",
		"CLI does not bypass policy, then invokes engine and emits receipt.", 1)
	policyNegationBody = strings.Replace(policyNegationBody,
		"Parse the CLI trigger, call the widget engine Execute method, and emit its receipt as JSON.",
		"CLI does not bypass policy, then invokes engine and emits receipt.", 1)

	tests := []struct {
		name          string
		title         string
		labels        []string
		body          string
		targetPrivate bool
		wantOK        bool
		wantReason    string
	}{
		{
			name:       "horizontal feature is rejected",
			title:      "feat(widget): add request types",
			body:       horizontalTracerBulletBody,
			wantReason: "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:   "vertical feature is admitted",
			title:  "feat(widget): submit work end to end",
			body:   verticalTracerBulletBody,
			wantOK: true,
		},
		{
			name:  "feature fragmentation spike is rejected",
			title: "feat(widget): split widget delivery",
			body: verticalTracerBulletBody + `

## Planned sub-tickets
1. TICKET-01A: add request types
2. TICKET-01B: add engine interface
3. TICKET-01C: add command adapter
4. TICKET-01D: add receipt writer
`,
			wantReason: "ISSUE_FRAGMENTATION_SPIKE",
		},
		{
			name:          "private target horizontal feature is rejected",
			title:         "feat(widget): add private request types",
			body:          horizontalTracerBulletBody,
			targetPrivate: true,
			wantReason:    "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:       "negated stage names do not form a tracer bullet",
			title:      "feat(widget): defer runtime connection",
			body:       negatedSpineBody,
			wantReason: "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:       "docs and tests do not dilute type only paths",
			title:      "feat(widget): add request contract types",
			body:       dilutedTypeOnlyBody,
			wantReason: "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:       "doc go does not dilute type only paths",
			title:      "feat(widget): add documented contract types",
			body:       typeOnlyWithDocBody,
			wantReason: "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:       "schema generator does not dilute type only paths",
			title:      "feat(widget): generate contract schema",
			body:       typeOnlyWithSchemaBody,
			wantReason: "ISSUE_HORIZONTAL_FRAGMENT",
		},
		{
			name:  "parenthesized fragments combine across heading aliases",
			title: "feat(widget): split planned and follow up work",
			body: verticalTracerBulletBody + `

## Planned tickets
1) TICKET-01A: add request types
2) TICKET-01B: add engine interface

## Follow-up issues
1) TICKET-01C: add command adapter
2) TICKET-01D: add receipt writer
`,
			wantReason: "ISSUE_FRAGMENTATION_SPIKE",
		},
		{
			name:   "policy negation does not negate positive execution stages",
			title:  "feat(widget): enforce policy in the vertical path",
			body:   policyNegationBody,
			wantOK: true,
		},
		{
			name:  "fragment count combines relevant headings",
			title: "feat(widget): split widget delivery across headings",
			body: verticalTracerBulletBody + `

## Child issues
- TICKET-01A: add request types
- TICKET-01B: add engine interface

## Planned sub-tickets
- TICKET-01C: add command adapter
- TICKET-01D: add receipt writer
`,
			wantReason: "ISSUE_FRAGMENTATION_SPIKE",
		},
		{
			name:   "bug metadata wins over feature metadata",
			title:  "feat(widget): restore request compatibility",
			labels: []string{"type:feat", "type:bug"},
			body:   horizontalTracerBulletBody,
			wantOK: true,
		},
		{
			name:   "docs metadata wins over feature metadata",
			title:  "feat(widget): document request compatibility",
			labels: []string{"type:feat", "type:docs"},
			body:   horizontalTracerBulletBody,
			wantOK: true,
		},
		{
			name:   "bug is exempt",
			title:  "fix(widget): restore request compatibility",
			labels: []string{"type:bug"},
			body:   horizontalTracerBulletBody,
			wantOK: true,
		},
		{
			name:   "docs are exempt",
			title:  "docs(widget): document request types",
			labels: []string{"type:docs"},
			body:   horizontalTracerBulletBody,
			wantOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			draft := IssueDraft{
				Title: tt.title,
				Body:  tt.body,
			}
			for _, label := range tt.labels {
				draft.Labels = append(draft.Labels, IssueLabel{Name: label})
			}
			review := ReviewIssueDraft(draft, Options{TargetPrivate: tt.targetPrivate})

			if review.OK != tt.wantOK {
				t.Fatalf("review.OK = %v, want %v; verdict=%q dispatchability=%q reasons=%v", review.OK, tt.wantOK, review.Verdict, review.Dispatchability, review.Reasons)
			}
			if tt.wantOK && review.Dispatchability != Dispatchable {
				t.Fatalf("dispatchability = %q, want %q; reasons=%v", review.Dispatchability, Dispatchable, review.Reasons)
			}
			if tt.wantReason != "" && !tracerBulletTestContains(review.Reasons, tt.wantReason) {
				t.Fatalf("reasons = %v, want %q", review.Reasons, tt.wantReason)
			}
		})
	}
}

func tracerBulletTestContains(values []string, want string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), want) {
			return true
		}
	}
	return false
}
