package epicprogress

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

var (
	benchCountsSink EpicCounts
	benchItemsSink  []TaskListItem
	benchIntSink    int
	benchMapSink    map[int]bool
)

const sampleEpicBody = `# Epic 5000: Next Generation Infrastructure

This epic tracks delivery of core infrastructure capabilities.
See architecture doc for details: docs/arch.md#overview

## Keystone deliverables
- [ ] #6001 Core engine integration
- [x] #6002 Initial protocol parser
- [ ] **Phase 1** #6003 — telemetry pipeline
- [x] plain checkbox without issue reference
- [ ] #6004 State reconciliation daemon
- [ ] note: sha1#9 is not an issue reference
- [x] #6005 Safe memory journal implementation

## Follow-on tasks
- [ ] #6006 External adapter bridge
- [x] #6007 Benchmarking and test harness
- [ ] #6008 Production deployment checklist
- [ ] another item without ref
- [x] #6009 Documentation and runbooks
- [ ] #6010 Rollback automation and canary gates

Related issues: #9999 (informational only, not a task row)
`

// TestBenchmarkSanity ensures that all benchmarked fixtures and operations execute
// cleanly with expected outputs during standard test execution.
func TestBenchmarkSanity(t *testing.T) {
	items := ParseTaskList(sampleEpicBody)
	if len(items) != 13 {
		t.Fatalf("ParseTaskList returned %d items, want 13", len(items))
	}
	total, checked := CountTaskList(sampleEpicBody)
	if total != 13 || checked != 5 {
		t.Fatalf("CountTaskList = %d checked / %d total, want 5/13", checked, total)
	}

	// Sanity check label runner
	labelPayload := makeBenchmarkLabelPayload(50)
	labelRunner := fakeRunner{
		"--label track-bench": {out: labelPayload, ok: true},
	}
	cLabel := Counts(labelRunner.run, "", EpicSpec{Number: 1000, Title: "bench-label", Label: "track-bench"})
	if cLabel.Err != "" || cLabel.Source != "label" || cLabel.Total != 50 || cLabel.Closed != 25 {
		t.Fatalf("Counts(label) failed sanity check: %+v", cLabel)
	}

	// Sanity check checklist runner
	checklistRunner := makeBenchmarkChecklistRunner(sampleEpicBody)
	cChecklist := Counts(checklistRunner.run, "", EpicSpec{Number: 5000, Title: "bench-checklist"})
	if cChecklist.Err != "" || cChecklist.Source != SourceChecklistIssueState || cChecklist.Total != 13 {
		t.Fatalf("Counts(checklist) failed sanity check: %+v", cChecklist)
	}
}

func makeBenchmarkLabelPayload(n int) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i := 1; i <= n; i++ {
		if i > 1 {
			sb.WriteString(",")
		}
		state := "OPEN"
		if i%2 == 0 {
			state = "CLOSED"
		}
		fmt.Fprintf(&sb, `{"number":%d,"state":%q}`, 1000+i, state)
	}
	sb.WriteString("]")
	return sb.String()
}

func makeBenchmarkChecklistRunner(body string) fakeRunner {
	bodyJSON, _ := json.Marshal(struct {
		Body string `json:"body"`
	}{Body: body})
	runner := fakeRunner{
		"view 5000": {out: string(bodyJSON), ok: true},
	}
	for i := 1; i <= 10; i++ {
		num := 6000 + i
		state := "OPEN"
		if i%2 == 0 {
			state = "CLOSED"
		}
		runner[fmt.Sprintf("view %d", num)] = struct {
			out string
			ok  bool
		}{out: fmt.Sprintf(`{"state":%q}`, state), ok: true}
	}
	return runner
}

// BenchmarkCountsByLabel measures resolving an epic with 50 child issues via its track label.
func BenchmarkCountsByLabel(b *testing.B) {
	payload := makeBenchmarkLabelPayload(50)
	fake := fakeRunner{
		"--label track-bench": {out: payload, ok: true},
	}
	spec := EpicSpec{
		Number: 1000,
		Title:  "benchmark-epic-label",
		Label:  "track-bench",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCountsSink = Counts(fake.run, "", spec)
	}
}

// BenchmarkCountsByChecklist measures resolving an epic via its body checklist with child issue cross-checking.
func BenchmarkCountsByChecklist(b *testing.B) {
	fake := makeBenchmarkChecklistRunner(sampleEpicBody)
	spec := EpicSpec{
		Number: 5000,
		Title:  "benchmark-epic-checklist",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCountsSink = Counts(fake.run, "", spec)
	}
}

// BenchmarkCountsByChecklistPlain measures checklist resolution when rows carry checkboxes but no issue references.
func BenchmarkCountsByChecklistPlain(b *testing.B) {
	plainBody := "- [ ] Task 1\n- [x] Task 2\n- [ ] Task 3\n- [x] Task 4\n- [ ] Task 5\n- [x] Task 6\n"
	plainJSON, _ := json.Marshal(struct {
		Body string `json:"body"`
	}{Body: plainBody})
	fake := fakeRunner{
		"view 5100": {out: string(plainJSON), ok: true},
	}
	spec := EpicSpec{
		Number: 5100,
		Title:  "benchmark-epic-checklist-plain",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCountsSink = Counts(fake.run, "", spec)
	}
}

// BenchmarkParseTaskList measures parsing GitHub task list markdown into structured items.
func BenchmarkParseTaskList(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchItemsSink = ParseTaskList(sampleEpicBody)
	}
}

// BenchmarkCountTaskList measures the checkbox-only counter over markdown body.
func BenchmarkCountTaskList(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total, checked := CountTaskList(sampleEpicBody)
		benchIntSink = total + checked
	}
}

// BenchmarkChildStates measures concurrent live issue state resolution across workers.
func BenchmarkChildStates(b *testing.B) {
	fake := makeBenchmarkChecklistRunner(sampleEpicBody)
	items := ParseTaskList(sampleEpicBody)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMapSink = childStates(fake.run, "", 5000, items)
	}
}
