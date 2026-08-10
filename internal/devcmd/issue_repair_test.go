package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func safeTemplateDraft(num int) issuecontract.IssueDraft {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=second-next; title=...}.gen)",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"- Parent: #1625",
		"",
		"## Why",
		"The generated body below the corrupt header is intact.",
		"",
		"## Initial scope",
		"Repair only the generated metadata header.",
	}, "\n")
	return issuecontract.IssueDraft{
		Number: num,
		Title:  "generation(second-next): build the optimizer",
		Body:   body,
		Labels: []issuecontract.IssueLabel{{Name: "generation"}, {Name: "gen/second-next"}},
	}
}

func unsafeTemplateProseDraft(num int) issuecontract.IssueDraft {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=x}.gen)",
		"An operator note wedged into the header block, not a bullet.",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"",
		"## Why",
		"Intact body.",
	}, "\n")
	return issuecontract.IssueDraft{
		Number: num, Title: "generation(x): thing", Body: body,
		Labels: []issuecontract.IssueLabel{{Name: "generation"}},
	}
}

func decodeRepair(t *testing.T, b []byte) issueRepairResult {
	t.Helper()
	var r issueRepairResult
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("decode issueRepairResult: %v\n%s", err, b)
	}
	return r
}

func repairRow(r issueRepairResult, num int) (issueRepairRow, bool) {
	for _, row := range r.Rows {
		if row.Number == num {
			return row, true
		}
	}
	return issueRepairRow{}, false
}

// recordingRunner captures gh argv and, for --body-file calls, the file content
// at call time (the temp file is removed once the apply returns).
func recordingRunner(calls *[][]string, bodies *[]string, url string, ok bool) issueCreateRunner {
	return func(args []string) (string, string, bool) {
		*calls = append(*calls, append([]string(nil), args...))
		for i := 0; i+1 < len(args); i++ {
			if args[i] == "--body-file" {
				b, _ := os.ReadFile(args[i+1])
				*bodies = append(*bodies, string(b))
			}
		}
		if !ok {
			return "", "gh boom", false
		}
		return url, "", true
	}
}

func TestIssueRepairDryRunPlanClassifiesWithoutWriting(t *testing.T) {
	issues := []issuecontract.IssueDraft{
		safeTemplateDraft(1727),
		unsafeTemplateProseDraft(9),
		{Number: 1207, Title: "a bare issue with no scope fields"},
	}
	var calls [][]string
	var bodies []string
	var out, errb bytes.Buffer
	code := runIssueRepairWith(&out, &errb, []string{"--json"}, issues, recordingRunner(&calls, &bodies, "", true))
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(calls) != 0 {
		t.Fatalf("dry-run must not write, got %v", calls)
	}
	r := decodeRepair(t, out.Bytes())
	if r.Live {
		t.Fatal("result.Live should be false for a dry-run")
	}
	// Safe template row -> auto-apply with a marker-free proposed body.
	if row, ok := repairRow(r, 1727); !ok {
		t.Fatal("missing row for safe template issue 1727")
	} else {
		// A corrupt-header issue is also scope-incomplete, so its PRIMARY kind is
		// scope; the template auto-apply keys on Kinds CONTAINING template.
		if !repairHasKind(row.Kinds, "template") {
			t.Fatalf("1727 kinds %v should include template", row.Kinds)
		}
		if row.Disposition != repairDispAutoApply {
			t.Fatalf("1727 = %+v, want auto-apply", row)
		}
		if row.Applied {
			t.Fatal("1727 must not be applied on a dry-run")
		}
		if strings.Contains(row.ProposedBody, "$(") || !strings.Contains(row.ProposedBody, "## Why") {
			t.Fatalf("1727 proposed_body wrong:\n%s", row.ProposedBody)
		}
	}
	// Unsafe template row -> propose-only with a fail-closed reason.
	if row, ok := repairRow(r, 9); !ok {
		t.Fatal("missing row for unsafe template issue 9")
	} else if row.Disposition != repairDispProposeOnly || row.Unsafe == "" {
		t.Fatalf("9 = %+v, want propose-only with an unsafe reason", row)
	}
	// Bare issue -> a non-template repairable row, propose-only, never written.
	if row, ok := repairRow(r, 1207); !ok {
		t.Fatal("missing row for bare issue 1207 (expected the default reviewer to flag it)")
	} else if row.Disposition != repairDispProposeOnly {
		t.Fatalf("1207 = %+v, want propose-only", row)
	}
	if r.Counts.AutoApply != 1 {
		t.Fatalf("counts.AutoApply = %d, want 1", r.Counts.AutoApply)
	}
}

func TestIssueRepairLiveAppliesSafeTemplateBody(t *testing.T) {
	issues := []issuecontract.IssueDraft{safeTemplateDraft(1727)}
	var calls [][]string
	var bodies []string
	var out, errb bytes.Buffer
	code := runIssueRepairWith(&out, &errb,
		[]string{"--live", "--max-apply", "5", "--json"}, issues,
		recordingRunner(&calls, &bodies, "https://x/1727", true))
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(calls) != 1 {
		t.Fatalf("expected one gh edit, got %v", calls)
	}
	got := strings.Join(calls[0][:4], " ")
	if got != "issue edit 1727 --body-file" {
		t.Fatalf("gh argv prefix = %q, want `issue edit 1727 --body-file`", got)
	}
	if len(bodies) != 1 || strings.Contains(bodies[0], "$(") || !strings.Contains(bodies[0], "## Generation stream") {
		t.Fatalf("written body wrong: %q", bodies)
	}
	r := decodeRepair(t, out.Bytes())
	if !r.Live || r.Counts.Applied != 1 {
		t.Fatalf("counts = %+v", r.Counts)
	}
	if row, _ := repairRow(r, 1727); !row.Applied || row.URL != "https://x/1727" {
		t.Fatalf("1727 row not marked applied: %+v", row)
	}
}

func TestIssueRepairLiveRefusesOverMaxApply(t *testing.T) {
	issues := []issuecontract.IssueDraft{safeTemplateDraft(1727), safeTemplateDraft(1728)}
	var calls [][]string
	var bodies []string
	var out, errb bytes.Buffer
	code := runIssueRepairWith(&out, &errb,
		[]string{"--live", "--max-apply", "1"}, issues,
		recordingRunner(&calls, &bodies, "", true))
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (blast-radius refusal); stderr=%s", code, errb.String())
	}
	if len(calls) != 0 {
		t.Fatalf("fuse must write nothing, got %v", calls)
	}
	if !strings.Contains(errb.String(), "max-apply") {
		t.Fatalf("stderr should explain the fuse: %s", errb.String())
	}
}

func TestIssueRepairKindFilter(t *testing.T) {
	issues := []issuecontract.IssueDraft{safeTemplateDraft(1727)}
	// Filtering to a kind the row does NOT carry excludes it entirely (the row's
	// kinds are scope/route/template, so "noise" matches none of them).
	var out, errb bytes.Buffer
	if code := runIssueRepairWith(&out, &errb, []string{"--json", "--kind", "noise"}, issues, (&captureGH{}).run); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if r := decodeRepair(t, out.Bytes()); len(r.Rows) != 0 {
		t.Fatalf("--kind noise should exclude the row, got %+v", r.Rows)
	}
	// Filtering to its own kind includes it.
	out.Reset()
	errb.Reset()
	if code := runIssueRepairWith(&out, &errb, []string{"--json", "--kind", "template"}, issues, (&captureGH{}).run); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if r := decodeRepair(t, out.Bytes()); len(r.Rows) != 1 || r.Rows[0].Number != 1727 {
		t.Fatalf("--kind template should include the row, got %+v", r.Rows)
	}
}

func TestRepairKindDispositionMapping(t *testing.T) {
	want := map[string]string{
		"scope":   repairDispProposeOnly,
		"noise":   repairDispProposeOnly,
		"route":   repairDispProposeOnly,
		"private": repairDispRefuse,
		"split":   repairDispDefer,
	}
	for kind, disp := range want {
		if got := repairKindDisposition[kind]; got != disp {
			t.Fatalf("repairKindDisposition[%q] = %q, want %q", kind, got, disp)
		}
	}
	if !scaffoldKind("scope") || scaffoldKind("template") || scaffoldKind("route") {
		t.Fatal("scaffoldKind classification wrong")
	}
}
