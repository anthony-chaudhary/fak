package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// captureGH is a fake issueCreateRunner: it records every gh argv it is handed
// and returns a canned result, so a test asserts on the built argv without ever
// invoking real gh.
type captureGH struct {
	calls [][]string
	out   string
	fail  bool
}

func (c *captureGH) run(args []string) (string, string, bool) {
	dup := append([]string(nil), args...)
	c.calls = append(c.calls, dup)
	if c.fail {
		return "", "gh boom", false
	}
	return c.out, "", true
}

func decodeEdit(t *testing.T, b []byte) issueEditResult {
	t.Helper()
	var r issueEditResult
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("decode issueEditResult: %v\n%s", err, b)
	}
	return r
}

func TestIssueEditDryRunRendersArgvWithoutCallingGH(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{out: "https://x/42"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "42", "--title", "New title", "--dry-run", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(gh.calls) != 0 {
		t.Fatalf("dry-run must not call gh, got %v", gh.calls)
	}
	r := decodeEdit(t, out.Bytes())
	if !r.DryRun || !r.OK || r.Issue != 42 {
		t.Fatalf("result = %+v", r)
	}
	if strings.Join(r.Args, " ") != "issue edit 42 --title New title" {
		t.Fatalf("args = %v", r.Args)
	}
}

func TestIssueEditLiveCallsRunnerWithBody(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{out: "https://x/42"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "42", "--body", "fixed body", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(gh.calls) != 1 {
		t.Fatalf("expected exactly one gh call, got %v", gh.calls)
	}
	if strings.Join(gh.calls[0], " ") != "issue edit 42 --body fixed body" {
		t.Fatalf("gh argv = %v", gh.calls[0])
	}
	r := decodeEdit(t, out.Bytes())
	if r.DryRun || !r.OK || r.URL != "https://x/42" {
		t.Fatalf("result = %+v", r)
	}
}

func TestIssueEditAddRemoveLabelArgv(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "7", "--add-label", "a,b", "--remove-label", "c", "--dry-run", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	r := decodeEdit(t, out.Bytes())
	if strings.Join(r.Args, " ") != "issue edit 7 --add-label a --add-label b --remove-label c" {
		t.Fatalf("args = %v", r.Args)
	}
}

func TestIssueEditRejectsMissingIssue(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--title", "x"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func TestIssueEditRejectsNoChange(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2 for nothing-to-change", code)
	}
}

func TestIssueEditRejectsBodyAndBodyFileTogether(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5", "--body", "a", "--body-file", "f"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2 for body XOR body-file", code)
	}
}

func TestIssueEditReportsGHFailure(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{fail: true}
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5", "--body", "x"}, gh.run); code != 1 {
		t.Fatalf("exit = %d, want 1 on gh failure", code)
	}
}

// labelAwareGH answers `gh label list` with a canned label set and every other gh
// argv with a canned URL, so a test exercises the #4047 closed-vocabulary label
// clamp end to end through the single injected runner — no real gh.
type labelAwareGH struct {
	labels    []string
	listFails bool
	editOut   string
	calls     [][]string
}

func (g *labelAwareGH) run(args []string) (string, string, bool) {
	g.calls = append(g.calls, append([]string(nil), args...))
	if len(args) >= 2 && args[0] == "label" && args[1] == "list" {
		if g.listFails {
			return "", "gh label list boom", false
		}
		rows := make([]struct {
			Name string `json:"name"`
		}, 0, len(g.labels))
		for _, l := range g.labels {
			rows = append(rows, struct {
				Name string `json:"name"`
			}{l})
		}
		b, _ := json.Marshal(rows)
		return string(b), "", true
	}
	return g.editOut, "", true
}

func (g *labelAwareGH) editArgv(t *testing.T) []string {
	t.Helper()
	for _, c := range g.calls {
		if len(c) >= 2 && c[0] == "issue" && c[1] == "edit" {
			return c
		}
	}
	t.Fatalf("no `issue edit` gh call recorded: %v", g.calls)
	return nil
}

// A hallucinated --add-label (one that names no real repo label) is clamped out
// before the `gh issue edit` argv is built, and the drop is surfaced loudly in the
// result + on stderr — the closed-vocabulary clamp at the actuator (#4047).
func TestIssueEditClampsHallucinatedAddLabel(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &labelAwareGH{labels: []string{"bug", "enhancement", "rsi"}, editOut: "https://x/9"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "9", "--add-label", "bug,made-up-label", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	joined := strings.Join(gh.editArgv(t), " ")
	if !strings.Contains(joined, "--add-label bug") {
		t.Fatalf("edit argv must keep the real label: %s", joined)
	}
	if strings.Contains(joined, "made-up-label") {
		t.Fatalf("edit argv must NOT carry the hallucinated label: %s", joined)
	}
	r := decodeEdit(t, out.Bytes())
	if len(r.DroppedLabels) != 1 || r.DroppedLabels[0] != "made-up-label" {
		t.Fatalf("dropped labels = %v, want [made-up-label]", r.DroppedLabels)
	}
	if len(r.AddLabels) != 1 || r.AddLabels[0] != "bug" {
		t.Fatalf("add labels = %v, want [bug]", r.AddLabels)
	}
	if !strings.Contains(errb.String(), "made-up-label") {
		t.Fatalf("stderr must loudly name the dropped label: %s", errb.String())
	}
}

// A proposed label that matches a real one only by case is KEPT (not dropped) and
// rewritten to the canonical spelling — attenuate-and-correct, stricter than the
// source shim's plain case-sensitive drop.
func TestIssueEditLabelClampCanonicalizesCase(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &labelAwareGH{labels: []string{"bug"}, editOut: "https://x/9"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "9", "--add-label", "BUG", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	r := decodeEdit(t, out.Bytes())
	if len(r.DroppedLabels) != 0 {
		t.Fatalf("BUG matches canonical bug — nothing should drop: %v", r.DroppedLabels)
	}
	if len(r.AddLabels) != 1 || r.AddLabels[0] != "bug" {
		t.Fatalf("add labels = %v, want canonical [bug]", r.AddLabels)
	}
	if !strings.Contains(strings.Join(gh.editArgv(t), " "), "--add-label bug") {
		t.Fatalf("edit argv must carry the canonical spelling: %v", gh.editArgv(t))
	}
}

// When the repo label set cannot be fetched, the clamp fails OPEN loudly: the
// proposed labels pass through unclamped (a read outage must not wedge a repair
// loop; an unknown label already errors at gh itself) and a warning is emitted.
func TestIssueEditLabelClampFailsOpenOnListError(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &labelAwareGH{listFails: true, editOut: "https://x/9"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "9", "--add-label", "whatever", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	r := decodeEdit(t, out.Bytes())
	if len(r.AddLabels) != 1 || r.AddLabels[0] != "whatever" {
		t.Fatalf("fail-open must keep proposed labels: %v", r.AddLabels)
	}
	if len(r.DroppedLabels) != 0 {
		t.Fatalf("fail-open drops nothing: %v", r.DroppedLabels)
	}
	if !strings.Contains(errb.String(), "skipping label clamp") {
		t.Fatalf("fail-open must warn: %s", errb.String())
	}
}

func TestIssueEditScrubsProtectedTitleAndBody(t *testing.T) {
	cpu, gpu := "da"+"33", "dgx"+"1"
	var got []string
	runner := func(args []string) (string, string, bool) {
		got = append([]string(nil), args...)
		return "https://example.invalid/9\n", "", true
	}
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "9", "--title", cpu + " recovery", "--body", "parity with " + gpu}, runner); code != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "CPU server recovery") || !strings.Contains(joined, "parity with GPU server") {
		t.Fatalf("gh argv not scrubbed: %#v", got)
	}
}
