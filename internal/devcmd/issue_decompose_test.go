package devcmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// oversizedEpic is a routeable issue whose declared step budget exceeds the
// dispatch-leaf cap, so issuecontract flags ReasonOversizedSteps → decompose
// target. budget = ceil(20/8) = 3.
func oversizedEpic(number int) issuepolicy.IssueDraft {
	return issuepolicy.IssueDraft{
		Number: number,
		Title:  "Rework the launcher",
		Body: "## In scope\nEverything about the launcher.\n\n" +
			"## Lane\ncmd\n\n## Expected steps\n20\n\n" +
			"## Done condition / witness\nAll launcher paths green.\n",
	}
}

// labelledEpic is flagged non-leaf purely by its `epic` label (isDispatchLeaf),
// with no step budget → scaffold budget floors at 2.
func labelledEpic(number int) issuepolicy.IssueDraft {
	return issuepolicy.IssueDraft{
		Number: number,
		Title:  "Auth epic",
		Body:   "## In scope\nAuth.\n",
		Labels: []issuepolicy.IssueLabel{{Name: "epic"}},
	}
}

// leafIssue is a small, complete, unlabelled unit: a dispatch leaf, never a
// decompose target.
func leafIssue(number int) issuepolicy.IssueDraft {
	return issuepolicy.IssueDraft{
		Number: number,
		Title:  "Fix typo in header",
		Body: "## In scope\nFix one typo.\n\n## Lane\ndocs\n\n" +
			"## Likely files\nREADME.md\n\n## Expected steps\n1\n\n" +
			"## Done condition / witness\nTypo gone.\n",
	}
}

// decomposeGHRunner is a fake gh executor that hands back a synthetic issue URL
// for each `issue create` (so number parsing + linking can be asserted) and
// records every argv it saw. Guarded so parallel use is safe.
type decomposeGHRunner struct {
	mu       sync.Mutex
	calls    [][]string
	nextNum  int
	failOnce bool // fail the first create call
	failed   bool
}

func (r *decomposeGHRunner) run(args []string) (string, string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, append([]string(nil), args...))
	if len(args) >= 2 && args[0] == "issue" && args[1] == "create" {
		if r.failOnce && !r.failed {
			r.failed = true
			return "", "gh: rate limited", false
		}
		r.nextNum++
		return fmt.Sprintf("https://github.com/o/r/issues/%d", 900+r.nextNum), "", true
	}
	// issue edit (parent link) succeeds silently.
	return "https://github.com/o/r/issues/edit", "", true
}

func (r *decomposeGHRunner) createCalls() [][]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out [][]string
	for _, c := range r.calls {
		if len(c) >= 2 && c[0] == "issue" && c[1] == "create" {
			out = append(out, c)
		}
	}
	return out
}

func (r *decomposeGHRunner) editCalls() [][]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out [][]string
	for _, c := range r.calls {
		if len(c) >= 2 && c[0] == "issue" && c[1] == "edit" {
			out = append(out, c)
		}
	}
	return out
}

func decodeDecomposeResult(t *testing.T, out []byte) decomposeResult {
	t.Helper()
	var res decomposeResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("decode result json: %v\n%s", err, out)
	}
	return res
}

func TestDecomposeDryRunScaffoldsEpicsAndIgnoresLeaves(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rr := &decomposeGHRunner{}
	issues := []issuepolicy.IssueDraft{oversizedEpic(10), leafIssue(11), labelledEpic(12)}

	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--json"}, issues, rr.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if len(rr.calls) != 0 {
		t.Fatalf("dry-run must not call gh, saw %d calls", len(rr.calls))
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	if !res.DryRun || res.Live {
		t.Fatalf("expected dry-run result, got live=%v dry=%v", res.Live, res.DryRun)
	}
	if res.Counts.Epics != 2 {
		t.Fatalf("epics = %d, want 2 (leaf ignored)", res.Counts.Epics)
	}
	byParent := map[int]decomposeRow{}
	for _, r := range res.Rows {
		byParent[r.Parent] = r
	}
	if _, ok := byParent[11]; ok {
		t.Fatalf("leaf issue #11 must not be a decompose row")
	}
	if got := byParent[10]; got.ChildBudget != 3 || len(got.Children) != 3 || got.Disposition != dispositionScaffold {
		t.Fatalf("oversized epic #10: budget=%d children=%d disp=%s, want 3/3/scaffold", got.ChildBudget, len(got.Children), got.Disposition)
	}
	if got := byParent[12]; got.ChildBudget != 2 || len(got.Children) != 2 {
		t.Fatalf("labelled epic #12: budget=%d children=%d, want 2/2", got.ChildBudget, len(got.Children))
	}
	// Scaffold children carry a parent-context pointer and the field skeleton.
	for _, ch := range byParent[10].Children {
		var body string
		for i, a := range ch.Args {
			if a == "--body" && i+1 < len(ch.Args) {
				body = ch.Args[i+1]
			}
		}
		if !strings.Contains(body, "Decomposed from #10") || !strings.Contains(body, "## Done condition / witness") {
			t.Fatalf("scaffold child body missing parent pointer or skeleton: %q", body)
		}
	}
}

func TestDecomposeLiveFromPlanFilesChildrenAndLinksParent(t *testing.T) {
	dir := t.TempDir()
	plan := []decomposePlanSpec{{
		Parent: 10,
		Children: []decomposeChildSpec{
			{Title: "Launcher: parse", Body: "## In scope\nparse\n", Labels: []string{"cmd"}},
			{Title: "Launcher: render", Body: "## In scope\nrender\n"},
		},
	}}
	planPath := filepath.Join(dir, "plan.json")
	writeDecomposePlanFile(t, planPath, plan)

	var stdout, stderr bytes.Buffer
	rr := &decomposeGHRunner{}
	issues := []issuepolicy.IssueDraft{oversizedEpic(10)}

	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--live", "--parent-baseline-points", "13", "--target-envelope", "- generated children passing strict review: = 100 percent", "--witnessed-envelope", "- generated children passing strict review: = 100 percent", "--from-plan", planPath, "--json"}, issues, rr.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	if res.Counts.ChildrenCreated != 2 || res.Counts.ParentsLinked != 1 || res.Counts.Failed != 0 {
		t.Fatalf("counts = %+v, want created=2 linked=1 failed=0", res.Counts)
	}
	creates := rr.createCalls()
	if len(creates) != 2 {
		t.Fatalf("create calls = %d, want 2", len(creates))
	}
	// First child carries its label through the argv.
	if !argvHas(creates[0], "--label", "cmd") {
		t.Fatalf("first child create missing --label cmd: %v", creates[0])
	}
	// Parent link edits #10 with a Blocked-by line naming both created children.
	edits := rr.editCalls()
	if len(edits) != 1 {
		t.Fatalf("edit calls = %d, want 1", len(edits))
	}
	if edits[0][2] != "10" {
		t.Fatalf("parent link edits #%s, want #10", edits[0][2])
	}
	body := argvValue(edits[0], "--body")
	if !strings.Contains(body, "Blocked by #901, #902") {
		t.Fatalf("parent body missing blocked-by line for created children: %q", body)
	}
	// The original epic body is preserved, not clobbered.
	if !strings.Contains(body, "Everything about the launcher") {
		t.Fatalf("parent link clobbered original body: %q", body)
	}
}

func TestDecomposeMaxCreateFuseRefusesBeforeGH(t *testing.T) {
	dir := t.TempDir()
	plan := []decomposePlanSpec{{
		Parent: 10,
		Children: []decomposeChildSpec{
			{Title: "a", Body: "## In scope\na\n"},
			{Title: "b", Body: "## In scope\nb\n"},
			{Title: "c", Body: "## In scope\nc\n"},
		},
	}}
	planPath := filepath.Join(dir, "plan.json")
	writeDecomposePlanFile(t, planPath, plan)

	var stdout, stderr bytes.Buffer
	rr := &decomposeGHRunner{}
	issues := []issuepolicy.IssueDraft{oversizedEpic(10)}

	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--live", "--parent-baseline-points", "13", "--target-envelope", "- generated children passing strict review: = 100 percent", "--witnessed-envelope", "- generated children passing strict review: = 100 percent", "--from-plan", planPath, "--max-create", "2"}, issues, rr.run)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (fuse tripped)", code)
	}
	if len(rr.calls) != 0 {
		t.Fatalf("fuse must trip before any gh call, saw %d", len(rr.calls))
	}
	if !strings.Contains(stderr.String(), "over --max-create=2") {
		t.Fatalf("stderr missing fuse message: %s", stderr.String())
	}
}

func TestDecomposeLiveScaffoldNeedsAllowStubs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rr := &decomposeGHRunner{}
	issues := []issuepolicy.IssueDraft{oversizedEpic(10)}

	// --live without --from-plan and without --allow-stubs: scaffolds are not
	// filed; run succeeds but touches no gh and warns.
	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--live", "--parent-baseline-points", "13", "--target-envelope", "- generated children passing strict review: = 100 percent", "--witnessed-envelope", "- generated children passing strict review: = 100 percent", "--json"}, issues, rr.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if len(rr.calls) != 0 {
		t.Fatalf("scaffold without --allow-stubs must not call gh, saw %d", len(rr.calls))
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	if res.Counts.ChildrenCreated != 0 {
		t.Fatalf("children created = %d, want 0", res.Counts.ChildrenCreated)
	}
	if !strings.Contains(stderr.String(), "--allow-stubs") {
		t.Fatalf("stderr should hint --allow-stubs: %s", stderr.String())
	}

	// With --allow-stubs the stubs are filed.
	stdout.Reset()
	stderr.Reset()
	rr2 := &decomposeGHRunner{}
	code = runIssueDecomposeWith(&stdout, &stderr, []string{"--live", "--parent-baseline-points", "13", "--target-envelope", "- generated children passing strict review: = 100 percent", "--witnessed-envelope", "- generated children passing strict review: = 100 percent", "--allow-stubs", "--json"}, issues, rr2.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if len(rr2.createCalls()) != 3 {
		t.Fatalf("allow-stubs create calls = %d, want 3", len(rr2.createCalls()))
	}
}

func TestDecomposeChildCreateFailureSkipsParentLink(t *testing.T) {
	dir := t.TempDir()
	plan := []decomposePlanSpec{{
		Parent: 10,
		Children: []decomposeChildSpec{
			{Title: "a", Body: "## In scope\na\n"},
			{Title: "b", Body: "## In scope\nb\n"},
		},
	}}
	planPath := filepath.Join(dir, "plan.json")
	writeDecomposePlanFile(t, planPath, plan)

	var stdout, stderr bytes.Buffer
	rr := &decomposeGHRunner{failOnce: true}
	issues := []issuepolicy.IssueDraft{oversizedEpic(10)}

	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--live", "--parent-baseline-points", "13", "--target-envelope", "- generated children passing strict review: = 100 percent", "--witnessed-envelope", "- generated children passing strict review: = 100 percent", "--from-plan", planPath, "--json"}, issues, rr.run)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (a child create failed)", code)
	}
	if len(rr.editCalls()) != 0 {
		t.Fatalf("parent must not be linked when a child create failed, saw %d edits", len(rr.editCalls()))
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	if res.Counts.Failed != 1 || res.Counts.ParentsLinked != 0 {
		t.Fatalf("counts = %+v, want failed=1 linked=0", res.Counts)
	}
}

func TestDecomposePlanOverrideOfNonEpicAndOrphanParent(t *testing.T) {
	dir := t.TempDir()
	plan := []decomposePlanSpec{
		{Parent: 11, Children: []decomposeChildSpec{{Title: "x", Body: "## In scope\nx\n"}}}, // #11 is a leaf → operator override
		{Parent: 99, Children: []decomposeChildSpec{{Title: "y", Body: "## In scope\ny\n"}}}, // #99 absent → orphan/error
	}
	planPath := filepath.Join(dir, "plan.json")
	writeDecomposePlanFile(t, planPath, plan)

	var stdout, stderr bytes.Buffer
	issues := []issuepolicy.IssueDraft{leafIssue(11)}

	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--from-plan", planPath, "--json"}, issues, nil)
	// Dry-run, but the orphan row is a failure → exit 1.
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (orphan parent)", code)
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	byParent := map[int]decomposeRow{}
	for _, r := range res.Rows {
		byParent[r.Parent] = r
	}
	over := byParent[11]
	if over.Disposition != dispositionDecompose || len(over.Reasons) == 0 || over.Reasons[0] != "OPERATOR_REQUESTED" {
		t.Fatalf("non-epic override #11: disp=%s reasons=%v, want decompose/OPERATOR_REQUESTED", over.Disposition, over.Reasons)
	}
	orphan := byParent[99]
	if orphan.Disposition != dispositionError || orphan.Error == "" {
		t.Fatalf("orphan #99: disp=%s err=%q, want error row", orphan.Disposition, orphan.Error)
	}
}

func TestDecomposeFromIssuesFile(t *testing.T) {
	dir := t.TempDir()
	// gh issue list --json shape: an array of {number,title,body,labels}.
	ghJSON := `[{"number":10,"title":"Rework the launcher","body":"## In scope\nAll.\n\n## Expected steps\n20\n","labels":[]}]`
	issuesPath := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issuesPath, []byte(ghJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runIssueDecomposeWith(&stdout, &stderr, []string{"--from-issues", issuesPath, "--json"}, nil, nil)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	res := decodeDecomposeResult(t, stdout.Bytes())
	if res.Counts.Epics != 1 {
		t.Fatalf("epics = %d, want 1", res.Counts.Epics)
	}
}

func TestParseCreatedIssueNumber(t *testing.T) {
	cases := []struct {
		in   string
		want int
		ok   bool
	}{
		{"https://github.com/o/r/issues/1234", 1234, true},
		{"  https://github.com/o/r/issues/7\n", 7, true},
		{"not-a-url", 0, false},
		{"https://github.com/o/r/issues/", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok := parseCreatedIssueNumber(c.in)
		if got != c.want || ok != c.ok {
			t.Fatalf("parseCreatedIssueNumber(%q) = %d,%v want %d,%v", c.in, got, ok, c.want, c.ok)
		}
	}
}

// --- helpers ---------------------------------------------------------------

func writeDecomposePlanFile(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func argvValue(argv []string, flag string) string {
	for i, a := range argv {
		if a == flag && i+1 < len(argv) {
			return argv[i+1]
		}
	}
	return ""
}

func argvHas(argv []string, flag, val string) bool {
	for i, a := range argv {
		if a == flag && i+1 < len(argv) && argv[i+1] == val {
			return true
		}
	}
	return false
}
