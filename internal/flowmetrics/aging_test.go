package flowmetrics

import (
	"bytes"
	"strings"
	"testing"
)

// agingFixture is the witness corpus #6197 names: THREE aging spans (started,
// never closed, quiet for well over AgingWIPDays) and ONE unstarted issue. The
// unstarted row is the load-bearing one — an open issue with no commit is
// backlog, and admitting it here is what turns an 86-item readout into a
// 1300-item one nobody reads.
func agingFixture() Input {
	issues := []Issue{
		{Number: 10, Title: "feat(gateway): oldest rot", CreatedAt: at(-60 * 24)},
		{Number: 11, Title: "fix(journal): middle rot", CreatedAt: at(-30 * 24)},
		{Number: 12, Title: "docs(x): youngest rot", CreatedAt: at(-20 * 24)},
		{Number: 13, Title: "feat(x): nobody started this", CreatedAt: at(-50 * 24)},
	}
	commits := []Commit{
		{SHA: "a1", When: at(-40 * 24), Leaf: "gateway", Issues: []int{10}},
		{SHA: "a2", When: at(-39 * 24), Leaf: "journal", Issues: []int{10}},
		{SHA: "a3", When: at(-38 * 24), Leaf: "gateway", Issues: []int{10}},
		{SHA: "b1", When: at(-20 * 24), Leaf: "journal", Issues: []int{11}},
		// #12's commits carry no `(fak <leaf>)` trailer, so its lane column
		// must render as an explicit dash rather than as blank space.
		{SHA: "c1", When: at(-10 * 24), Issues: []int{12}},
		{SHA: "c2", When: at(-9 * 24), Issues: []int{12}},
	}
	return Input{Issues: issues, Commits: commits, Now: base, WindowDays: 30, Tree: TreeWIP{Measured: true}}
}

// agingRows returns the readout's issue rows in printed order, dropping the
// header and any truncation footer, so a test can assert on the LIST rather than
// on substring presence anywhere in the output.
func agingRows(out string) []string {
	var rows []string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "  #") {
			rows = append(rows, line)
		}
	}
	return rows
}

// TestAgingWIPReadoutListsOnlyStartedAgingWorkOldestFirst is the issue's
// witness: against a corpus of three aging spans and one unstarted issue, the
// readout prints exactly the three aging spans in descending age order.
func TestAgingWIPReadoutListsOnlyStartedAgingWorkOldestFirst(t *testing.T) {
	rep := Build(agingFixture())
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	out := buf.String()

	rows := agingRows(out)
	if len(rows) != 3 {
		t.Fatalf("readout printed %d rows, want exactly 3:\n%s", len(rows), out)
	}
	for i, want := range []string{"#10", "#11", "#12"} {
		if !strings.HasPrefix(strings.TrimSpace(rows[i]), want) {
			t.Fatalf("row %d = %q, want it to lead with %s (oldest first)", i, rows[i], want)
		}
	}
	if strings.Contains(out, "#13") {
		t.Fatalf("unstarted issue #13 appeared in the readout — that is backlog, not WIP:\n%s", out)
	}
	// Age, commit count and lane are the three facts that let an operator pick
	// one of these up; a row missing any of them is a count with extra words.
	if !strings.Contains(rows[0], "40.0d") || !strings.Contains(rows[0], "3 commit(s)") ||
		!strings.Contains(rows[0], "gateway,journal") {
		t.Fatalf("oldest row lost age/commits/lanes: %q", rows[0])
	}
	if !strings.Contains(rows[2], " - ") {
		t.Fatalf("leafless row should print a dash for its lanes: %q", rows[2])
	}
	if !strings.Contains(out, "over 7d") {
		t.Fatalf("header must name the threshold that defines the set:\n%s", out)
	}
}

// TestAgingWIPReadoutCountMatchesTheKPI is the done condition: the printed set
// and the graded number are the same fold. If these ever disagree, the readout
// is pointing an operator at a different set than the defect line names.
func TestAgingWIPReadoutCountMatchesTheKPI(t *testing.T) {
	rep := Build(agingFixture())
	k := findKPI(t, rep, "aging_wip")
	if int(k.Value) != rep.AgingTotal {
		t.Fatalf("aging_wip kpi value %v != AgingTotal %d", k.Value, rep.AgingTotal)
	}
	if rep.AgingTotal != len(rep.Aging) {
		t.Fatalf("untruncated list length %d != total %d", len(rep.Aging), rep.AgingTotal)
	}
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	if got := len(agingRows(buf.String())); got != int(k.Value) {
		t.Fatalf("readout printed %d rows but the aging_wip kpi counts %v", got, k.Value)
	}
}

// TestAgingWIPReadoutExcludesYoungAndClosedWork pins the two exclusions the
// count depends on. Work started yesterday is in flight but not rotting, and
// counting it would inflate the target set past the point of being actionable.
func TestAgingWIPReadoutExcludesYoungAndClosedWork(t *testing.T) {
	in := agingFixture()
	in.Issues = append(in.Issues,
		Issue{Number: 14, Title: "fresh", CreatedAt: at(-3 * 24)},
		Issue{Number: 15, Title: "landed", CreatedAt: at(-40 * 24), ClosedAt: ptr(at(-1 * 24))},
	)
	in.Commits = append(in.Commits,
		Commit{SHA: "d1", When: at(-2 * 24), Leaf: "x", Issues: []int{14}},
		Commit{SHA: "e1", When: at(-30 * 24), Leaf: "x", Issues: []int{15}},
	)
	rep := Build(in)
	if rep.AgingTotal != 3 {
		t.Fatalf("AgingTotal = %d, want 3 (a 2d-old start is not stalled, a closed issue is done)", rep.AgingTotal)
	}
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	out := buf.String()
	for _, unwanted := range []string{"#14", "#15"} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("%s must not appear in the aging readout:\n%s", unwanted, out)
		}
	}
}

// TestAgingWIPReadoutBoundsTheListButNotTheCount keeps truncation honest: the
// list is capped so the payload stays bounded, and the SIZE of the problem is
// still reported in full. Printing 2 rows and calling that the total is how a
// limit turns into a false all-clear.
func TestAgingWIPReadoutBoundsTheListButNotTheCount(t *testing.T) {
	in := agingFixture()
	in.AgingLimit = 2
	rep := Build(in)
	if len(rep.Aging) != 2 || rep.AgingTotal != 3 {
		t.Fatalf("limit 2 gave %d rows / total %d, want 2 / 3", len(rep.Aging), rep.AgingTotal)
	}
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	out := buf.String()
	if got := len(agingRows(out)); got != 2 {
		t.Fatalf("readout printed %d rows under a limit of 2", got)
	}
	if !strings.Contains(out, "3 issue(s)") {
		t.Fatalf("header must report the full set size, not the truncated one:\n%s", out)
	}
	if !strings.Contains(out, "1 more not listed") {
		t.Fatalf("truncated readout must say how many rows it withheld:\n%s", out)
	}
}

// TestAgingWIPReadoutOnACleanCorpusSaysNone pins the empty case. A silent
// readout reads as a broken command; "none" reads as the answer it is.
func TestAgingWIPReadoutOnACleanCorpusSaysNone(t *testing.T) {
	rep := Build(healthyInput(base))
	if rep.AgingTotal != 0 {
		t.Fatalf("healthy corpus has AgingTotal = %d, want 0", rep.AgingTotal)
	}
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	if out := buf.String(); !strings.Contains(out, "none") {
		t.Fatalf("clean readout = %q, want an explicit none", out)
	}
}

// TestAgingWIPOverTheCapIsNamedInTheHeader keeps the readout bound to the fixed
// ceiling: the operator sees the same over-cap fact the defect line asserts.
func TestAgingWIPOverTheCapIsNamedInTheHeader(t *testing.T) {
	in := healthyInput(base)
	for i := 200; i < 215; i++ {
		in.Issues = append(in.Issues, Issue{Number: i, CreatedAt: at(-31 * 24)})
		in.Commits = append(in.Commits, Commit{
			SHA: "s" + string(rune('a'+i-200)), When: at(-30 * 24), Leaf: "x", Issues: []int{i},
		})
	}
	rep := Build(in)
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	out := buf.String()
	if !strings.Contains(out, "3 over the cap of 12") {
		t.Fatalf("header must quantify the overage (15 stalled vs a cap of 12):\n%s", out)
	}
}

// TestAgingWIPTitlesAreClippedNotWrapped keeps one pathological title from
// destroying the alignment of every other row.
func TestAgingWIPTitlesAreClippedNotWrapped(t *testing.T) {
	in := agingFixture()
	in.Issues[0].Title = strings.Repeat("x", 200)
	rep := Build(in)
	var buf bytes.Buffer
	RenderAging(&buf, rep, base)
	row := agingRows(buf.String())[0]
	if len(row) > 120 || !strings.HasSuffix(row, "...") {
		t.Fatalf("long title was not clipped (len %d): %q", len(row), row)
	}
}
