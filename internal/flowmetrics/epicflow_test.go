package flowmetrics

import (
	"testing"
	"time"
)

func TestParseTaskListSkipsFencedExamplesAndFencesForeignRefs(t *testing.T) {
	body := "Plan:\n" +
		"- [x] #101 land the core\n" +
		"- [ ] #102 wire the CLI\n" +
		"- [ ] no child ref on this line\n" +
		"\n" +
		"Example of the shape we want:\n" +
		"```\n" +
		"- [ ] #999 this is a quoted example, not our scope\n" +
		"```\n" +
		"- [ ] see llama.cpp #14762 for prior art\n"
	checked, unchecked, children := ParseTaskList(body)
	if checked != 1 {
		t.Fatalf("checked = %d, want 1", checked)
	}
	// The fenced line must not inflate the count; the two real unchecked
	// lines plus the foreign-ref line are 3.
	if unchecked != 3 {
		t.Fatalf("unchecked = %d, want 3 (fenced example excluded)", unchecked)
	}
	if len(children) != 2 || children[0] != 101 || children[1] != 102 {
		t.Fatalf("children = %v, want [101 102] — #999 is fenced prose, #14762 is foreign", children)
	}
}

func TestBuildEpicProgressPrefersWitnessedChildren(t *testing.T) {
	open := base
	// Four children; two closed, at +10h and +30h.
	byNumber := map[int]Issue{
		1: {Number: 1, CreatedAt: open, ClosedAt: ptr(open.Add(10 * time.Hour))},
		2: {Number: 2, CreatedAt: open, ClosedAt: ptr(open.Add(30 * time.Hour))},
		3: {Number: 3, CreatedAt: open},
		4: {Number: 4, CreatedAt: open},
	}
	epic := Issue{
		Number:    50,
		Title:     "Epic: the thing",
		CreatedAt: open,
		Body:      "- [ ] #1\n- [ ] #2\n- [ ] #3\n- [ ] #4\n",
	}
	ep := BuildEpicProgress(epic, byNumber, nil)
	if ep.Basis != "children" {
		t.Fatalf("basis = %q, want children", ep.Basis)
	}
	// Every box is unticked, yet half the work is provably done — which is
	// exactly the gap this basis choice exists to close.
	if ep.Checked != 0 {
		t.Fatalf("checked = %d, want 0", ep.Checked)
	}
	if ep.ChildrenClosed != 2 || ep.Fraction != 0.5 {
		t.Fatalf("closed/fraction = %d/%v, want 2/0.5", ep.ChildrenClosed, ep.Fraction)
	}
	// 25% of 4 children is 1 child -> the first close at +10h.
	if got, ok := ep.HoursTo[25]; !ok || got != 10 {
		t.Fatalf("hours_to[25] = %v (present=%v), want 10", got, ok)
	}
	if got, ok := ep.HoursTo[50]; !ok || got != 30 {
		t.Fatalf("hours_to[50] = %v (present=%v), want 30", got, ok)
	}
	// Unreached thresholds must be ABSENT, never zero — a zero would read as
	// "reached instantly".
	if _, ok := ep.HoursTo[75]; ok {
		t.Fatalf("hours_to[75] must be absent until reached")
	}
	if _, ok := ep.Milestones[100]; ok {
		t.Fatalf("milestones[100] must be absent until reached")
	}
}

func TestBuildEpicProgressUsesCeilingDivisionForThresholds(t *testing.T) {
	open := base
	// Three children, all closed: 50% must need 2 children, not 1.
	byNumber := map[int]Issue{}
	for i := 1; i <= 3; i++ {
		byNumber[i] = Issue{Number: i, CreatedAt: open, ClosedAt: ptr(open.Add(time.Duration(i*10) * time.Hour))}
	}
	ep := BuildEpicProgress(
		Issue{Number: 60, CreatedAt: open, Body: "- [ ] #1\n- [ ] #2\n- [ ] #3\n"},
		byNumber, nil)
	if ep.HoursTo[50] != 20 {
		t.Fatalf("hours_to[50] = %v, want 20 (2nd of 3 children, not the 1st)", ep.HoursTo[50])
	}
	if ep.HoursTo[100] != 30 {
		t.Fatalf("hours_to[100] = %v, want 30", ep.HoursTo[100])
	}
	if ep.Fraction != 1 {
		t.Fatalf("fraction = %v, want 1", ep.Fraction)
	}
}

func TestBuildEpicProgressCountsUnresolvableChildrenAsOpen(t *testing.T) {
	// A child absent from the lookup is still declared work. Counting it as
	// done would let a dangling reference inflate progress.
	ep := BuildEpicProgress(
		Issue{Number: 70, CreatedAt: base, Body: "- [x] #1\n- [x] #2\n"},
		map[int]Issue{1: {Number: 1, CreatedAt: base, ClosedAt: ptr(base.Add(time.Hour))}},
		nil)
	if ep.ChildrenClosed != 1 || ep.Fraction != 0.5 {
		t.Fatalf("closed/fraction = %d/%v, want 1/0.5 despite both boxes ticked",
			ep.ChildrenClosed, ep.Fraction)
	}
	if ep.Basis != "children" {
		t.Fatalf("basis = %q, want children", ep.Basis)
	}
}

func TestBuildEpicProgressFallsBackToCheckboxWithNoMilestones(t *testing.T) {
	ep := BuildEpicProgress(
		Issue{Number: 80, CreatedAt: base, Body: "- [x] wrote it\n- [ ] tested it\n- [ ] shipped it\n"},
		map[int]Issue{}, nil)
	if ep.Basis != "checkbox" {
		t.Fatalf("basis = %q, want checkbox", ep.Basis)
	}
	if ep.Fraction < 0.33 || ep.Fraction > 0.34 {
		t.Fatalf("fraction = %v, want ~1/3", ep.Fraction)
	}
	// A tick carries no date, so inventing a milestone from it would be a
	// fabricated timestamp.
	if len(ep.Milestones) != 0 || len(ep.HoursTo) != 0 {
		t.Fatalf("checkbox basis must emit no milestones, got %v", ep.HoursTo)
	}
}

func TestBuildEpicProgressReportsNothingToMeasure(t *testing.T) {
	ep := BuildEpicProgress(Issue{Number: 90, CreatedAt: base, Body: "just prose"}, map[int]Issue{}, nil)
	if ep.Basis != "none" || ep.Fraction != -1 {
		t.Fatalf("basis/fraction = %q/%v, want none/-1 — unknown is not 0%%", ep.Basis, ep.Fraction)
	}
}

func TestIsAggregateAcceptsLabelTitleAndTaskList(t *testing.T) {
	cases := []struct {
		name string
		iss  Issue
		want bool
	}{
		{"label", Issue{Labels: []string{"bug", "epic"}}, true},
		{"title epic", Issue{Title: "Epic: consolidate the gateway"}, true},
		{"title bracket", Issue{Title: "[epic] something"}, true},
		{"title track", Issue{Title: "track(gateway): rollup"}, true},
		{"long task list", Issue{Title: "fix a thing", Body: "- [ ] a\n- [ ] b\n- [ ] c\n"}, true},
		{"leaf", Issue{Title: "fix(gateway): one thing", Body: "- [ ] a\n"}, false},
		{"epicenter is not an epic prefix match we care about", Issue{Title: "reduce blast radius"}, false},
	}
	for _, c := range cases {
		if got := IsAggregate(c.iss, 3); got != c.want {
			t.Fatalf("%s: IsAggregate = %v, want %v", c.name, got, c.want)
		}
	}
	// minItems 0 disables the task-list arm entirely.
	if IsAggregate(Issue{Body: "- [ ] a\n- [ ] b\n"}, 0) {
		t.Fatalf("minItems 0 must disable the task-list arm")
	}
}
