package dosdecision

import (
	"reflect"
	"strings"
	"testing"
)

func refuseRow(lane, reason string) Row {
	return Row{
		"kind":          KindArbiterRefuse,
		"resolver_kind": "HUMAN",
		"lane":          lane,
		"reason_token":  "",
		"reason_text":   reason,
		"age_seconds":   164000,
		"source_path":   ".dos/lane-journal.jsonl",
		"evidence":      []any{"journal seq #1733"},
	}
}

func TestReleasedCollisionLeavesTheActiveQueue(t *testing.T) {
	row := refuseRow("cmd", "lane 'cmd' cannot share live lane 'devcmd': exact-glob overlap: "+
		"identical glob claimed by both lanes (2: cmd/fak/**) — same write region, hard collision regardless of ratio.")

	held := Revalidate([]Row{row}, LiveSet{Lanes: []string{"devcmd"}, Known: true})
	if len(held.Active) != 1 || len(held.Superseded) != 0 || held.Cleared != 0 {
		t.Fatalf("blocker live: active=%d superseded=%d cleared=%d", len(held.Active), len(held.Superseded), held.Cleared)
	}

	after := Revalidate([]Row{row}, LiveSet{Lanes: []string{}, Known: true})
	if len(after.Active) != 0 {
		t.Fatalf("released blocker still active: %+v", after.Active)
	}
	if after.Cleared != 1 || len(after.Superseded) != 1 {
		t.Fatalf("cleared=%d superseded=%d, want 1/1", after.Cleared, len(after.Superseded))
	}

	hist := after.Superseded[0]
	if hist["resolved"] != true || hist["resolution"] != ResolutionLeaseReleased {
		t.Fatalf("history not annotated: %+v", hist)
	}
	ev, _ := hist["resolution_evidence"].(string)
	if !strings.Contains(ev, "cmd") || !strings.Contains(ev, "devcmd") {
		t.Fatalf("evidence names no blocker: %q", ev)
	}
	if hist["age_seconds"] != 164000 || hist["reason_text"] != row["reason_text"] {
		t.Fatalf("history lost original fields: %+v", hist)
	}

	if _, mutated := row["resolved"]; mutated {
		t.Fatalf("input row mutated: %+v", row)
	}
}

func TestUnreadableLiveSetClearsNothing(t *testing.T) {
	rows := []Row{refuseRow("hooks", "lane 'hooks' is already held by a live loop — pick a different --lane or wait.")}
	got := Revalidate(rows, LiveSet{Known: false})
	if len(got.Active) != 1 || got.Cleared != 0 || len(got.Superseded) != 0 {
		t.Fatalf("unknown live set cleared work: %+v", got)
	}
}

func TestOwnLaneStillHeldKeepsTheRowActive(t *testing.T) {
	rows := []Row{refuseRow("hooks", "lane 'hooks' is already held by a live loop — pick a different --lane or wait.")}
	got := Revalidate(rows, LiveSet{Lanes: []string{"HOOKS"}, Known: true})
	if len(got.Active) != 1 || got.Cleared != 0 {
		t.Fatalf("case-folded live lane did not match: %+v", got)
	}
	got = Revalidate(rows, LiveSet{Lanes: []string{"gateway"}, Known: true})
	if got.Cleared != 1 {
		t.Fatalf("released own-lane holder not cleared: %+v", got)
	}
}

func TestNonRefusalKindsPassThrough(t *testing.T) {
	rows := []Row{
		{"kind": "LIVENESS", "lane": "cmd", "reason_text": "SPINNING past budget"},
		{"kind": "HOST_QUEUE_ITEM", "key": "bench-6349-blank", "action": "OPEN_ISSUE"},
		{"kind": "WEDGE", "lane": "cmd", "reason_text": "lane 'cmd' wedged"},
	}
	got := Revalidate(rows, LiveSet{Lanes: nil, Known: true})
	if len(got.Active) != 3 || got.Cleared != 0 {
		t.Fatalf("non-lease kinds revalidated: active=%d cleared=%d", len(got.Active), got.Cleared)
	}
}

func TestUndecidableRefusalStaysActive(t *testing.T) {
	rows := []Row{{"kind": KindArbiterRefuse, "resolver_kind": "HUMAN", "reason_text": "lane refused"}}
	got := Revalidate(rows, LiveSet{Lanes: nil, Known: true})
	if len(got.Active) != 1 || got.Cleared != 0 {
		t.Fatalf("undecidable row dropped: %+v", got)
	}
}

func TestAlreadyResolvedRowIsHistoryNotCleanup(t *testing.T) {
	row := refuseRow("cmd", "lane 'cmd' is already held by a live loop — pick a different --lane or wait.")
	row["resolved"] = true
	got := Revalidate([]Row{row}, LiveSet{Lanes: nil, Known: true})
	if len(got.Superseded) != 1 || len(got.Active) != 0 {
		t.Fatalf("resolved row misplaced: %+v", got)
	}
	if got.Cleared != 0 {
		t.Fatalf("pre-resolved row counted as cleanup: cleared=%d", got.Cleared)
	}
}

func TestBlockingLanesReadsEveryRefusalShape(t *testing.T) {
	for _, tc := range []struct {
		name string
		row  Row
		want []string
	}{
		{
			name: "empty tree vs live lane",
			row: refuseRow("Bash", "lane 'Bash' has an EMPTY tree (unknown blast radius) and cannot share live lane "+
				"'nightrun' — unknown blast radius is never safe to admit concurrently."),
			want: []string{"bash", "nightrun"},
		},
		{
			name: "exclusive lane live",
			row: refuseRow("coordinateoperator", "an exclusive lane is live (lane='global', kind='global', "+
				"loop='20260811-0900'); it touches the whole portfolio — wait for it to finish."),
			want: []string{"coordinateoperator", "global"},
		},
		{
			name: "cluster decoration and path namespace",
			row:  refuseRow("a/b/apply cluster (AFR, ALO)", "lane 'apply' is already held by a live loop — pick a different --lane or wait."),
			want: []string{"apply"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := BlockingLanes(tc.row); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("BlockingLanes=%v want %v", got, tc.want)
			}
		})
	}
}

func TestRevalidatePreservesInputOrder(t *testing.T) {
	rows := []Row{
		refuseRow("cmd", "lane 'cmd' cannot share live lane 'devcmd': exact-glob overlap."),
		{"kind": "LIVENESS", "lane": "issue6329nodecompare"},
		refuseRow("hooks", "lane 'hooks' is already held by a live loop."),
	}
	got := Revalidate(rows, LiveSet{Lanes: []string{"devcmd"}, Known: true})
	if len(got.Active) != 2 || got.Active[0]["lane"] != "cmd" || got.Active[1]["kind"] != "LIVENESS" {
		t.Fatalf("order not preserved: %+v", got.Active)
	}
	if got.Cleared != 1 || got.Superseded[0]["lane"] != "hooks" {
		t.Fatalf("wrong row cleared: %+v", got)
	}
}

func TestLaneKeyNormalization(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"devcmd", "devcmd"},
		{"  devcmd  ", "devcmd"},
		{"DEVCMD", "devcmd"},
		{"a/b/apply cluster (AFR, ALO)", "apply"},
		{"apply (AFR)", "apply"},
		{"internal/dosdecision", "dosdecision"},
		{"internal\\dosdecision", "dosdecision"},
		{"", ""},
		{"   ", ""},
	}
	for _, tc := range cases {
		if got := LaneKey(tc.input); got != tc.want {
			t.Errorf("LaneKey(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestNeedsLiveSet(t *testing.T) {
	if NeedsLiveSet(nil) {
		t.Fatal("nil rows should not need live set")
	}
	if NeedsLiveSet([]Row{{"kind": "LIVENESS"}}) {
		t.Fatal("non-refusal row should not need live set")
	}
	resolved := refuseRow("cmd", "lane 'cmd' held")
	resolved["resolved"] = true
	if NeedsLiveSet([]Row{resolved}) {
		t.Fatal("resolved refusal should not need live set")
	}
	active := refuseRow("cmd", "lane 'cmd' held")
	if !NeedsLiveSet([]Row{resolved, active}) {
		t.Fatal("unresolved refusal must need live set")
	}
}

func TestRevalidatePartialBlockersHeld(t *testing.T) {
	row := refuseRow("cmd", "lane 'cmd' cannot share live lane 'devcmd'")
	// When one blocker is held and another is free, the row remains active.
	res := Revalidate([]Row{row}, LiveSet{Lanes: []string{"devcmd"}, Known: true})
	if len(res.Active) != 1 || res.Cleared != 0 {
		t.Fatalf("expected active when devcmd held: %+v", res)
	}
	res = Revalidate([]Row{row}, LiveSet{Lanes: []string{"cmd"}, Known: true})
	if len(res.Active) != 1 || res.Cleared != 0 {
		t.Fatalf("expected active when cmd held: %+v", res)
	}
	res = Revalidate([]Row{row}, LiveSet{Lanes: []string{}, Known: true})
	if len(res.Active) != 0 || res.Cleared != 1 || len(res.Superseded) != 1 {
		t.Fatalf("expected cleared when all blockers free: %+v", res)
	}
}

func TestRevalidateNilAndEmptyInput(t *testing.T) {
	empty := Revalidate(nil, LiveSet{Known: true})
	if len(empty.Active) != 0 || len(empty.Superseded) != 0 || empty.Cleared != 0 {
		t.Fatalf("unexpected result for nil input: %+v", empty)
	}

	row := refuseRow("cmd", "lane 'cmd' held")
	withNil := Revalidate([]Row{nil, row, nil}, LiveSet{Lanes: []string{}, Known: true})
	if len(withNil.Active) != 0 || len(withNil.Superseded) != 1 || withNil.Cleared != 1 {
		t.Fatalf("unexpected result with nil elements: %+v", withNil)
	}
}
