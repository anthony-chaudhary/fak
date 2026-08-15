package dosdecision

import (
	"reflect"
	"strings"
	"testing"
)

// refuseRow is the row `dos decisions --json` emits for an OP_REFUSE: the kernel
// lifts the journal entry's lane and prose verbatim (dos/decisions.py
// `_from_lane_journal`), and the prose is the arbiter/admission refusal string.
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

// TestReleasedCollisionLeavesTheActiveQueue is the #6494 witness: a lane
// collision refusal is real work while the blocking lease is live, and stops being
// work the moment that lease is released — even though the refusal row itself is
// unchanged (its journal entry never learned the blocker went away).
func TestReleasedCollisionLeavesTheActiveQueue(t *testing.T) {
	// 1. `dos lease-lane acquire --lane devcmd` succeeded for a sibling, then this
	//    loop asked for `cmd` and the arbiter refused on an exact-glob collision.
	row := refuseRow("cmd", "lane 'cmd' cannot share live lane 'devcmd': exact-glob overlap: "+
		"identical glob claimed by both lanes (2: cmd/fak/**) — same write region, hard collision regardless of ratio.")

	// 2. While `devcmd` is held, the refusal is genuine unresolved human work.
	held := Revalidate([]Row{row}, LiveSet{Lanes: []string{"devcmd"}, Known: true})
	if len(held.Active) != 1 || len(held.Superseded) != 0 || held.Cleared != 0 {
		t.Fatalf("blocker live: active=%d superseded=%d cleared=%d", len(held.Active), len(held.Superseded), held.Cleared)
	}

	// 3. `dos lease-lane release --lane devcmd` — the live set is now empty.
	after := Revalidate([]Row{row}, LiveSet{Lanes: []string{}, Known: true})
	if len(after.Active) != 0 {
		t.Fatalf("released blocker still active: %+v", after.Active)
	}
	if after.Cleared != 1 || len(after.Superseded) != 1 {
		t.Fatalf("cleared=%d superseded=%d, want 1/1", after.Cleared, len(after.Superseded))
	}

	// 4. History is preserved, annotated, and still carries the original fields.
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

	// 5. The caller's row is untouched — annotation happens on a copy.
	if _, mutated := row["resolved"]; mutated {
		t.Fatalf("input row mutated: %+v", row)
	}
}

// TestUnreadableLiveSetClearsNothing pins the fail-closed rule: a kernel we could
// not read is not an empty kernel.
func TestUnreadableLiveSetClearsNothing(t *testing.T) {
	rows := []Row{refuseRow("hooks", "lane 'hooks' is already held by a live loop — pick a different --lane or wait.")}
	got := Revalidate(rows, LiveSet{Known: false})
	if len(got.Active) != 1 || got.Cleared != 0 || len(got.Superseded) != 0 {
		t.Fatalf("unknown live set cleared work: %+v", got)
	}
}

func TestOwnLaneStillHeldKeepsTheRowActive(t *testing.T) {
	// An "already held" refusal names no other lane: its blocker is the prior
	// holder of its OWN lane, so a live `hooks` lease must keep it active.
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
	// No lane field and prose that names no lane: we cannot prove the contention
	// is over, so the row is never dropped.
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
