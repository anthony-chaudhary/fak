package metrics

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestLifecycleEventKindsClosed binds the closed lifecycle vocabulary the issue names:
// promotion, demotion, relabeling, retirement — in that order, each with a human label.
// A drift here is a spec change, not a silent kind addition.
func TestLifecycleEventKindsClosed(t *testing.T) {
	want := []LifecycleEventKind{EventPromotion, EventDemotion, EventRelabeling, EventRetirement}
	if len(LifecycleEventKinds) != len(want) {
		t.Fatalf("LifecycleEventKinds = %v, want %v", LifecycleEventKinds, want)
	}
	for i, k := range want {
		if LifecycleEventKinds[i] != k {
			t.Fatalf("LifecycleEventKinds[%d] = %q, want %q", i, LifecycleEventKinds[i], k)
		}
		if label := LifecycleEventKinds[i].Label(); label == "" || label == string(k) {
			t.Fatalf("kind %q has no human label (got %q)", k, label)
		}
	}
}

// TestLifecycleValidate pins the admission gate: a well-formed, witnessed change of each
// kind is accepted, and every refusal reason is reachable. Direction is decided by horizon
// rank alone (now->future) — the orthogonality guarantee, so priority never enters. This is
// the discipline the ledger exists for: an unwitnessed or ill-directed change is refused,
// not silently recorded.
func TestLifecycleValidate(t *testing.T) {
	cases := []struct {
		name string
		ev   LifecycleEvent
		want LifecycleRefusal
	}{
		// Accepted: witnessed and correctly directed.
		{"promote-nearer", LifecycleEvent{Option: "kv", Kind: EventPromotion, From: "next", To: "now", Evidence: "R1 landed"}, ""},
		{"demote-farther", LifecycleEvent{Option: "kv", Kind: EventDemotion, From: "now", To: "second-next", Evidence: "bench slipped"}, ""},
		{"relabel", LifecycleEvent{Option: "kv", Kind: EventRelabeling, From: "kv-cache", To: "kv-groups"}, ""},
		{"retire", LifecycleEvent{Option: "old", Kind: EventRetirement, From: "future", Evidence: "subsumed"}, ""},
		// Refusals: one per reason.
		{"no-option", LifecycleEvent{Kind: EventPromotion, From: "next", To: "now", Evidence: "x"}, RefusalNoOption},
		{"unknown-kind", LifecycleEvent{Option: "o", Kind: "sideways", From: "a", To: "b"}, RefusalUnknownKind},
		{"promote-unwitnessed", LifecycleEvent{Option: "o", Kind: EventPromotion, From: "next", To: "now"}, RefusalNoEvidence},
		{"retire-unwitnessed", LifecycleEvent{Option: "o", Kind: EventRetirement, From: "now"}, RefusalNoEvidence},
		{"promote-wrong-way", LifecycleEvent{Option: "o", Kind: EventPromotion, From: "now", To: "future", Evidence: "x"}, RefusalBadDirection},
		{"demote-wrong-way", LifecycleEvent{Option: "o", Kind: EventDemotion, From: "future", To: "now", Evidence: "x"}, RefusalBadDirection},
		{"promote-off-vocab", LifecycleEvent{Option: "o", Kind: EventPromotion, From: "someday", To: "now", Evidence: "x"}, RefusalBadDirection},
		{"relabel-noop", LifecycleEvent{Option: "o", Kind: EventRelabeling, From: "same", To: "same"}, RefusalBadRelabel},
		{"relabel-empty", LifecycleEvent{Option: "o", Kind: EventRelabeling, From: "", To: "x"}, RefusalBadRelabel},
		{"retire-with-destination", LifecycleEvent{Option: "o", Kind: EventRetirement, From: "now", To: "next", Evidence: "x"}, RefusalBadRetirement},
	}
	for _, tc := range cases {
		if got := tc.ev.Validate(); got != tc.want {
			t.Fatalf("%s: Validate() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// TestLedgerAppendRefusesUnwitnessed proves Append is the durable gate: a refused event is
// NOT recorded (the ledger never carries an unwitnessed change) and its reason is returned,
// while a valid event is recorded.
func TestLedgerAppendRefusesUnwitnessed(t *testing.T) {
	var l LifecycleLedger
	if r := l.Append(LifecycleEvent{Seq: 1, Option: "o", Kind: EventPromotion, From: "next", To: "now"}); r != RefusalNoEvidence {
		t.Fatalf("unwitnessed promotion should refuse with %q, got %q", RefusalNoEvidence, r)
	}
	if len(l.Events) != 0 {
		t.Fatalf("refused event must not be recorded, ledger has %d events", len(l.Events))
	}
	if r := l.Append(LifecycleEvent{Seq: 2, Option: "o", Kind: EventPromotion, From: "next", To: "now", Evidence: "landed"}); r != "" {
		t.Fatalf("witnessed promotion should be accepted, got refusal %q", r)
	}
	if len(l.Events) != 1 {
		t.Fatalf("accepted event should be recorded, ledger has %d events", len(l.Events))
	}
}

// TestMarshalLineStampsSchema proves each durable row carries the versioned schema tag and
// round-trips through JSON — the shape a future agent replays from disk.
func TestMarshalLineStampsSchema(t *testing.T) {
	ev := LifecycleEvent{Seq: 7, Option: "toon", Kind: EventPromotion, From: "second-next", To: "now", Evidence: "#3067"}
	line, err := ev.MarshalLine()
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	var back LifecycleEvent
	if err := json.Unmarshal([]byte(line), &back); err != nil {
		t.Fatalf("row is not valid JSON: %v\n%s", err, line)
	}
	if back.Schema != LifecycleLedgerSchema {
		t.Fatalf("row schema = %q, want %q", back.Schema, LifecycleLedgerSchema)
	}
	if back.Option != "toon" || back.Kind != EventPromotion || back.To != "now" {
		t.Fatalf("row did not round-trip: %+v", back)
	}
}

// TestFoldReplaysToCurrentState proves the ledger folds down to each option's current
// standing: a promotion/demotion sets the current horizon, a relabeling sets the label,
// and a retirement marks the option retired and clears its horizon — the state a future
// agent recovers without rereading the epic.
func TestFoldReplaysToCurrentState(t *testing.T) {
	var l LifecycleLedger
	appends := []LifecycleEvent{
		{Seq: 1, Option: "kv", Kind: EventDemotion, From: "next", To: "second-next", Evidence: "slipped"},
		{Seq: 2, Option: "kv", Kind: EventPromotion, From: "second-next", To: "next", Evidence: "R1 landed"},
		{Seq: 3, Option: "kv", Kind: EventRelabeling, From: "kv-cache", To: "kv-groups"},
		{Seq: 4, Option: "old", Kind: EventRetirement, From: "future", Evidence: "subsumed"},
	}
	for _, e := range appends {
		if r := l.Append(e); r != "" {
			t.Fatalf("seq %d refused: %q", e.Seq, r)
		}
	}
	states := l.Fold()
	if len(states) != 2 {
		t.Fatalf("expected 2 options, got %+v", states)
	}
	// Sorted by option: "kv" then "old".
	kv, old := states[0], states[1]
	if kv.Option != "kv" || kv.Stream != "next" || kv.Label != "kv-groups" || kv.Retired {
		t.Fatalf("kv folded wrong: %+v", kv)
	}
	if kv.Events != 3 || kv.LastKind != EventRelabeling {
		t.Fatalf("kv event roll-up wrong: %+v", kv)
	}
	if old.Option != "old" || !old.Retired || old.Stream != "" {
		t.Fatalf("old should be retired with no stream: %+v", old)
	}

	counts := l.CountByKind()
	if counts[EventPromotion] != 1 || counts[EventDemotion] != 1 || counts[EventRelabeling] != 1 || counts[EventRetirement] != 1 {
		t.Fatalf("CountByKind = %+v, want one of each", counts)
	}
}

// TestLifecycleRender proves the ledger renders the orthogonality header (naming priority,
// shared trunk, and feature gates), the per-kind counts, each event's transition and
// evidence, and the folded current state — and is deterministic.
func TestLifecycleRender(t *testing.T) {
	var l LifecycleLedger
	for _, e := range []LifecycleEvent{
		{Seq: 1, Option: "toon-wire", Kind: EventPromotion, From: "second-next", To: "now", Evidence: "#3067 shipped", Reason: "wire format stable"},
		{Seq: 2, Option: "old-axis", Kind: EventRetirement, From: "future", Evidence: "subsumed by roadmap"},
	} {
		if r := l.Append(e); r != "" {
			t.Fatalf("seq %d refused: %q", e.Seq, r)
		}
	}
	out := l.Render()

	if !strings.Contains(out, OrthogonalityNote) {
		t.Fatalf("render missing orthogonality note:\n%s", out)
	}
	for _, kw := range []string{"priority", "shared trunk", "feature gate"} {
		if !strings.Contains(strings.ToLower(out), kw) {
			t.Fatalf("orthogonality note does not name %q:\n%s", kw, out)
		}
	}
	for _, k := range LifecycleEventKinds {
		if !strings.Contains(out, string(k)) {
			t.Fatalf("render missing kind %q in counts:\n%s", k, out)
		}
	}
	for _, kw := range []string{
		LifecycleLedgerSchema,        // durable schema tag in the header
		"gen/second-next -> gen/now", // the promotion transition
		"#3067 shipped",              // its evidence
		"wire format stable",         // its reason
		"gen/future -> retired",      // the retirement transition
		"old-axis: retired",          // the folded standing
	} {
		if !strings.Contains(out, kw) {
			t.Fatalf("render missing %q:\n%s", kw, out)
		}
	}
	if l.Render() != out {
		t.Fatal("Render is not deterministic")
	}
}

// TestEmptyLifecycleRender checks the empty case reads honestly: zero counts, a "(none)"
// events line, and no current state — never a blank or a panic.
func TestEmptyLifecycleRender(t *testing.T) {
	out := LifecycleLedger{}.Render()
	if !strings.Contains(out, "(none)") {
		t.Fatalf("empty ledger should render (none):\n%s", out)
	}
	if !strings.Contains(out, "promotion=0") {
		t.Fatalf("empty ledger should render zero counts:\n%s", out)
	}
}
