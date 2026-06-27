package abi

import "testing"

// uniEmitter is a v0.1-style observer: it does NOT implement EventSubscriber, so
// it receives every event kind (the universal default).
type uniEmitter struct{ id string }

func (uniEmitter) Emit(Event) {}

// selEmitter scopes itself to specific kinds via EventSubscriber.
type selEmitter struct {
	id   string
	subs []EventKind
}

func (selEmitter) Emit(Event)                   {}
func (s selEmitter) Subscriptions() []EventKind { return s.subs }

func emitterIDs(es []Emitter) []string {
	out := make([]string, 0, len(es))
	for _, e := range es {
		switch v := e.(type) {
		case uniEmitter:
			out = append(out, v.id)
		case selEmitter:
			out = append(out, v.id)
		}
	}
	return out
}

func eqStrs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestEmittersForFanout asserts the per-kind fan-out (Fix B): an event reaches the
// universal observers PLUS the selective ones that subscribed to that kind, in
// registration order; selective observers are NOT invoked for kinds they did not
// ask for.
func TestEmittersForFanout(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterEmitter(uniEmitter{"u1"})
	RegisterEmitter(selEmitter{"s_deny", []EventKind{EvDeny}})
	RegisterEmitter(uniEmitter{"u2"})
	RegisterEmitter(selEmitter{"t_sub_deny", []EventKind{EvSubmit, EvDeny}})

	cases := []struct {
		kind EventKind
		want []string
	}{
		{EvDeny, []string{"u1", "s_deny", "u2", "t_sub_deny"}}, // everyone wants it
		{EvSubmit, []string{"u1", "u2", "t_sub_deny"}},         // not s_deny
		{EvComplete, []string{"u1", "u2"}},                     // no selective wants it
	}
	for _, c := range cases {
		if got := emitterIDs(EmittersFor(c.kind)); !eqStrs(got, c.want) {
			t.Errorf("EmittersFor(%d) = %v, want %v", c.kind, got, c.want)
		}
	}
}

// TestEmittersForDefaultsToAll asserts backward compatibility: with no selective
// observer registered, EVERY kind (including an unknown one) fans out to ALL
// observers in registration order — exactly the v0.1 behavior.
func TestEmittersForDefaultsToAll(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	RegisterEmitter(uniEmitter{"a"})
	RegisterEmitter(uniEmitter{"b"})
	for _, k := range []EventKind{EvSubmit, EvDeny, EvComplete, EvRungLabel, EventKind(9999)} {
		if got := emitterIDs(EmittersFor(k)); !eqStrs(got, []string{"a", "b"}) {
			t.Errorf("EmittersFor(%d) = %v, want all observers [a b]", k, got)
		}
	}
}

// TestEmittersForZeroAlloc proves the fan-out itself stays allocation-free on both
// the indexed (selective) and fallback (universal-only) paths, with many observers
// registered — so emit() adds no per-call allocation no matter how many ideas tap
// the event stream.
func TestEmittersForZeroAlloc(t *testing.T) {
	ResetForTest()
	defer ResetForTest()
	for i := 0; i < 64; i++ {
		RegisterEmitter(uniEmitter{"u"})
		RegisterEmitter(selEmitter{"s", []EventKind{EvDeny}})
	}
	if a := testing.AllocsPerRun(200, func() { benchSink += len(EmittersFor(EvDeny)) }); a != 0 {
		t.Errorf("EmittersFor(indexed kind) allocates %.2f/op; want 0", a)
	}
	if a := testing.AllocsPerRun(200, func() { benchSink += len(EmittersFor(EvComplete)) }); a != 0 {
		t.Errorf("EmittersFor(fallback kind) allocates %.2f/op; want 0", a)
	}
}

// TestRedundantDecisionEvent pins the shared de-double rule the decision-stream
// folders (journal/harvest/trajectory) agree on: an EvDecide is redundant iff a
// following dedicated event re-carries the SAME decision — a deny (paired EvDeny),
// or a Submit-path require-witness intermediate (SeqNo!=0, resolved next). Every
// other event is the canonical single counting point.
func TestRedundantDecisionEvent(t *testing.T) {
	submitted := &ToolCall{SeqNo: 7}
	proposed := &ToolCall{SeqNo: 0} // pure Decide() path — no follow-up event
	cases := []struct {
		name string
		ev   Event
		want bool
	}{
		{"decide-deny is redundant (EvDeny follows)", Event{Kind: EvDecide, Call: submitted, Verdict: &Verdict{Kind: VerdictDeny}}, true},
		{"decide-deny on proposed call is still redundant", Event{Kind: EvDecide, Call: proposed, Verdict: &Verdict{Kind: VerdictDeny}}, true},
		{"decide-allow is canonical", Event{Kind: EvDecide, Call: submitted, Verdict: &Verdict{Kind: VerdictAllow}}, false},
		{"decide-transform is canonical", Event{Kind: EvDecide, Call: submitted, Verdict: &Verdict{Kind: VerdictTransform}}, false},
		{"require-witness intermediate on submit is redundant", Event{Kind: EvDecide, Call: submitted, Verdict: &Verdict{Kind: VerdictRequireWitness}}, true},
		{"require-witness on pure Decide is final, not redundant", Event{Kind: EvDecide, Call: proposed, Verdict: &Verdict{Kind: VerdictRequireWitness}}, false},
		{"the dedicated EvDeny is never skipped", Event{Kind: EvDeny, Call: submitted, Verdict: &Verdict{Kind: VerdictDeny}}, false},
		{"a result-side quarantine is never skipped", Event{Kind: EvQuarantine, Call: submitted, Verdict: &Verdict{Kind: VerdictQuarantine}}, false},
		{"nil verdict is not redundant", Event{Kind: EvDecide, Call: submitted}, false},
	}
	for _, tc := range cases {
		if got := RedundantDecisionEvent(tc.ev); got != tc.want {
			t.Errorf("%s: RedundantDecisionEvent = %v, want %v", tc.name, got, tc.want)
		}
	}
}
