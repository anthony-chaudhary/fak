package fleetbus

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// countingApplier records every directive it was actually asked to apply, which is
// how these tests tell "answered" apart from "applied".
type countingApplier struct {
	seen []string
	out  Outcome
}

func (c *countingApplier) Apply(d Directive) Outcome {
	c.seen = append(c.seen, d.ID)
	return c.out
}

func publish(t *testing.T, b Bus, issuer string, op Op, payload string, sel Selector, ttl time.Duration, now time.Time) Directive {
	t.Helper()
	d, r := NewDirective(issuer, op, payload, sel, ttl, "", now)
	if r != nil {
		t.Fatalf("NewDirective: %v", r)
	}
	if err := b.Publish(d); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	return d
}

func TestDrainAppliesOnceAcrossRepeatedDrains(t *testing.T) {
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	d := publish(t, b, "op", "steer", "go", Selector{All: true}, time.Minute, testNow)

	ap := &countingApplier{out: OutcomeApplied("steered 3 sessions", 3)}
	first, err := Drain(b, self, ap, testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if first.Matched != 1 || first.Applied != 1 || len(first.Acks) != 1 {
		t.Fatalf("first drain = %+v, want matched=1 applied=1 with one ack", first)
	}
	if first.Acks[0].Witness != "steered 3 sessions" || first.Acks[0].Affected != 3 {
		t.Fatalf("the ack lost the applier's witness: %+v", first.Acks[0])
	}

	// At-least-once DELIVERY: the directive is still in the log. Exactly-once
	// APPLICATION: the applier must not be called a second time.
	second, err := Drain(b, self, ap, testNow.Add(time.Second))
	if err != nil {
		t.Fatalf("second Drain: %v", err)
	}
	if second.Matched != 1 || second.AlreadyDone != 1 || second.Applied != 0 || len(second.Acks) != 0 {
		t.Fatalf("second drain = %+v, want the redelivery recognised and nothing re-applied", second)
	}
	if len(ap.seen) != 1 {
		t.Fatalf("applier ran %d times across two drains, want 1", len(ap.seen))
	}
	if acks, _ := b.Acks(d.ID); len(acks) != 1 {
		t.Fatalf("%d acks on the bus, want 1", len(acks))
	}
}

func TestDrainSkipsUnaddressedDirectivesSilently(t *testing.T) {
	b := testBus(t)
	self := testInstance(t, "serve-1", "box-a", "serve", testNow)
	mine := publish(t, b, "op", "pause", "", Selector{Machine: []string{"box-a"}}, time.Minute, testNow)
	theirs := publish(t, b, "op", "pause", "", Selector{Machine: []string{"box-b"}}, time.Minute, testNow.Add(time.Second))

	ap := &countingApplier{out: OutcomeApplied("paused", 1)}
	rep, err := Drain(b, self, ap, testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rep.Seen != 2 || rep.Matched != 1 {
		t.Fatalf("drain = %+v, want seen=2 matched=1", rep)
	}
	if len(ap.seen) != 1 || ap.seen[0] != mine.ID {
		t.Fatalf("applier saw %v, want only %s", ap.seen, mine.ID)
	}
	// An unaddressed directive draws no ack at all. Silence is safe here only
	// because the control point derives the expected set from the roster.
	if acks, _ := b.Acks(theirs.ID); len(acks) != 0 {
		t.Fatalf("acked a directive addressed to another machine: %+v", acks)
	}
}

func TestDrainAcksAnExpiredDirectiveWithoutApplyingIt(t *testing.T) {
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	d := publish(t, b, "op", "steer", "go", Selector{All: true}, 10*time.Second, testNow)

	ap := &countingApplier{out: OutcomeApplied("steered", 1)}
	rep, err := Drain(b, self, ap, testNow.Add(time.Minute))
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if len(ap.seen) != 0 {
		t.Fatal("an expired directive was applied — a stale `go` firing late is the failure the TTL exists to stop")
	}
	if rep.Expired != 1 || rep.Applied != 0 {
		t.Fatalf("drain = %+v, want expired=1 applied=0", rep)
	}
	acks, _ := b.Acks(d.ID)
	if len(acks) != 1 || acks[0].Status != AckExpired || acks[0].Reason != Expired {
		t.Fatalf("acks = %+v, want one expired ack carrying %s", acks, Expired)
	}
	// "Nobody was listening in time" must stay distinguishable from "nobody ever
	// answered", so the lapse is a recorded answer, not silence.
	if !strings.Contains(acks[0].Detail, "not applied") {
		t.Errorf("expired ack detail = %q, want it to say the directive was not applied", acks[0].Detail)
	}
}

func TestDrainNormalizesATokenlessRefusal(t *testing.T) {
	// An ack that fails validation is never written, which would turn a refusal
	// into silence — the exact phantom this package prevents. So a refusal with no
	// token is repaired loudly rather than dropped.
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	d := publish(t, b, "op", "steer", "go", Selector{All: true}, time.Minute, testNow)

	rep, err := Drain(b, self, ApplierFunc(func(Directive) Outcome {
		return Outcome{Status: AckRefused, Detail: "no owned loop"}
	}), testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rep.Refused != 1 {
		t.Fatalf("drain = %+v, want refused=1", rep)
	}
	acks, _ := b.Acks(d.ID)
	if len(acks) != 1 {
		t.Fatalf("acks = %+v, want the refusal recorded", acks)
	}
	if acks[0].Reason != ApplyRefused {
		t.Fatalf("reason = %q, want %q", acks[0].Reason, ApplyRefused)
	}
	if !strings.Contains(acks[0].Detail, "no owned loop") {
		t.Errorf("detail = %q, want the applier's own words carried verbatim", acks[0].Detail)
	}
}

func TestDrainRefusesASilentApplier(t *testing.T) {
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	d := publish(t, b, "op", "nonsense", "", Selector{All: true}, time.Minute, testNow)

	rep, err := Drain(b, self, ApplierFunc(func(Directive) Outcome { return Outcome{} }), testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rep.Refused != 1 {
		t.Fatalf("drain = %+v, want a zero Outcome to read as a refusal, never as success", rep)
	}
	acks, _ := b.Acks(d.ID)
	if len(acks) != 1 || acks[0].Status != AckRefused || acks[0].Reason != ApplyRefused {
		t.Fatalf("acks = %+v, want one tokened refusal", acks)
	}
}

func TestDrainCarriesAnUnknownOpRefusalThrough(t *testing.T) {
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	d := publish(t, b, "op", "teleport", "", Selector{All: true}, time.Minute, testNow)

	_, err := Drain(b, self, ApplierFunc(func(d Directive) Outcome {
		return OutcomeRefused(UnknownOp, "op %q is outside this applier's vocabulary", d.Op)
	}), testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	acks, _ := b.Acks(d.ID)
	if len(acks) != 1 || acks[0].Reason != UnknownOp {
		t.Fatalf("acks = %+v, want %s — an op nobody understands must be visible, not absorbed into outstanding", acks, UnknownOp)
	}
}

func TestDrainKeepsGoingPastOneUnwritableAck(t *testing.T) {
	// One directive that cannot be answered must not strand the rest of the pass:
	// a fleet-wide `pause` blocked by a single bad row would be worse than the
	// transport failure that caused it. The failure is collected, never swallowed.
	b := testBus(t)
	self := testInstance(t, "serve-1", "box", "serve", testNow)
	bad := publish(t, b, "op", "steer", "one", Selector{All: true}, time.Minute, testNow)
	good := publish(t, b, "op", "steer", "two", Selector{All: true}, time.Minute, testNow.Add(time.Second))

	wrapped := &failingAckBus{DirBus: b, failFor: bad.ID}
	rep, err := Drain(wrapped, self, ApplierFunc(func(Directive) Outcome { return OutcomeApplied("ok", 1) }), testNow)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rep.Matched != 2 {
		t.Fatalf("drain = %+v, want both directives attempted", rep)
	}
	if len(rep.Errors) != 1 || !strings.Contains(rep.Errors[0], bad.ID) {
		t.Fatalf("errors = %v, want the one unwritable ack named", rep.Errors)
	}
	if acks, _ := b.Acks(good.ID); len(acks) != 1 {
		t.Fatalf("the second directive was never acked: %+v", acks)
	}
}

// failingAckBus makes one directive's ack unwritable so the drain's
// keep-going-and-report behaviour is observable.
type failingAckBus struct {
	*DirBus
	failFor string
}

func (f *failingAckBus) Ack(a Ack) error {
	if a.Directive == f.failFor {
		return errors.New("disk on fire")
	}
	return f.DirBus.Ack(a)
}
