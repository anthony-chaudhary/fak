package dosadapter

// adversarial_1619_test.go — independent, adversarial test author for fak-private#1619.
//
// This file was written from the SPEC ONLY, by an author who did NOT write the
// implementation. It deliberately exercises edges the happy-path regression
// (lease_generation_test.go) does not: the exact staleness boundary, a negative
// maxAge, a zero `now` against a client that has a clock, shared/exclusive
// matrix behaviour through ArbitrateSet, disjoint-tree no-false-refusal,
// multi-holder single-intersection refusal + token identity, constructor
// equivalence, Describe() content, the empty-but-unreadable fail-closed case,
// and holder-order independence.
//
// Helper names are prefixed adv1619 to avoid colliding with helpers/tests in the
// sibling files of this package.

import (
	"errors"
	"strings"
	"testing"
	"time"
)

var adv1619Now = time.Date(2026, 9, 21, 15, 0, 0, 0, time.UTC)

func adv1619Noon() time.Time { return adv1619Now }

// adv1619Exclusive builds a bare exclusive holder/request over one tree.
func adv1619Exclusive(id, lane string, tree ...string) LeaseRequest {
	return LeaseRequest{
		ID:       id,
		Lane:     lane,
		LaneKind: "leaf",
		LockMode: LockModeExclusive,
		Tree:     tree,
	}
}

// adv1619Shared builds a shared holder/request over one tree.
func adv1619Shared(id, lane string, tree ...string) LeaseRequest {
	return LeaseRequest{
		ID:       id,
		Lane:     lane,
		LaneKind: "leaf",
		LockMode: LockModeShared,
		Tree:     tree,
	}
}

// TestAdversarial1619StaleBoundaryAtExactlyMaxAge pins the staleness boundary.
//
// DEFENSIBLE EXPECTATION: a set aged EXACTLY maxAge (now - observed == maxAge)
// must NOT be stale. Reasoning: `Stale(now, maxAge)` is a freshness AGE LIMIT,
// and the natural, conventional reading of an age limit is `age > maxAge` is
// stale (i.e. a row is fresh while `age <= maxAge`). A maxAge is a TTL: a witness
// exactly at its TTL boundary is still valid (cf. this package's own freshness
// convention in CheckFreshness, which expires only when `at.After(ValidUntil)`,
// never `at.Equal`). Choosing strict `>` also keeps the boundary deterministic
// and half-open, so a caller that samples at exactly the limit cannot flap.
//
// A negative maxAge is a nonsensical/absurd limit; the fail-closed contract says
// unreadable/absent evidence must read as stale, so a negative maxAge must make
// everything stale (you cannot claim a negative freshness window is satisfied).
func TestAdversarial1619StaleBoundaryAtExactlyMaxAge(t *testing.T) {
	maxAge := 5 * time.Minute

	// Exactly at the limit: age == maxAge -> fresh (not stale).
	atLimit := NewLiveLeaseSet(nil, adv1619Now.Add(-maxAge), 11, LeaseSourceLiveJournal)
	if atLimit.Stale(adv1619Now, maxAge) {
		t.Errorf("set aged exactly maxAge reported stale; want fresh (age==maxAge is within the window): %s", atLimit.Describe())
	}

	// One nanosecond past the limit: stale.
	past := NewLiveLeaseSet(nil, adv1619Now.Add(-maxAge-time.Nanosecond), 11, LeaseSourceLiveJournal)
	if !past.Stale(adv1619Now, maxAge) {
		t.Errorf("set aged maxAge+1ns reported fresh; want stale: %s", past.Describe())
	}

	// One nanosecond inside the limit: fresh.
	inside := NewLiveLeaseSet(nil, adv1619Now.Add(-maxAge+time.Nanosecond), 11, LeaseSourceLiveJournal)
	if inside.Stale(adv1619Now, maxAge) {
		t.Errorf("set aged maxAge-1ns reported stale; want fresh: %s", inside.Describe())
	}

	// Negative maxAge: fail closed -> always stale, even for a just-observed set.
	negative := NewLiveLeaseSet(nil, adv1619Now, 12, LeaseSourceLiveJournal)
	if !negative.Stale(adv1619Now, -time.Minute) {
		t.Errorf("negative maxAge did not fail closed; a non-positive window must read stale: %s", negative.Describe())
	}
}

// TestAdversarial1619ZeroNowStampsProvenanceAndVerdictCorrectly asserts that a
// zero `now` handed to ArbitrateSet does NOT corrupt provenance: the outcome must
// still carry the SET's own Source/Generation/ObservedUnix (a property of the
// snapshot, not of the caller's clock), and the verdict must follow the holders.
func TestAdversarial1619ZeroNowStampsProvenanceAndVerdictCorrectly(t *testing.T) {
	// Client HAS a clock; the caller then passes the zero time explicitly.
	client := NewAdapterClient(WithClock(adv1619Noon), WithIssuer("fak/adversarial-1619"))

	req := adv1619Exclusive("adv-zero-now-req", "model", "internal/model/**")
	holder := adv1619Exclusive("adv-zero-now-holder", "native-rocm-model-test", "internal/model/**")

	set := DeriveLiveLeaseSet([]LeaseRequest{holder}, adv1619Now.Add(-time.Minute), 21, LeaseSourceSnapshot)

	outcome, err := client.ArbitrateSet(req, set, time.Time{})
	if !errors.Is(err, ErrArbitrationRefused) {
		t.Fatalf("ArbitrateSet(zero now) over conflicting holder error = %v, want ErrArbitrationRefused", err)
	}
	if outcome.Outcome != OutcomeRefuse {
		t.Errorf("ArbitrateSet(zero now) outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
	}
	// Provenance is a property of the set, not of `now`; a zero now must not blank it.
	if outcome.LeaseSource != LeaseSourceSnapshot {
		t.Errorf("ArbitrateSet(zero now) LeaseSource = %q, want %q (zero now must not blank set provenance)",
			outcome.LeaseSource, LeaseSourceSnapshot)
	}
	if outcome.LeaseGeneration != 21 {
		t.Errorf("ArbitrateSet(zero now) LeaseGeneration = %d, want 21", outcome.LeaseGeneration)
	}
	if outcome.LeaseObservedUnix != set.ObservedUnix {
		t.Errorf("ArbitrateSet(zero now) LeaseObservedUnix = %d, want %d", outcome.LeaseObservedUnix, set.ObservedUnix)
	}

	// And a zero now must not spuriously refuse a genuinely empty live set.
	empty := DeriveLiveLeaseSet(nil, adv1619Now, 22, LeaseSourceLiveJournal)
	if got, err := client.ArbitrateSet(adv1619Exclusive("adv-zero-now-free", "docs", "docs/**"), empty, time.Time{}); err != nil {
		t.Fatalf("ArbitrateSet(zero now) over empty set error = %v, want nil", err)
	} else if got.Outcome != OutcomeAcquire {
		t.Errorf("ArbitrateSet(zero now) over empty set outcome = %q, want %q", got.Outcome, OutcomeAcquire)
	}
}

// TestAdversarial1619SharedExclusiveMatrixThroughArbitrateSet drives the full
// lock-mode matrix through ArbitrateSet (not the bare CheckDisjoint helper).
func TestAdversarial1619SharedExclusiveMatrixThroughArbitrateSet(t *testing.T) {
	client := NewAdapterClient(WithClock(adv1619Noon))
	tree := "internal/gateway/**"

	t.Run("two shared holders on same tree do not refuse each other", func(t *testing.T) {
		set := DeriveLiveLeaseSet([]LeaseRequest{
			adv1619Shared("adv-sh-1", "gw-a", tree),
		}, adv1619Now, 31, LeaseSourceLiveJournal)
		out, err := client.ArbitrateSet(adv1619Shared("adv-sh-2", "gw-b", tree), set, adv1619Now)
		if err != nil {
			t.Fatalf("shared/shared ArbitrateSet error = %v, want nil", err)
		}
		if out.Outcome != OutcomeAcquire {
			t.Errorf("shared/shared outcome = %q, want %q", out.Outcome, OutcomeAcquire)
		}
	})

	t.Run("exclusive request against shared holder on same tree refuses", func(t *testing.T) {
		set := DeriveLiveLeaseSet([]LeaseRequest{
			adv1619Shared("adv-sh-3", "gw-c", tree),
		}, adv1619Now, 32, LeaseSourceLiveJournal)
		out, err := client.ArbitrateSet(adv1619Exclusive("adv-ex-on-sh", "gw-d", tree), set, adv1619Now)
		if !errors.Is(err, ErrArbitrationRefused) {
			t.Fatalf("exclusive-on-shared error = %v, want ErrArbitrationRefused", err)
		}
		if out.Outcome != OutcomeRefuse {
			t.Errorf("exclusive-on-shared outcome = %q, want %q", out.Outcome, OutcomeRefuse)
		}
		if out.Reason.Token != ReasonCollisionRisk {
			t.Errorf("exclusive-on-shared token = %q, want %q", out.Reason.Token, ReasonCollisionRisk)
		}
	})

	t.Run("shared request against exclusive holder on same tree refuses", func(t *testing.T) {
		set := DeriveLiveLeaseSet([]LeaseRequest{
			adv1619Exclusive("adv-ex-1", "gw-e", tree),
		}, adv1619Now, 33, LeaseSourceLiveJournal)
		out, err := client.ArbitrateSet(adv1619Shared("adv-sh-on-ex", "gw-f", tree), set, adv1619Now)
		if !errors.Is(err, ErrArbitrationRefused) {
			t.Fatalf("shared-on-exclusive error = %v, want ErrArbitrationRefused", err)
		}
		if out.Outcome != OutcomeRefuse {
			t.Errorf("shared-on-exclusive outcome = %q, want %q", out.Outcome, OutcomeRefuse)
		}
	})
}

// TestAdversarial1619DisjointTreesSameSetAcquires guards against a false refusal:
// holders whose trees are provably disjoint must not block an exclusive request.
func TestAdversarial1619DisjointTreesSameSetAcquires(t *testing.T) {
	client := NewAdapterClient(WithClock(adv1619Noon))

	set := DeriveLiveLeaseSet([]LeaseRequest{
		adv1619Exclusive("adv-disjoint-1", "gateway", "internal/gateway/**"),
		adv1619Exclusive("adv-disjoint-2", "dosadapter", "internal/dosadapter/**"),
	}, adv1619Now, 41, LeaseSourceLiveJournal)

	out, err := client.ArbitrateSet(
		adv1619Exclusive("adv-disjoint-req", "docs", "docs/**"),
		set, adv1619Now,
	)
	if err != nil {
		t.Fatalf("ArbitrateSet() over disjoint holders error = %v, want nil (false refusal)", err)
	}
	if out.Outcome != OutcomeAcquire {
		t.Errorf("ArbitrateSet() over disjoint holders outcome = %q, want %q", out.Outcome, OutcomeAcquire)
	}
}

// TestAdversarial1619OneIntersectingHolderAmongManyRefuses asserts that a single
// intersecting holder among several disjoint peers is sufficient to refuse, and
// that the structured token is exactly COLLISION_RISK.
func TestAdversarial1619OneIntersectingHolderAmongManyRefuses(t *testing.T) {
	client := NewAdapterClient(WithClock(adv1619Noon))

	set := DeriveLiveLeaseSet([]LeaseRequest{
		adv1619Exclusive("adv-many-a", "gateway", "internal/gateway/**"),
		adv1619Exclusive("adv-many-b", "docs", "docs/**"),
		// The one intersector:
		adv1619Exclusive("adv-many-hit", "native-rocm-model-test", "internal/model/**"),
		adv1619Exclusive("adv-many-c", "cmd", "cmd/fak/**"),
	}, adv1619Now, 51, LeaseSourceLiveJournal)

	out, err := client.ArbitrateSet(
		adv1619Exclusive("adv-many-req", "model", "internal/model/**"),
		set, adv1619Now,
	)
	if !errors.Is(err, ErrArbitrationRefused) {
		t.Fatalf("ArbitrateSet() with one intersector error = %v, want ErrArbitrationRefused", err)
	}
	if out.Outcome != OutcomeRefuse {
		t.Errorf("ArbitrateSet() with one intersector outcome = %q, want %q", out.Outcome, OutcomeRefuse)
	}
	if out.Reason.Token != ReasonCollisionRisk {
		t.Errorf("ArbitrateSet() with one intersector token = %q, want %q", out.Reason.Token, ReasonCollisionRisk)
	}
	if out.Reason.Category != CategoryMisroute {
		t.Errorf("COLLISION_RISK category = %q, want %q", out.Reason.Category, CategoryMisroute)
	}
	if !out.Reason.Refusal {
		t.Errorf("COLLISION_RISK Refusal = false, want true (blocking)")
	}
}

// TestAdversarial1619DeriveAndNewConstructorEquivalent asserts the two constructors
// produce observably identical snapshots for identical inputs.
func TestAdversarial1619DeriveAndNewConstructorEquivalent(t *testing.T) {
	holders := []LeaseRequest{
		adv1619Exclusive("adv-eq-1", "gateway", "internal/gateway/**"),
		adv1619Shared("adv-eq-2", "docs", "docs/**"),
	}
	observed := adv1619Now.Add(-2 * time.Minute)

	derived := DeriveLiveLeaseSet(holders, observed, 61, LeaseSourceSnapshot)
	made := NewLiveLeaseSet(holders, observed, 61, LeaseSourceSnapshot)

	if derived.Source != made.Source {
		t.Errorf("Source differs: Derive=%q New=%q", derived.Source, made.Source)
	}
	if derived.Generation != made.Generation {
		t.Errorf("Generation differs: Derive=%d New=%d", derived.Generation, made.Generation)
	}
	if derived.ObservedUnix != made.ObservedUnix {
		t.Errorf("ObservedUnix differs: Derive=%d New=%d", derived.ObservedUnix, made.ObservedUnix)
	}
	if d, m := derived.Stale(adv1619Now, time.Minute), made.Stale(adv1619Now, time.Minute); d != m {
		t.Errorf("Stale() differs for identical snapshots: Derive=%v New=%v", d, m)
	}
}

// TestAdversarial1619DescribeIsNonEmptyAndNamesSource asserts Describe() is a
// non-empty diagnostic that mentions the source token.
func TestAdversarial1619DescribeIsNonEmptyAndNamesSource(t *testing.T) {
	cases := []struct {
		source LeaseSource
	}{
		{LeaseSourceSnapshot},
		{LeaseSourceLiveJournal},
		{LeaseSourceCaller},
		{LeaseSourceInMemory},
		{LeaseSourceUnreadable},
	}
	for _, tc := range cases {
		set := NewLiveLeaseSet(nil, adv1619Now, 71, tc.source)
		desc := set.Describe()
		if strings.TrimSpace(desc) == "" {
			t.Errorf("Describe() empty for source %q", tc.source)
			continue
		}
		if !strings.Contains(desc, string(tc.source)) {
			t.Errorf("Describe() = %q, want it to contain source token %q", desc, tc.source)
		}
	}
}

// TestAdversarial1619EmptyUnreadableSetIsStaleAndFailClosed pins the two facts
// the fail-closed contract demands for an EMPTY set that could not be read.
//
// EXPECTATION: the set must report Stale(true) (no evidence of freshness), yet
// arbitration against it must ACQUIRE, because fail-closed refusal is defined by
// the presence of a CONFLICTING HOLDER, not by the emptiness of the snapshot. An
// unreadable snapshot with zero holders carries no evidence of a collision, so
// refusing would be a different (and wrong) failure mode: it would deadlock every
// caller whenever the journal is merely unreadable. The set's staleness is the
// signal for the caller to RE-DERIVE; it is not itself a collision. (Contrast the
// spec's other fail-closed clause: a STALE set that DOES contain a conflicting
// exclusive holder must still REFUSE — covered by the sibling regression test.)
func TestAdversarial1619EmptyUnreadableSetIsStaleAndFailClosed(t *testing.T) {
	client := NewAdapterClient(WithClock(adv1619Noon))

	unreadable := NewLiveLeaseSet(nil, time.Time{}, 0, LeaseSourceUnreadable)
	if !unreadable.Stale(adv1619Now, 5*time.Minute) {
		t.Fatalf("empty unreadable set not reported stale; must fail closed: %s", unreadable.Describe())
	}

	out, err := client.ArbitrateSet(
		adv1619Exclusive("adv-unreadable-req", "docs", "docs/**"),
		unreadable, adv1619Now,
	)
	if err != nil {
		t.Fatalf("ArbitrateSet() over empty unreadable set error = %v, want nil (no collision evidence)", err)
	}
	if out.Outcome != OutcomeAcquire {
		t.Errorf("ArbitrateSet() over empty unreadable set outcome = %q, want %q", out.Outcome, OutcomeAcquire)
	}
	if out.LeaseSource != LeaseSourceUnreadable {
		t.Errorf("outcome LeaseSource = %q, want %q (provenance must be honest about the unreadable source)",
			out.LeaseSource, LeaseSourceUnreadable)
	}
}

// TestAdversarial1619HolderOrderIndependence asserts the verdict and the deciding
// provenance do not depend on the order in which holders appear in the set.
func TestAdversarial1619HolderOrderIndependence(t *testing.T) {
	client := NewAdapterClient(WithClock(adv1619Noon))

	a := adv1619Exclusive("adv-ord-a", "gateway", "internal/gateway/**")
	b := adv1619Exclusive("adv-ord-b", "model", "internal/model/**")
	c := adv1619Exclusive("adv-ord-c", "docs", "docs/**")

	forward := DeriveLiveLeaseSet([]LeaseRequest{a, b, c}, adv1619Now, 81, LeaseSourceLiveJournal)
	reverse := DeriveLiveLeaseSet([]LeaseRequest{c, b, a}, adv1619Now, 81, LeaseSourceLiveJournal)

	req := adv1619Exclusive("adv-ord-req", "model-2", "internal/model/**")

	outF, errF := client.ArbitrateSet(req, forward, adv1619Now)
	outR, errR := client.ArbitrateSet(req, reverse, adv1619Now)

	if (errF == nil) != (errR == nil) {
		t.Fatalf("order changed error-ness: forward err=%v reverse err=%v", errF, errR)
	}
	if outF.Outcome != outR.Outcome {
		t.Errorf("order changed outcome: forward=%q reverse=%q", outF.Outcome, outR.Outcome)
	}
	if outF.Reason.Token != outR.Reason.Token {
		t.Errorf("order changed reason token: forward=%q reverse=%q", outF.Reason.Token, outR.Reason.Token)
	}
	if !errors.Is(errF, ErrArbitrationRefused) || !errors.Is(errR, ErrArbitrationRefused) {
		t.Errorf("expected both orders to refuse on the intersecting holder; forward=%v reverse=%v", errF, errR)
	}

	// Disjoint request: both orders must acquire (no order-dependent false refusal).
	reqFree := adv1619Exclusive("adv-ord-free", "cmd", "cmd/fak/**")
	ofF, eF := client.ArbitrateSet(reqFree, forward, adv1619Now)
	ofR, eR := client.ArbitrateSet(reqFree, reverse, adv1619Now)
	if eF != nil || eR != nil {
		t.Fatalf("order changed disjoint verdict: forward err=%v reverse err=%v", eF, eR)
	}
	if ofF.Outcome != OutcomeAcquire || ofR.Outcome != OutcomeAcquire {
		t.Errorf("disjoint request not acquired in both orders: forward=%q reverse=%q", ofF.Outcome, ofR.Outcome)
	}
}
