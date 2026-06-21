package vdso

// revoke_test.go — the integrity-direction eraser (revoke.go): refutation evicts every
// consumer of a witness, refuses re-admission, and is orthogonal to the consistency
// clock. These are the C4 (causal refutation eviction) tests the Part B residue calls
// for: two reads under one witness, one refuting event, both evicted.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// roCallW is roCall plus an external world-state witness in Meta (the read declares the
// git commit / blob / lease epoch it is valid under).
func roCallW(tool, args, witness string) *abi.ToolCall {
	c := roCall(tool, args)
	c.Meta["witness"] = witness
	return c
}

// A refutation of one witness evicts EVERY entry admitted under it — the causal
// consumer-set eviction — and leaves a sibling witness's entry warm. This is the
// "two sessions, one refuting commit, both evicted" shape from the Part B residue.
func TestRevoke_EvictsAllConsumersOfWitness(t *testing.T) {
	v := New(32)
	a := roCallW("get_doc", `{"id":"a"}`, "commit-w1")
	b := roCallW("get_doc", `{"id":"b"}`, "commit-w1")
	c := roCallW("get_doc", `{"id":"c"}`, "commit-w2")

	fillAndExpectHit(t, v, a, `{"body":"A"}`)
	fillAndExpectHit(t, v, b, `{"body":"B"}`)
	fillAndExpectHit(t, v, c, `{"body":"C"}`)

	if got := v.Revoke("commit-w1"); got != 2 {
		t.Fatalf("Revoke(commit-w1) evicted=%d, want 2 (both w1 consumers)", got)
	}
	if hits(t, v, a) || hits(t, v, b) {
		t.Errorf("a w1 entry still hits after its witness was refuted (stale/poison serve)")
	}
	if !hits(t, v, c) {
		t.Errorf("the w2 entry was evicted by a w1 refutation — eviction must be witness-targeted, not a flush")
	}
	if v.Revocations() != 1 {
		t.Errorf("Revocations()=%d, want 1", v.Revocations())
	}
}

// Revocation is ORTHOGONAL to the consistency clock: it strands an entry with NO write
// having occurred (worldVer unchanged), and advances the integrity clock instead. This
// is the property worldVer alone cannot express — a refutation is not a write.
func TestRevoke_OrthogonalToWorldVersion(t *testing.T) {
	v := New(8)
	r := roCallW("get_policy", `{"k":"refund"}`, "lease-7")
	fillAndExpectHit(t, v, r, `{"fee":50}`)

	wvBefore := v.WorldVersion()
	teBefore := v.TrustEpoch()

	v.Revoke("lease-7")

	if hits(t, v, r) {
		t.Errorf("entry served after its witness was refuted")
	}
	if v.WorldVersion() != wvBefore {
		t.Errorf("WorldVersion advanced on a refutation (=%d, was %d) — a refutation is NOT a write", v.WorldVersion(), wvBefore)
	}
	if v.TrustEpoch() != teBefore+1 {
		t.Errorf("TrustEpoch=%d, want %d (a refutation must advance the integrity clock)", v.TrustEpoch(), teBefore+1)
	}
}

// A refuted witness must REFUSE re-admission: the durable CAS makes the bytes
// content-stable, so without the gate the next read would silently repopulate the
// evicted (poisoned) entry.
func TestRevoke_RefusesReAdmitUnderRefutedWitness(t *testing.T) {
	v := New(8)
	r := roCallW("get_doc", `{"id":"x"}`, "commit-bad")

	v.Revoke("commit-bad") // refute first (no entry yet) — still latches
	if !v.Revoked("commit-bad") {
		t.Fatalf("witness not marked refuted after Revoke")
	}
	// A later completion under the refuted witness must NOT fill the cache.
	v.Emit(completeEvent(r, `{"body":"poison"}`))
	if hits(t, v, r) {
		t.Errorf("read re-admitted under a refuted witness — the CAS silently repopulated the poison")
	}
}

// The revoked-witness ledger is bounded. Once exact history overflows, the vDSO
// fails closed for witness-bearing entries whose witness is not in the retained
// exact set: they might be one of the trimmed refutations.
func TestRevoke_BoundsWitnessLedgerFailClosedOnOverflow(t *testing.T) {
	v := New(8)
	v.SetRevokedWitnessLimit(2)

	safe := roCallW("get_doc", `{"id":"safe"}`, "commit-safe")
	fillAndExpectHit(t, v, safe, `{"body":"safe"}`)

	v.Revoke("commit-w1")
	v.Revoke("commit-w2")
	if v.RevocationOverflow() {
		t.Fatal("overflow before cap pressure")
	}
	if got := v.RevokedWitnesses(); got != 2 {
		t.Fatalf("retained revoked witnesses=%d, want 2", got)
	}

	v.Revoke("commit-w3")
	if !v.RevocationOverflow() {
		t.Fatal("revoked-witness overflow bit not set after cap pressure")
	}
	if got := v.RevokedWitnesses(); got != 2 {
		t.Fatalf("retained revoked witnesses=%d, want bounded cap 2", got)
	}
	if !v.Revoked("commit-w1") {
		t.Fatal("trimmed revoked witness did not fail closed")
	}
	if !v.Revoked("commit-never-seen") {
		t.Fatal("unknown witness did not fail closed after revoked-ledger overflow")
	}
	if hits(t, v, safe) {
		t.Fatal("witness-bearing entry survived overflow despite unknown-witness fail-closed policy")
	}

	unknown := roCallW("get_doc", `{"id":"unknown"}`, "commit-never-seen")
	v.Emit(completeEvent(unknown, `{"body":"unknown"}`))
	if hits(t, v, unknown) {
		t.Fatal("unknown witness re-admitted after revoked-ledger overflow")
	}

	plain := roCall("get_doc", `{"id":"plain"}`)
	fillAndExpectHit(t, v, plain, `{"body":"plain"}`)
	if !hits(t, v, plain) {
		t.Fatal("witnessless cache entry should remain governed only by the consistency eraser")
	}
}

// An entry with NO external witness is untouched by any refutation — the witness binding
// is purely additive (v0.1 entries behave exactly as before).
func TestRevoke_NoWitnessEntriesUnaffected(t *testing.T) {
	v := New(8)
	r := roCall("get_doc", `{"id":"plain"}`) // no witness in Meta
	fillAndExpectHit(t, v, r, `{"body":"plain"}`)

	v.Revoke("commit-w1") // refute an unrelated witness
	v.Revoke("")          // empty-witness no-op

	if !hits(t, v, r) {
		t.Errorf("a witness-less entry was evicted by an unrelated refutation — binding must be additive")
	}
}

// The empty witness is a no-op (no eviction, no epoch bump, no event).
func TestRevoke_EmptyWitnessNoOp(t *testing.T) {
	v := New(8)
	if got := v.Revoke(""); got != 0 {
		t.Errorf("Revoke(\"\")=%d, want 0", got)
	}
	if v.TrustEpoch() != 0 || v.Revocations() != 0 {
		t.Errorf("empty-witness Revoke perturbed the integrity clock (epoch=%d, count=%d)", v.TrustEpoch(), v.Revocations())
	}
}

// A refutation publishes one typed Revocation on the coherence bus, naming the witness,
// the local eviction count, and the post-bump integrity epoch — the cross-agent /
// cross-process "what changed" signal a remote private cache invalidates on.
func TestRevoke_PublishesOnCoherenceBus(t *testing.T) {
	v := New(8)
	var got []Revocation
	cancel := v.SubscribeRevocations(func(rv Revocation) { got = append(got, rv) })
	defer cancel()

	a := roCallW("get_doc", `{"id":"a"}`, "commit-w1")
	b := roCallW("get_doc", `{"id":"b"}`, "commit-w1")
	fillAndExpectHit(t, v, a, `{"body":"A"}`)
	fillAndExpectHit(t, v, b, `{"body":"B"}`)

	v.Revoke("commit-w1")

	if len(got) != 1 {
		t.Fatalf("subscriber saw %d revocations, want 1", len(got))
	}
	rv := got[0]
	if rv.Witness != "commit-w1" {
		t.Errorf("Revocation.Witness=%q, want commit-w1", rv.Witness)
	}
	if rv.Evicted != 2 {
		t.Errorf("Revocation.Evicted=%d, want 2", rv.Evicted)
	}
	if rv.TrustEpoch != 1 {
		t.Errorf("Revocation.TrustEpoch=%d, want 1", rv.TrustEpoch)
	}
	if rv.Seq == 0 {
		t.Errorf("Revocation.Seq=0, want a nonzero coherence-bus sequence")
	}

	// After cancel, a further refutation is not delivered.
	cancel()
	v.Revoke("commit-w2")
	if len(got) != 1 {
		t.Errorf("cancel() did not unsubscribe — saw %d events", len(got))
	}
}

// Sanity: Lookup under a refuted witness returns ok=false (-> the engine, a fresh call) —
// the sound direction of the coherence tradeoff.
func TestRevoke_Sound_RevokedNeverServed(t *testing.T) {
	v := New(8)
	r := roCallW("get_doc", `{"id":"a"}`, "commit-w1")
	fillAndExpectHit(t, v, r, `{"body":"A"}`)
	v.Revoke("commit-w1")
	if _, ok := v.Lookup(context.Background(), r); ok {
		t.Errorf("Lookup served a refuted entry — must miss so the engine is consulted fresh")
	}
}
