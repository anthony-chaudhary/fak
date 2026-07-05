package main

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// stubKnownBadWitness swaps the injectable witness seam for a fixed verdict and restores
// it when the test ends — the same pattern as the dispatch-witness seams. `ok` is the
// green/red the resolve gate reads; capture receives the tree globs the witness was asked
// to prove over, so a test can assert resolve ran the witness against the SIGNATURE's tree.
func stubKnownBadWitness(t *testing.T, ok bool, capture *[]string) {
	t.Helper()
	prev := knownBadWitness
	t.Cleanup(func() { knownBadWitness = prev })
	knownBadWitness = func(dir, kind string, treeGlobs []string, commit string) knownBadWitnessResult {
		if capture != nil {
			*capture = append([]string(nil), treeGlobs...)
		}
		if ok {
			return knownBadWitnessResult{OK: true, Kind: kind, Detail: "stub green over " + strings.Join(treeGlobs, ",")}
		}
		return knownBadWitnessResult{OK: false, Kind: kind, Detail: "stub red over " + strings.Join(treeGlobs, ",")}
	}
}

// resolveJSON is the subset of the resolve verb's --json envelope the tests assert on.
type resolveJSON struct {
	OK        bool                   `json:"ok"`
	Signature string                 `json:"signature"`
	Witness   string                 `json:"witness"`
	Reason    string                 `json:"reason"`
	LeaseID   string                 `json:"lease_id"`
	Record    *knownbad.Record       `json:"record"`
	Lease     *leaseref.FenceVerdict `json:"lease"`
}

// TestKnownBadResolveRefusedWithoutWitness is one arm of the W6 (#2718) done condition: a
// resolve whose witness does NOT pass leaves the signature OPEN and is refused with the
// structured KNOWN_BAD_NOT_WITNESSED reason (exit leaserefRefused). Nothing is appended —
// the ledger's latest row is still open, so the parked fleet stays parked. A self-report
// (a red witness) never releases the fleet.
func TestKnownBadResolveRefusedWithoutWitness(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	const now = int64(1_700_000_000)

	sig := knownbad.Signature("build", []string{"internal/foo/**"}, "")
	var rb bytes.Buffer
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ledger", ledger,
	}, now); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}

	// The witness does NOT pass -> resolve must refuse and leave the signature open.
	var capturedTree []string
	stubKnownBadWitness(t, false, &capturedTree)

	var ob bytes.Buffer
	rc := runKnownBad(&ob, &ob, []string{
		"resolve", "--by", "fixer", "--dir", dir, "--ledger", ledger, "--json", sig,
	}, now)
	if rc != leaserefRefused {
		t.Fatalf("unwitnessed resolve rc=%d, want %d (refused) out=%q", rc, leaserefRefused, ob.String())
	}
	var res resolveJSON
	if err := json.Unmarshal(ob.Bytes(), &res); err != nil {
		t.Fatalf("resolve --json invalid: %v\nout=%q", err, ob.String())
	}
	if res.OK || res.Reason != reasonKnownBadNotWitnessed {
		t.Fatalf("unwitnessed resolve not refused with %s: %+v", reasonKnownBadNotWitnessed, res)
	}
	// The witness was run over the SIGNATURE's tree, not some ad-hoc set.
	if len(capturedTree) != 1 || capturedTree[0] != "internal/foo" {
		t.Errorf("witness ran over %v, want the signature's tree [internal/foo]", capturedTree)
	}

	// The ledger's signature is still LIVE (open) — the refused resolve appended nothing.
	records, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if _, live := knownbad.FindLatestLive(records, sig, now); !live {
		t.Fatalf("a refused resolve must leave the signature LIVE (open); it is not")
	}
	for _, r := range records {
		if r.Signature == sig && r.Resolved() {
			t.Fatalf("a refused resolve must not append a resolved row, found: %+v", r)
		}
	}
}

// TestKnownBadResolveReleasesOnWitness is the closing arm of the W6 done condition and the
// skip->dispatchable transcript: a recorded+claimed signature is resolved on a GREEN
// witness. The resolve (1) flips the signature open -> resolved with the witness stamp,
// (2) drops the fixer's exclusive lease (W5), and (3) — proven through the real
// applyKnownBadHold scope-hold (W4) seam — turns a previously BLOCKED_BY_KNOWN_BAD issue
// back into a dispatchable candidate on the next tick. Dropping only one of the hold/lease
// would leave the fleet half-stuck; this asserts BOTH.
func TestKnownBadResolveReleasesOnWitness(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	const now = int64(1_700_000_000)

	sig := knownbad.Signature("build", []string{"internal/foo/**"}, "")

	// record the shared known-bad, then elect a fixer (claim acquires the exclusive lease).
	var rb bytes.Buffer
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ledger", ledger,
	}, now); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}
	rb.Reset()
	if rc := runKnownBad(&rb, &rb, []string{
		"claim", "--by", "fixer", "--dir", dir, "--ledger", ledger, sig,
	}, now); rc != 0 {
		t.Fatalf("claim rc=%d out=%q", rc, rb.String())
	}

	// BEFORE the resolve: the live signature holds the intersecting issue #101 out of
	// dispatch (the W4 scope-hold reading the ledger). This is the "skipped" state.
	recordsBefore, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger before: %v", err)
	}
	heldBefore := applyKnownBadHold(twoLanePayload(), recordsBefore, now)
	if len(knownBadBlockedSkipped(heldBefore)) != 1 {
		t.Fatalf("precondition: issue #101 must be BLOCKED_BY_KNOWN_BAD before resolve, held=%+v", knownBadBlockedSkipped(heldBefore))
	}
	if _, ok := heldBefore.Lanes["foo"]; ok {
		t.Fatalf("precondition: lane foo must be held before resolve")
	}

	// The witness passes -> resolve flips open -> resolved and releases the lease.
	var capturedTree []string
	stubKnownBadWitness(t, true, &capturedTree)

	var ob bytes.Buffer
	rc := runKnownBad(&ob, &ob, []string{
		"resolve", "--by", "fixer", "--witness", "tests", "--dir", dir, "--ledger", ledger, "--json", sig,
	}, now)
	if rc != 0 {
		t.Fatalf("witnessed resolve rc=%d (want 0) out=%q", rc, ob.String())
	}
	var res resolveJSON
	if err := json.Unmarshal(ob.Bytes(), &res); err != nil {
		t.Fatalf("resolve --json invalid: %v\nout=%q", err, ob.String())
	}
	if !res.OK || res.Witness != "tests" || res.Record == nil {
		t.Fatalf("witnessed resolve envelope malformed: %+v", res)
	}
	if res.Record.Status != knownbad.StatusResolved || res.Record.ResolvedBy != "fixer" || res.Record.Witness != "tests" {
		t.Fatalf("resolved row not stamped correctly: %+v", res.Record)
	}
	if res.Record.ResolvedAtUnix != now {
		t.Errorf("resolve stamped at %d, want injected now=%d", res.Record.ResolvedAtUnix, now)
	}
	if len(capturedTree) != 1 || capturedTree[0] != "internal/foo" {
		t.Errorf("witness ran over %v, want the signature's tree [internal/foo]", capturedTree)
	}

	// (2) The fixer's exclusive lease was DROPPED (the W5 release). The verdict is OK and
	// the ref is gone from the store.
	if res.Lease == nil || !res.Lease.OK {
		t.Fatalf("resolve must drop the fixer lease, verdict=%+v", res.Lease)
	}
	store := leaseref.NewInDir(dir)
	if _, ok, err := store.Get(context.Background(), knownbad.LeaseID(sig)); err != nil || ok {
		t.Fatalf("fixer lease must be released after resolve: present=%v err=%v", ok, err)
	}

	// (1)+(3) AFTER the resolve: the signature is no longer live, so the W4 scope-hold
	// reading the SAME ledger no longer holds #101 — it routes as a dispatchable candidate.
	recordsAfter, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger after: %v", err)
	}
	if _, live := knownbad.FindLatestLive(recordsAfter, sig, now); live {
		t.Fatalf("resolved signature must not be live anymore")
	}
	freed := applyKnownBadHold(twoLanePayload(), recordsAfter, now)
	if got := len(knownBadBlockedSkipped(freed)); got != 0 {
		t.Fatalf("after resolve, NO issue may be BLOCKED_BY_KNOWN_BAD, got %d held", got)
	}
	foo, ok := freed.Lanes["foo"]
	if !ok || len(foo.Issues) != 1 || foo.Issues[0] != 101 {
		t.Fatalf("after resolve, previously-held issue #101 must route in lane foo, got %+v (ok=%v)", foo, ok)
	}
	routable := false
	for _, iss := range freed.Issues {
		if iss.Number == 101 {
			routable = true
		}
	}
	if !routable {
		t.Fatalf("after resolve, issue #101 must be offered as a routable candidate again")
	}
	// Sanity: the freed payload is the full un-held two-lane payload (both lanes routed).
	if _, ok := freed.Lanes["bar"]; !ok {
		t.Errorf("disjoint lane bar must remain routed throughout")
	}
}

// TestKnownBadResolvePreconditions pins the resolve verb's non-witness control paths:
// missing/malformed signature and an unknown witness kind are usage errors (exit 2); a
// resolve on a signature with no LIVE ledger row is a precondition error (exit 2). None of
// these run the witness (there is nothing to resolve), so the fleet is never touched.
func TestKnownBadResolvePreconditions(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	// A green stub proves these bail BEFORE the witness/ledger flip: even with a green
	// witness available, each case must still exit 2 on its precondition.
	stubKnownBadWitness(t, true, nil)

	cases := [][]string{
		{"resolve", "--dir", dir, "--ledger", ledger},                                   // no signature
		{"resolve", "--dir", dir, "--ledger", ledger, "::::"},                           // no ref-safe content is still a live-lookup miss
		{"resolve", "--dir", dir, "--ledger", ledger, "sha256:neverrecorded"},           // not in the ledger
		{"resolve", "--dir", dir, "--ledger", ledger, "--witness", "bogus", "sha256:x"}, // unknown witness kind
	}
	for _, argv := range cases {
		var out bytes.Buffer
		if rc := runKnownBad(&out, &out, argv, 1); rc != 2 {
			t.Errorf("resolve %v rc=%d, want 2 (out=%q)", argv, rc, out.String())
		}
	}
}
