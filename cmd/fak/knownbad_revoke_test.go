package main

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// revokeJSON is the subset of the revoke verb's --json envelope the tests assert on.
type revokeJSON struct {
	OK        bool                   `json:"ok"`
	Signature string                 `json:"signature"`
	Reason    string                 `json:"reason"`
	Record    *knownbad.Record       `json:"record"`
	Lease     *leaseref.FenceVerdict `json:"lease"`
}

// notLiveJSON is the shape a claim/resolve/revoke emits when refusing an aged-out or revoked
// signature (the KNOWN_BAD_EXPIRED_OR_REVOKED gate).
type notLiveJSON struct {
	OK     bool   `json:"ok"`
	Verb   string `json:"verb"`
	Reason string `json:"reason"`
	State  string `json:"state"`
}

// TestKnownBadRevokeReleasesWithoutWitness is the W8 (#2720) done-condition witness: the
// UNWITNESSED release valve. A recorded+claimed signature is revoked with NO witness — the
// escape hatch resolve deliberately withholds. The revoke (1) flips open -> revoked with the
// operator's prose reason, (2) drops the fixer's exclusive lease (W5), and (3) — through the
// real applyKnownBadHold scope-hold (W4) seam — turns a previously BLOCKED_BY_KNOWN_BAD issue
// back into a dispatchable candidate. No witness stub is installed at all: a revoke must not
// run one.
func TestKnownBadRevokeReleasesWithoutWitness(t *testing.T) {
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

	// BEFORE the revoke: the live signature holds the intersecting issue #101 out of dispatch.
	recordsBefore, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger before: %v", err)
	}
	heldBefore := applyKnownBadHold(twoLanePayload(), recordsBefore, now)
	if len(knownBadBlockedSkipped(heldBefore)) != 1 {
		t.Fatalf("precondition: issue #101 must be BLOCKED_BY_KNOWN_BAD before revoke, held=%+v", knownBadBlockedSkipped(heldBefore))
	}

	// Revoke with a reason, NO witness stub installed (a revoke must not run one).
	var ob bytes.Buffer
	rc := runKnownBad(&ob, &ob, []string{
		"revoke", "--by", "operator", "--reason", "flaky, not a shared bug",
		"--dir", dir, "--ledger", ledger, "--json", sig,
	}, now)
	if rc != 0 {
		t.Fatalf("revoke rc=%d (want 0) out=%q", rc, ob.String())
	}
	var res revokeJSON
	if err := json.Unmarshal(ob.Bytes(), &res); err != nil {
		t.Fatalf("revoke --json invalid: %v\nout=%q", err, ob.String())
	}
	if !res.OK || res.Record == nil {
		t.Fatalf("revoke envelope malformed: %+v", res)
	}
	if res.Record.Status != knownbad.StatusRevoked || res.Record.RevokedBy != "operator" {
		t.Fatalf("revoked row not stamped correctly: %+v", res.Record)
	}
	if res.Record.RevokeReason != "flaky, not a shared bug" || res.Record.RevokedAtUnix != now {
		t.Errorf("revoke reason/instant wrong: %+v", res.Record)
	}
	if res.Record.Witness != "" {
		t.Errorf("a revoke must carry NO witness: %q", res.Record.Witness)
	}

	// (2) The fixer's exclusive lease was DROPPED (same release arm as resolve).
	if res.Lease == nil || !res.Lease.OK {
		t.Fatalf("revoke must drop the fixer lease, verdict=%+v", res.Lease)
	}
	store := leaseref.NewInDir(dir)
	if _, ok, err := store.Get(context.Background(), knownbad.LeaseID(sig)); err != nil || ok {
		t.Fatalf("fixer lease must be released after revoke: present=%v err=%v", ok, err)
	}

	// (1)+(3) AFTER the revoke: the signature is no longer live, so #101 routes again.
	recordsAfter, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger after: %v", err)
	}
	if _, live := knownbad.FindLatestLive(recordsAfter, sig, now); live {
		t.Fatalf("revoked signature must not be live anymore")
	}
	freed := applyKnownBadHold(twoLanePayload(), recordsAfter, now)
	if got := len(knownBadBlockedSkipped(freed)); got != 0 {
		t.Fatalf("after revoke, NO issue may be BLOCKED_BY_KNOWN_BAD, got %d held", got)
	}
	if foo, ok := freed.Lanes["foo"]; !ok || len(foo.Issues) != 1 || foo.Issues[0] != 101 {
		t.Fatalf("after revoke, previously-held issue #101 must route in lane foo, got %+v (ok=%v)", foo, ok)
	}
}

// TestKnownBadRevokePreconditions pins the revoke verb's control paths: a missing signature, a
// missing --reason (a revoke is a judgement, so the audit reason is required), and too many
// positionals are all usage errors (exit 2). None append anything.
func TestKnownBadRevokePreconditions(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	const now = int64(1_700_000_000)

	// Record a live signature so the ONLY failure under test is the flag precondition.
	var rb bytes.Buffer
	sig := knownbad.Signature("build", []string{"internal/foo/**"}, "")
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ledger", ledger,
	}, now); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}

	cases := [][]string{
		{"revoke", "--dir", dir, "--ledger", ledger, "--reason", "x"},           // no signature
		{"revoke", "--dir", dir, "--ledger", ledger, sig},                       // no --reason
		{"revoke", "--dir", dir, "--ledger", ledger, "--reason", "x", sig, sig}, // two positionals
	}
	for _, argv := range cases {
		var out bytes.Buffer
		if rc := runKnownBad(&out, &out, argv, now); rc != 2 {
			t.Errorf("revoke %v rc=%d, want 2 (out=%q)", argv, rc, out.String())
		}
	}
	// The signature is still live — none of the refused revokes appended a row.
	records, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if _, live := knownbad.FindLatestLive(records, sig, now); !live {
		t.Fatalf("a refused revoke must leave the signature live; it is not")
	}
}

// TestKnownBadActOnExpiredOrRevokedRefused is the W8 structured-refuse witness: claim, resolve,
// and revoke against a signature that WAS recorded but is no longer live are each refused with
// the closed-vocabulary KNOWN_BAD_EXPIRED_OR_REVOKED reason (exit leaserefRefused) — distinct
// from a NEVER-recorded signature, which stays a bare usage error (exit 2). Covers both
// retraction paths: a lapsed TTL (expired) and an operator revoke.
func TestKnownBadActOnExpiredOrRevokedRefused(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	const recAt = int64(1_700_000_000)
	sig := knownbad.Signature("build", []string{"internal/foo/**"}, "")

	// A green witness stub is installed so a resolve refusal here is proven to come from the
	// not-live gate, NOT from a missing witness.
	stubKnownBadWitness(t, true, nil)

	// Record with a short explicit TTL, then advance the clock past it -> EXPIRED.
	var rb bytes.Buffer
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ttl", "100", "--ledger", ledger,
	}, recAt); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}
	expired := recAt + 1000

	// Each verb, against the EXPIRED signature, is refused with the structured reason.
	for _, verb := range []string{"claim", "resolve", "revoke"} {
		argv := []string{verb, "--by", "x", "--dir", dir, "--ledger", ledger, "--json"}
		if verb == "revoke" {
			argv = append(argv, "--reason", "stale")
		}
		argv = append(argv, sig)
		var out bytes.Buffer
		rc := runKnownBad(&out, &out, argv, expired)
		if rc != leaserefRefused {
			t.Fatalf("%s on expired sig rc=%d, want %d (refused) out=%q", verb, rc, leaserefRefused, out.String())
		}
		var res notLiveJSON
		if err := json.Unmarshal(out.Bytes(), &res); err != nil {
			t.Fatalf("%s --json invalid: %v out=%q", verb, err, out.String())
		}
		if res.OK || res.Reason != reasonKnownBadExpiredOrRevoked || res.State != "expired" {
			t.Fatalf("%s on expired sig not refused with %s/expired: %+v", verb, reasonKnownBadExpiredOrRevoked, res)
		}
	}

	// A NEVER-recorded signature stays a plain usage error (exit 2), not a structured refuse:
	// there is nothing to point the caller at.
	var out bytes.Buffer
	if rc := runKnownBad(&out, &out, []string{
		"claim", "--by", "x", "--dir", dir, "--ledger", ledger, "sha256:neverrecorded",
	}, expired); rc != 2 {
		t.Errorf("claim on never-recorded sig rc=%d, want 2 (out=%q)", rc, out.String())
	}

	// Now REVOKE a fresh signature, then confirm acting on the revoked row also refuses with
	// state "revoked" (the operator-retraction path, distinct from expiry).
	rb.Reset()
	sig2 := knownbad.Signature("test", []string{"internal/bar/**"}, "")
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/bar/**", "--reason", "test", "--ledger", ledger,
	}, recAt); rc != 0 {
		t.Fatalf("record sig2 rc=%d out=%q", rc, rb.String())
	}
	rb.Reset()
	if rc := runKnownBad(&rb, &rb, []string{
		"revoke", "--reason", "wrong tree", "--dir", dir, "--ledger", ledger, sig2,
	}, recAt); rc != 0 {
		t.Fatalf("revoke sig2 rc=%d out=%q", rc, rb.String())
	}
	var rout bytes.Buffer
	rc := runKnownBad(&rout, &rout, []string{
		"claim", "--by", "x", "--dir", dir, "--ledger", ledger, "--json", sig2,
	}, recAt)
	if rc != leaserefRefused {
		t.Fatalf("claim on revoked sig rc=%d, want %d out=%q", rc, leaserefRefused, rout.String())
	}
	var res notLiveJSON
	if err := json.Unmarshal(rout.Bytes(), &res); err != nil {
		t.Fatalf("claim-on-revoked --json invalid: %v out=%q", err, rout.String())
	}
	if res.State != "revoked" || res.Reason != reasonKnownBadExpiredOrRevoked {
		t.Fatalf("claim on revoked sig not refused with revoked state: %+v", res)
	}
}

// TestKnownBadRecordDefaultTTLIsBounded is the shell arm of the self-healing invariant: a
// `record` with NO --ttl stamps the bounded default (not 0 = forever), so the signature
// EXPIRES on its own. An explicit --ttl 0 opts back into a durable no-expiry signature.
func TestKnownBadRecordDefaultTTLIsBounded(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const recAt = int64(1_700_000_000)

	// record WITHOUT --ttl -> bounded default.
	var rb bytes.Buffer
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ledger", ledger, "--json",
	}, recAt); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}
	var rec knownbad.Record
	if err := json.Unmarshal(rb.Bytes(), &rec); err != nil {
		t.Fatalf("record --json invalid: %v out=%q", err, rb.String())
	}
	if rec.TTLSeconds != knownbad.DefaultRecordTTLSeconds {
		t.Fatalf("default record ttl = %d, want bounded default %d", rec.TTLSeconds, knownbad.DefaultRecordTTLSeconds)
	}

	// It matches now but NOT past the default window — it self-heals.
	if rc := runKnownBad(&rb, &rb, []string{"match", "--tree", "internal/foo/x.go", "--ledger", ledger}, recAt); rc != 3 {
		t.Errorf("default-ttl signature must match within its window, rc=%d", rc)
	}
	past := recAt + knownbad.DefaultRecordTTLSeconds + 1
	if rc := runKnownBad(&rb, &rb, []string{"match", "--tree", "internal/foo/x.go", "--ledger", ledger}, past); rc != 0 {
		t.Errorf("default-ttl signature must expire past its window, rc=%d", rc)
	}

	// record --ttl 0 -> durable, no expiry even far in the future.
	ledger2 := filepath.Join(t.TempDir(), "known-bad.jsonl")
	rb.Reset()
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build", "--ttl", "0", "--ledger", ledger2, "--json",
	}, recAt); rc != 0 {
		t.Fatalf("record --ttl 0 rc=%d out=%q", rc, rb.String())
	}
	var durable knownbad.Record
	if err := json.Unmarshal(rb.Bytes(), &durable); err != nil {
		t.Fatalf("record --json invalid: %v out=%q", err, rb.String())
	}
	if durable.TTLSeconds != 0 {
		t.Fatalf("--ttl 0 must stamp 0 (no expiry), got %d", durable.TTLSeconds)
	}
	if rc := runKnownBad(&rb, &rb, []string{"match", "--tree", "internal/foo/x.go", "--ledger", ledger2}, past); rc != 3 {
		t.Errorf("a --ttl 0 signature must never expire, rc=%d at far-future clock", rc)
	}
}
