package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// gitInitKnownBad spins up a real, empty git repo for the exclusive-lease store the
// claim verb writes refs into (skips when git is unavailable, like the leaseref
// end-to-end tests).
func gitInitKnownBad(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

// claimJSON is the subset of the claim verb's --json envelope the test asserts on.
type claimJSON struct {
	OK        bool   `json:"ok"`
	Signature string `json:"signature"`
	LeaseID   string `json:"lease_id"`
	Fixer     string `json:"fixer"`
	Reason    string `json:"reason"`
}

// TestKnownBadClaimElectsExactlyOneFixer is the W5 (#2717) done-condition witness:
// two agents race `fak knownbad claim <sig>` over the SAME live signature; exactly
// one wins (exit 0, stamped onto the ledger), and the other is REFUSED (exit 3,
// KNOWN_BAD_ALREADY_CLAIMED) carrying the WINNER's identity — never a bare "refused".
// The election is enforced by the exclusive dos lease, so the ledger ends with a
// single claimed row owned by the winner. A re-claim by the winner renews (exit 0);
// a claim on an unrecorded signature is a precondition error (exit 2).
func TestKnownBadClaimElectsExactlyOneFixer(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")
	const now = int64(1_700_000_000)

	// Record the shared known-bad; the recorded signature is what the agents claim.
	sig := knownbad.Signature("build", []string{"internal/foo/**"}, "")
	var rb bytes.Buffer
	if rc := runKnownBad(&rb, &rb, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build",
		"--note", "shared foo break", "--by", "discoverer", "--ledger", ledger,
	}, now); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, rb.String())
	}

	// Two agents race the same signature concurrently over the same lease store.
	type result struct {
		rc  int
		out string
	}
	holders := []string{"agent-A", "agent-B"}
	results := make([]result, len(holders))
	var wg sync.WaitGroup
	for i, who := range holders {
		wg.Add(1)
		go func(i int, who string) {
			defer wg.Done()
			var out bytes.Buffer
			rc := runKnownBad(&out, &out, []string{
				"claim", "--by", who, "--dir", dir, "--ledger", ledger, "--json", sig,
			}, now)
			results[i] = result{rc: rc, out: out.String()}
		}(i, who)
	}
	wg.Wait()

	// Exactly one winner (rc 0) and one refusal (rc 3).
	var winners, losers []claimJSON
	for i, r := range results {
		var cj claimJSON
		if err := json.Unmarshal([]byte(r.out), &cj); err != nil {
			t.Fatalf("claim[%s] --json invalid: %v\nout=%q", holders[i], err, r.out)
		}
		switch r.rc {
		case 0:
			winners = append(winners, cj)
		case leaserefRefused:
			losers = append(losers, cj)
		default:
			t.Fatalf("claim[%s] rc=%d (want 0 or %d) out=%q", holders[i], r.rc, leaserefRefused, r.out)
		}
	}
	if len(winners) != 1 || len(losers) != 1 {
		t.Fatalf("election was not exactly-one: %d winners, %d losers", len(winners), len(losers))
	}
	win, lose := winners[0], losers[0]

	// The winner owns the fix; the loser is refused with the STRUCTURED reason and is
	// handed the winner's identity (a pointer to the fixer, not a bare "refused").
	if !win.OK || win.Fixer == "" || win.Signature != sig {
		t.Fatalf("winner envelope malformed: %+v", win)
	}
	if lose.OK || lose.Reason != reasonKnownBadAlreadyClaimed {
		t.Fatalf("loser not refused with %s: %+v", reasonKnownBadAlreadyClaimed, lose)
	}
	if lose.Fixer != win.Fixer {
		t.Fatalf("loser was not told the winner's identity: loser.Fixer=%q, winner=%q", lose.Fixer, win.Fixer)
	}
	if win.LeaseID != knownbad.LeaseID(sig) {
		t.Errorf("winner lease id = %q, want %q", win.LeaseID, knownbad.LeaseID(sig))
	}

	// The exclusive lease ref is held by the winner (the mechanism, not just the report).
	store := leaseref.NewInDir(dir)
	rec, ok, err := store.Get(context.Background(), knownbad.LeaseID(sig))
	if err != nil || !ok {
		t.Fatalf("lease %s not present after claim: ok=%v err=%v", knownbad.LeaseID(sig), ok, err)
	}
	if rec.Holder != win.Fixer {
		t.Errorf("lease holder = %q, want the winner %q", rec.Holder, win.Fixer)
	}

	// The ledger ends with exactly one CLAIMED row for the signature, owned by the winner.
	records, err := readKnownBadLedger(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	claimed := 0
	for _, r := range records {
		if r.Signature == sig && r.Claimed() {
			claimed++
			if r.ClaimedBy != win.Fixer {
				t.Errorf("claimed row owned by %q, want winner %q", r.ClaimedBy, win.Fixer)
			}
			if r.ClaimedAtUnix != now {
				t.Errorf("claim stamped at %d, want injected now=%d", r.ClaimedAtUnix, now)
			}
		}
	}
	if claimed != 1 {
		t.Fatalf("ledger has %d claimed rows for the signature, want exactly 1", claimed)
	}

	// A re-claim by the winner RENEWS its own lease (idempotent, exit 0) — not a refusal.
	var reb bytes.Buffer
	if rc := runKnownBad(&reb, &reb, []string{
		"claim", "--by", win.Fixer, "--dir", dir, "--ledger", ledger, sig,
	}, now); rc != 0 {
		t.Fatalf("winner re-claim rc=%d (want 0, renew) out=%q", rc, reb.String())
	}
}

// TestKnownBadClaimPreconditions pins the claim verb's non-race control paths: a
// missing signature and a malformed signature are usage errors (exit 2), and a
// claim on a signature that is not a LIVE ledger row is a precondition error (exit
// 2) — you cannot elect a fixer for a failure that was never recorded.
func TestKnownBadClaimPreconditions(t *testing.T) {
	dir := gitInitKnownBad(t)
	ledger := filepath.Join(dir, "known-bad.jsonl")

	cases := [][]string{
		{"claim", "--dir", dir, "--ledger", ledger},                         // no signature
		{"claim", "--dir", dir, "--ledger", ledger, "::::"},                 // no ref-safe content
		{"claim", "--dir", dir, "--ledger", ledger, "sha256:neverrecorded"}, // not in the ledger
	}
	for _, argv := range cases {
		var out bytes.Buffer
		if rc := runKnownBad(&out, &out, argv, 1); rc != 2 {
			t.Errorf("claim %v rc=%d, want 2 (out=%q)", argv, rc, out.String())
		}
	}
}
