package recall

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// TestParity_MatchingDigestAgrees: two replicas of one resource sharing a content address
// corroborate — no disagreement, safe to reuse.
func TestParity_MatchingDigestAgrees(t *testing.T) {
	d := Digest([]byte("one true answer"))
	w := "parity:" + t.Name() + ":w"
	f := ParityCheck([]Page{
		{Step: 0, Digest: d, Witness: w, TrustEpoch: 1},
		{Step: 1, Digest: d, Witness: w, TrustEpoch: 1},
	})
	if f.Verdict != ParityAgree {
		t.Fatalf("matching-digest replicas => ParityAgree, got %s", f.Verdict)
	}
	if len(f.SealSteps) != 0 || f.RequireWitness {
		t.Fatalf("agreement must seal nothing and require nothing: %+v", f)
	}
}

// TestParity_OneRevokedWitnessResolvesAndSeals: replicas disagree on bytes, but exactly one
// is carried by a live witness — that one is authoritative and the revoked replica is sealed.
func TestParity_OneRevokedWitnessResolvesAndSeals(t *testing.T) {
	liveW := "parity:" + t.Name() + ":live"
	revW := "parity:" + t.Name() + ":revoked"
	vdso.Default.Revoke(revW)

	live := Page{Step: 0, Digest: Digest([]byte("authoritative bytes")), Witness: liveW, TrustEpoch: 2, Durability: "durable"}
	stale := Page{Step: 1, Digest: Digest([]byte("divergent stale bytes")), Witness: revW, TrustEpoch: 1, Durability: "durable"}

	f := ParityCheck([]Page{live, stale})
	if f.Verdict != ParityResolved {
		t.Fatalf("one live + one revoked witness on differing bytes => ParityResolved, got %s", f.Verdict)
	}
	if f.Authoritative != 0 {
		t.Fatalf("the live-witnessed replica (step 0) must be authoritative, got %d", f.Authoritative)
	}
	if len(f.SealSteps) != 1 || f.SealSteps[0] != 1 {
		t.Fatalf("the revoked divergent replica (step 1) must be sealable, got %v", f.SealSteps)
	}
	if !contains(f.Disagree, "digest") || !contains(f.Disagree, "witness") {
		t.Fatalf("disagreement should name digest+witness, got %v", f.Disagree)
	}
}

// TestParity_TwoLiveWitnessesRequiresWitness: replicas disagree on bytes with two live
// witnesses — no clear latest valid witness, so refuse reuse / require a witness.
func TestParity_TwoLiveWitnessesRequiresWitness(t *testing.T) {
	w1 := "parity:" + t.Name() + ":w1"
	w2 := "parity:" + t.Name() + ":w2"
	a := Page{Step: 0, Digest: Digest([]byte("version A")), Witness: w1, TrustEpoch: 5}
	b := Page{Step: 1, Digest: Digest([]byte("version B")), Witness: w2, TrustEpoch: 9}

	f := ParityCheck([]Page{a, b})
	if f.Verdict != ParityConflict {
		t.Fatalf("two live witnesses on differing bytes => ParityConflict, got %s", f.Verdict)
	}
	if !f.RequireWitness {
		t.Fatal("a conflict must require a witness rather than pick one")
	}
	if len(f.SealSteps) != 0 {
		t.Fatalf("a conflict must abstain (seal nothing), got %v", f.SealSteps)
	}
	if f.Authoritative != -1 {
		t.Fatalf("a conflict has no authoritative replica, got %d", f.Authoritative)
	}
	// A higher trust epoch must NOT silently win — that is exactly the "pick one" the check exists to refuse.
}

// TestParity_ScanCountsDisagreements proves ParityScan folds caller-grouped resources into
// the audit counts, and GroupByResource buckets by a caller-supplied key.
func TestParity_ScanCountsDisagreements(t *testing.T) {
	liveW := "parity:" + t.Name() + ":live"
	revW := "parity:" + t.Name() + ":revoked"
	w1 := "parity:" + t.Name() + ":w1"
	w2 := "parity:" + t.Name() + ":w2"
	vdso.Default.Revoke(revW)

	agreeD := Digest([]byte("agreed"))
	pages := []Page{
		// resource "kb": agreement (same digest)
		{Step: 0, Role: "kb", Digest: agreeD, Witness: liveW},
		{Step: 1, Role: "kb", Digest: agreeD, Witness: liveW},
		// resource "policy": resolved (live vs revoked, differing digests)
		{Step: 2, Role: "policy", Digest: Digest([]byte("p-live")), Witness: liveW},
		{Step: 3, Role: "policy", Digest: Digest([]byte("p-stale")), Witness: revW},
		// resource "wiki": conflict (two live witnesses, differing digests)
		{Step: 4, Role: "wiki", Digest: Digest([]byte("w-A")), Witness: w1},
		{Step: 5, Role: "wiki", Digest: Digest([]byte("w-B")), Witness: w2},
	}
	groups := GroupByResource(pages, func(p Page) string { return p.Role })
	if len(groups) != 3 {
		t.Fatalf("expected 3 resource groups, got %d", len(groups))
	}
	rep := ParityScan(groups)
	if rep.Agreements != 1 || rep.Resolved != 1 || rep.Conflicts != 1 {
		t.Fatalf("want agree=1 resolved=1 conflict=1, got %+v", rep)
	}
	if rep.Resources != 3 || len(rep.Findings) != 3 {
		t.Fatalf("report should cover 3 resources, got %+v", rep)
	}
}

// TestParity_GroupByResourceDropsKeylessPages: a page with no resource identity is not a
// parity candidate (parity is only meaningful where identity is known).
func TestParity_GroupByResourceDropsKeylessPages(t *testing.T) {
	pages := []Page{
		{Step: 0, Role: "kb", Digest: "d0"},
		{Step: 1, Role: "", Digest: "d1"}, // no resource identity
	}
	groups := GroupByResource(pages, func(p Page) string { return p.Role })
	if len(groups) != 1 || len(groups["kb"]) != 1 {
		t.Fatalf("keyless page must be dropped, got %+v", groups)
	}
}

// TestParity_CheckIsPure proves ParityCheck reads only — it never mutates the replicas it
// inspects (it is advisory; sealing is the caller's fail-closed choice).
func TestParity_CheckIsPure(t *testing.T) {
	revW := "parity:" + t.Name() + ":revoked"
	vdso.Default.Revoke(revW)
	in := []Page{
		{Step: 0, Digest: Digest([]byte("A")), Witness: "parity:" + t.Name() + ":live"},
		{Step: 1, Digest: Digest([]byte("B")), Witness: revW},
	}
	before := append([]Page(nil), in...)
	_ = ParityCheck(in)
	for i := range in {
		if in[i] != before[i] {
			t.Fatalf("ParityCheck mutated replica %d: %+v != %+v", i, in[i], before[i])
		}
	}
}

// TestParity_AuditOverSessionPageTable proves the recall-selection-time hook: Session.
// ParityAudit groups the loaded page table by a caller-supplied key and reports the
// disagreement without mutating the page table.
func TestParity_AuditOverSessionPageTable(t *testing.T) {
	liveW := "parity:" + t.Name() + ":live"
	revW := "parity:" + t.Name() + ":revoked"
	vdso.Default.Revoke(revW)
	pages := []Page{
		{Step: 0, Role: "policy", Digest: Digest([]byte("p-live")), Witness: liveW},
		{Step: 1, Role: "policy", Digest: Digest([]byte("p-stale")), Witness: revW},
	}
	s := &Session{Manifest: Manifest{Pages: pages}}
	rep := s.ParityAudit(func(p Page) string { return p.Role })
	if rep.Resolved != 1 || rep.Conflicts != 0 || rep.Agreements != 0 {
		t.Fatalf("audit should resolve the single policy disagreement: %+v", rep)
	}
	if s.Manifest.Pages[0].Quarantined || s.Manifest.Pages[1].Quarantined {
		t.Fatal("ParityAudit must not mutate the page table (advisory only)")
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
