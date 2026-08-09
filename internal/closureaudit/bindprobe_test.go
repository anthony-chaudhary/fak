package closureaudit

import "testing"

// The fixtures below are REAL commits and REAL `dos commit-audit` rows read off
// this repo's trunk on 2026-08-07, not invented shapes — the probe is only worth
// anything if it reproduces the states the live ledger actually contains.

func okAudit(kind string) Audit {
	return Audit{Verdict: "OK", Witness: "diff-witnessed", ClaimKind: kind}
}

// #5027 burned 22 dispatch runs. Its three commits are the canonical near-miss
// spread: one body-only mention, and two subject cites that DO bind. The point of
// this test is the refutation the census produced — the `(fak cmd)` stamp on
// e68be47327 is irrelevant to closure, so the probe must call it BOUND.
func TestProbeReproducesIssue5027(t *testing.T) {
	iss := Issue{Number: 5027, Title: "feat(steerpr): report a forming unit as N of M expected commits landed", State: "OPEN"}
	cands := []BindCandidate{
		{Commit: Commit{SHA: "3c4424894e", Subject: "test(steerpr): bind the partial denominator to issuefanout's minted marker key (#5027) (fak steerpr)"}, Depth: 6},
		{Commit: Commit{SHA: "e68be47327", Subject: "test(cmd): add the steer-prs partial-state operator gate (#5027) (fak cmd)"}, Depth: 329},
		{Commit: Commit{SHA: "c18a6db1e3", Subject: "feat(steer): wire the partial-membership state into the steer PRs overlay (fak cmd)", Body: "part of the #5027 overlay"}, Depth: 900},
	}
	audits := map[string]Audit{"3c4424894e": okAudit("test"), "e68be47327": okAudit("test"), "c18a6db1e3": okAudit("code_effect")}

	p := Probe(iss, cands, audits, DefaultScanLimit)
	if p.Verdict != BindBound {
		t.Fatalf("verdict = %q, want %q", p.Verdict, BindBound)
	}
	if p.BindingSHA != "3c4424894e" {
		t.Fatalf("binding sha = %q, want the shallowest binding commit 3c4424894e", p.BindingSHA)
	}
	if !p.Shipped() {
		t.Fatal("Shipped() = false on a BOUND probe")
	}
	// The body-only commit must be named as a mention_only near-miss, and the
	// leaf-mismatched-but-subject-citing one must NOT be a near-miss at all.
	var mention, wrongLeaf bool
	for _, m := range p.NearMisses {
		if m.SHA == "c18a6db1e3" && m.Kind == MissMentionOnly {
			mention = true
		}
		if m.SHA == "e68be47327" {
			wrongLeaf = true
		}
	}
	if !mention {
		t.Errorf("body-only commit c18a6db1e3 not reported as %s; near misses = %+v", MissMentionOnly, p.NearMisses)
	}
	if wrongLeaf {
		t.Error("e68be47327 reported as a near-miss: the (fak <leaf>) trailer is matched against touched PATHS, " +
			"never against an issue, so a leaf mismatch can never break a closure binding")
	}
}

// The state this file exists to name: the work shipped, the commit is witnessed,
// and the consumer's scan window has moved past it — so a dispatcher that only
// looks `scan_limit` commits back will re-select this issue forever.
func TestProbeNamesOutOfScanWindowBinding(t *testing.T) {
	iss := Issue{Number: 2728, Title: "docs(mac): retire the no-SOTA-speed fence", State: "OPEN"}
	cands := []BindCandidate{{
		Commit: Commit{SHA: "edcace6bde", Subject: "docs(serving): fix the dual-track doc's dead rg anchors (#2728) (fak docs)"},
		Depth:  1269,
	}}
	audits := map[string]Audit{"edcace6bde": okAudit("doc")}

	p := Probe(iss, cands, audits, DefaultScanLimit)
	if p.Verdict != BindOutOfWindow {
		t.Fatalf("verdict = %q, want %q", p.Verdict, BindOutOfWindow)
	}
	if !p.Shipped() {
		t.Fatal("Shipped() = false on an out-of-window binding: the work DID land, so a dispatcher must not re-select it")
	}
	if p.BindingDepth != 1269 || p.BindingSHA != "edcace6bde" {
		t.Fatalf("binding = %s@%d, want edcace6bde@1269", p.BindingSHA, p.BindingDepth)
	}
	var named bool
	for _, m := range p.NearMisses {
		if m.Kind == MissOutOfScanWindow {
			named = true
		}
	}
	if !named {
		t.Errorf("no %s near-miss emitted; near misses = %+v", MissOutOfScanWindow, p.NearMisses)
	}
	// The same facts with the window rung disabled are a plain BOUND.
	if q := Probe(iss, cands, audits, 0); q.Verdict != BindBound {
		t.Errorf("scanLimit=0 verdict = %q, want %q — the window rung is the CONSUMER's limit, not the issue's", q.Verdict, BindBound)
	}
}

// A doc-kind claim binds a docs-shaped issue and only a docs-shaped issue (#2998).
func TestProbeClaimKindGate(t *testing.T) {
	cands := []BindCandidate{{Commit: Commit{SHA: "aaaaaaaaaa", Subject: "docs(x): write it up (#4242)"}, Depth: 3}}
	audits := map[string]Audit{"aaaaaaaaaa": okAudit("doc")}

	feature := Probe(Issue{Number: 4242, Title: "feat(x): build the thing", State: "OPEN"}, cands, audits, DefaultScanLimit)
	if feature.Verdict != BindUnbound {
		t.Fatalf("feature issue verdict = %q, want %q", feature.Verdict, BindUnbound)
	}
	if len(feature.NearMisses) != 1 || feature.NearMisses[0].Kind != MissNonBindingClaim {
		t.Fatalf("near misses = %+v, want one %s", feature.NearMisses, MissNonBindingClaim)
	}
	docs := Probe(Issue{Number: 4242, Title: "docs(x): write the thing up", State: "OPEN"}, cands, audits, DefaultScanLimit)
	if docs.Verdict != BindBound {
		t.Fatalf("docs-rung issue verdict = %q, want %q", docs.Verdict, BindBound)
	}
}

// An unknown claim_kind must fail OPEN — the gate demotes only a KNOWN doc/triage
// claim, so a legacy audit row can never be slandered into UNBOUND.
func TestClaimKindBindsFailsOpenOnUnknownKind(t *testing.T) {
	if !ClaimKindBinds(Audit{Verdict: "OK", Witness: "diff-witnessed"}, "feat(x): a feature") {
		t.Fatal("empty claim_kind must fail OPEN")
	}
	if ClaimKindBinds(Audit{ClaimKind: "triage"}, "feat(x): a feature") {
		t.Fatal("a known triage claim must not bind a feature issue")
	}
	if !ClaimKindBinds(Audit{ClaimKind: "triage"}, "docs(x): a doc") {
		t.Fatal("a triage claim must bind a docs-shaped issue")
	}
}

// A resolving cite whose diff does not witness its own claim is a near-miss, not a
// binding — and the probe must say which rung failed, with the observed verdict.
func TestProbeNamesUnwitnessedCite(t *testing.T) {
	iss := Issue{Number: 77, Title: "feat(x): do it", State: "OPEN"}
	cands := []BindCandidate{{Commit: Commit{SHA: "bbbbbbbbbb", Subject: "feat(x): do it (#77) (fak x)"}, Depth: 1}}
	audits := map[string]Audit{"bbbbbbbbbb": {Verdict: "CLAIM_UNWITNESSED", Witness: "subject-only", ClaimKind: "code_effect"}}

	p := Probe(iss, cands, audits, DefaultScanLimit)
	if p.Verdict != BindUnbound {
		t.Fatalf("verdict = %q, want %q", p.Verdict, BindUnbound)
	}
	if len(p.NearMisses) != 1 || p.NearMisses[0].Kind != MissUnwitnessed {
		t.Fatalf("near misses = %+v, want one %s", p.NearMisses, MissUnwitnessed)
	}
	if p.NearMisses[0].Detail == "" || p.Next == "" {
		t.Fatal("a near-miss must carry both a detail and a checkable next step")
	}
}

// An issue nothing references is genuinely undispatched work — the probe must say
// so plainly rather than implying a broken binding.
func TestProbeNoReferenceIsNotANearMiss(t *testing.T) {
	p := Probe(Issue{Number: 999, Title: "feat(y): unstarted", State: "OPEN"},
		[]BindCandidate{{Commit: Commit{SHA: "cccccccccc", Subject: "feat(z): something else (#1) (fak z)"}, Depth: 2}},
		map[string]Audit{"cccccccccc": okAudit("code_effect")}, DefaultScanLimit)
	if p.Verdict != BindUnbound {
		t.Fatalf("verdict = %q, want %q", p.Verdict, BindUnbound)
	}
	if len(p.NearMisses) != 0 {
		t.Fatalf("near misses = %+v, want none", p.NearMisses)
	}
	if p.Shipped() {
		t.Fatal("Shipped() = true with no referencing commit")
	}
}

// The probe is a pure fold: the same facts in any order must give the same verdict,
// binding sha, and near-miss ordering.
func TestProbeIsOrderIndependent(t *testing.T) {
	iss := Issue{Number: 5027, Title: "feat(steerpr): report a forming unit", State: "OPEN"}
	a := BindCandidate{Commit: Commit{SHA: "3c4424894e", Subject: "test(steerpr): bind it (#5027) (fak steerpr)"}, Depth: 6}
	b := BindCandidate{Commit: Commit{SHA: "e68be47327", Subject: "test(cmd): gate it (#5027) (fak cmd)"}, Depth: 329}
	audits := map[string]Audit{"3c4424894e": okAudit("test"), "e68be47327": okAudit("test")}

	fwd := Probe(iss, []BindCandidate{a, b}, audits, DefaultScanLimit)
	rev := Probe(iss, []BindCandidate{b, a}, audits, DefaultScanLimit)
	if fwd.Verdict != rev.Verdict || fwd.BindingSHA != rev.BindingSHA || fwd.BindingDepth != rev.BindingDepth {
		t.Fatalf("order-dependent fold: %+v vs %+v", fwd, rev)
	}
}
