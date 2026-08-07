package superloop

// steerpr_overlay_test.go — the witnesses for issue #5039: the operator-steerability
// overlay registered as a MEMBER of tend-scoreboards, so it is ENTERED by the
// worst-first walk rather than remembered by an operator.
//
// The registry's value is that no member can go dark unnoticed. Before this
// registration the overlay could stop ticking for a week and nothing surfaced it —
// exactly the failure mode tend-scoreboards exists to eliminate. These tests pin the
// four properties the issue's done-condition names, each as its own witness:
//
//	registered   — the overlay rides tend-scoreboards as one KindLoop member, NOT a
//	               new intent (a second intent walking the same debt would fragment
//	               the fold), and carries no scorecard ref of its own
//	runnable     — its Enter is the concrete verb that retires the debt it measures
//	               (`fak steer prs`), classified FrontRunnable so a headless drive can
//	               execute the worklist's action column exactly as printed
//	real number  — a stale/dark overlay folds a real debt number and out-ranks a clean
//	               scorecard in the worst-first worklist
//	UNMEASURED   — a broken/absent overlay read surfaces UNMEASURED and blocks
//	               Satisfied; it never reads as a clean zero

import (
	"strings"
	"testing"
)

// steerprOverlayLoopRef is the overlay maintenance loop's stable identity — the kind
// loopfleet.Fold stamps on the docs/nightrun/steerpr-overlay.jsonl rows (#5023), and
// therefore the exact ref the walk shell resolves this member's health by. A drift
// between the two silently turns the member UNMEASURED forever, so it is pinned here.
const steerprOverlayLoopRef = "steerpr-overlay"

// steerprOverlayEnter is the runnable front door the issue names: the read-only fold
// that recomputes the residual pile the overlay measures. It exits 0 on a successful
// fold (the --check flag, deliberately NOT used here, is what turns residual into a
// nonzero exit), so a headless drive can run it as its own witness.
const steerprOverlayEnter = "fak steer prs"

// steerprOverlayMember returns the registered overlay member from a walked intent,
// failing the test when it is absent — every witness below reads the member from the
// REGISTRY rather than restating it, so a registry edit cannot pass a stale literal.
func steerprOverlayMember(t *testing.T, s Super) Member {
	t.Helper()
	for _, m := range s.Members {
		if m.Kind == KindLoop && m.Ref == steerprOverlayLoopRef {
			return m
		}
	}
	t.Fatalf("super loop %q does not register the overlay maintenance loop %q as a KindLoop member", s.Name, steerprOverlayLoopRef)
	return Member{}
}

// TestSteerprOverlayIsATendScoreboardsMember pins the registration itself: the overlay
// is a MEMBER of the existing reporting intent, not a new super loop, and it is the one
// intent that claims it. A second claimant would double-enter the same surface.
func TestSteerprOverlayIsATendScoreboardsMember(t *testing.T) {
	if _, ok := Lookup(steerprOverlayLoopRef); ok {
		t.Fatalf("%q is registered as its own super loop — the overlay MUTATES a ledger, so it is a member, never an interior node (issue #5039 out-of-scope)", steerprOverlayLoopRef)
	}
	var claimants []string
	for _, s := range Registry() {
		for _, m := range s.Members {
			if m.Kind == KindLoop && m.Ref == steerprOverlayLoopRef {
				claimants = append(claimants, s.Name)
			}
		}
	}
	if len(claimants) != 1 || claimants[0] != "tend-scoreboards" {
		t.Fatalf("overlay loop claimants = %v, want exactly [tend-scoreboards] — a second claimant enters the same surface twice", claimants)
	}
}

// TestSteerprOverlayMemberIsRunnableAsPrinted pins the enter hint the issue requires:
// the concrete command that retires the member's debt, and a FrontRunnable
// classification so the drive rung can actually execute it behind the lease. A skill
// front door ("/x") or an empty Enter would leave the action column unrunnable.
func TestSteerprOverlayMemberIsRunnableAsPrinted(t *testing.T) {
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	m := steerprOverlayMember(t, sb)
	if m.Enter != steerprOverlayEnter {
		t.Errorf("overlay member Enter = %q, want %q — the worklist action must be the verb that retires its debt", m.Enter, steerprOverlayEnter)
	}
	if strings.TrimSpace(m.Why) == "" {
		t.Error("overlay member has no Why — the worklist must say what going dark costs")
	}
	fd := FrontDoorFor(DriveDecision{Member: m})
	if fd.Kind != FrontRunnable {
		t.Errorf("overlay front door = %q, want %q — a headless drive must be able to run the printed action", fd.Kind, FrontRunnable)
	}
	if fd.Command != steerprOverlayEnter {
		t.Errorf("overlay front-door command = %q, want %q", fd.Command, steerprOverlayEnter)
	}
}

// TestSteerprOverlayAddsNoScorecardRef pins the once-only fold this registration must
// not disturb: the overlay member carries LIVENESS, not a scorecard key, so it adds
// nothing to ScorecardRefs() and cannot double-count any card's debt at the root.
//
// The osp_residual CARD landed separately (#5022) as its own KindScorecard member, so
// the ref is now legitimately present in the registry — what stays forbidden is the
// LIVENESS member growing a scorecard ref of its own. The assertion is therefore
// pinned to the overlay member itself rather than to the absence of the card key.
func TestSteerprOverlayAddsNoScorecardRef(t *testing.T) {
	for _, ref := range ScorecardRefs() {
		if ref == steerprOverlayLoopRef {
			t.Errorf("scorecard ref %q leaked into the registry from the overlay registration — this member registers LIVENESS only", ref)
		}
	}
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	for _, m := range sb.Members {
		if m.Ref == steerprOverlayLoopRef && m.Kind != KindLoop {
			t.Errorf("overlay member %q has kind %q, want %q — its debt is liveness, never a scorecard", m.Ref, m.Kind, KindLoop)
		}
	}
	if rep := Graph(); !rep.OnceOnly {
		t.Errorf("registering the overlay broke the once-only fold: %v", rep.DoubleCounted)
	}
}

// TestSteerprOverlayStaleFoldsARealDebtNumber is the "real debt number" witness: a
// stale overlay loop carries debt and out-ranks a clean scorecard in the worst-first
// worklist, with a runnable revive action. This is what makes a week of silence
// visible instead of remembered.
func TestSteerprOverlayStaleFoldsARealDebtNumber(t *testing.T) {
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	overlay := steerprOverlayMember(t, sb)
	rep := Walk(sb, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "product"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "release"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "steer"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "milestone"}, Measured: true, Debt: 0},
		{Member: overlay, Measured: true, Debt: 1, Dark: true},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if rep.Satisfied {
		t.Error("a dark overlay must keep the intent UNSATISFIED — a stopped overlay is not a tended scoreboard")
	}
	if rep.Dark != 1 {
		t.Errorf("Dark = %d, want 1", rep.Dark)
	}
	if len(rep.Worklist) == 0 {
		t.Fatal("empty worklist: the dark overlay must be enterable")
	}
	top := rep.Worklist[0]
	if top.Member.Ref != steerprOverlayLoopRef {
		t.Fatalf("worst-first head = %q, want the dark overlay %q — a dark loop out-ranks clean scorecards", top.Member.Ref, steerprOverlayLoopRef)
	}
	if top.Debt != 1 {
		t.Errorf("overlay worklist debt = %d, want the folded 1 (a real number, not a placeholder)", top.Debt)
	}
	if !strings.Contains(top.Action, steerprOverlayEnter) {
		t.Errorf("overlay action = %q, want it to name the runnable %q", top.Action, steerprOverlayEnter)
	}
}

// TestSteerprOverlayUnreadSurfacesUnmeasured is the honesty witness the acceptance gate
// names: break the overlay read (no ledger on this host, or one loopfleet cannot fold)
// and the member surfaces UNMEASURED and blocks Satisfied — never a clean 0. An absent
// signal must not be reported as a healthy one.
func TestSteerprOverlayUnreadSurfacesUnmeasured(t *testing.T) {
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	overlay := steerprOverlayMember(t, sb)
	rep := Walk(sb, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "product"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "release"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "steer"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "milestone"}, Measured: true, Debt: 0},
		// The shell's KindLoop miss path: no foldable ledger for this ref on this host.
		{Member: overlay, Measured: false, Detail: "no ledger on this host (loop has not run here)"},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if rep.Unmeasured != 1 {
		t.Errorf("Unmeasured = %d, want 1 — a broken overlay read is a known gap", rep.Unmeasured)
	}
	if rep.Satisfied {
		t.Error("an UNMEASURED overlay must block Satisfied — an unread surface can never read clean")
	}
	if len(rep.Worklist) == 0 {
		t.Fatal("empty worklist: an unmeasured overlay must still be surfaced for entry")
	}
	top := rep.Worklist[0]
	if top.Member.Ref != steerprOverlayLoopRef {
		t.Fatalf("worst-first head = %q, want the unmeasured overlay %q — an unread member ranks with the darkest", top.Member.Ref, steerprOverlayLoopRef)
	}
	if top.Debt != 0 {
		t.Errorf("unmeasured overlay debt = %d, want 0 — an unread member must carry NO fabricated number; its urgency rides the UNMEASURED flag", top.Debt)
	}
	if !strings.Contains(top.Action, steerprOverlayEnter) {
		t.Errorf("unmeasured overlay action = %q, want it to name the runnable %q", top.Action, steerprOverlayEnter)
	}
}
