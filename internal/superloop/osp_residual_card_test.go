package superloop

// osp_residual_card_test.go — the #5022 registration witnesses.
//
// The card makes "how many forming units owe an operator attention" a POSTED
// number on the tend-scoreboards steering feed, so a rising residual pile is
// visible without anyone remembering to run `fak steer prs`. What has to hold:
//
//   - it is a REAL control-pane card key (not a ref pointing at nothing),
//   - it is walked by EXACTLY ONE intent (no debt double-count at the root),
//   - a measured count folds as real debt and can out-rank its neighbours, and
//   - an unread card surfaces UNMEASURED and blocks Satisfied — never a clean 0.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

const ospResidualRef = "osp_residual"

// ospResidualMember returns the registered card member of tend-scoreboards.
func ospResidualMember(t *testing.T) Member {
	t.Helper()
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	for _, m := range sb.Members {
		if m.Ref == ospResidualRef {
			return m
		}
	}
	t.Fatalf("tend-scoreboards does not carry the %q card — the residual pile cannot trend, red a walk, or be posted", ospResidualRef)
	return Member{}
}

// TestOSPResidualIsARealCardOnTheSteeringFeed pins the registration itself: the card
// rides the reporting family beside steerability/milestone, as a debt-bearing
// scorecard member whose ref resolves in the control pane the walk actually reads.
func TestOSPResidualIsARealCardOnTheSteeringFeed(t *testing.T) {
	m := ospResidualMember(t)
	if m.Kind != KindScorecard {
		t.Errorf("%q kind = %q, want %q — it carries a measurable NUMBER, not liveness", ospResidualRef, m.Kind, KindScorecard)
	}
	if strings.TrimSpace(m.Enter) == "" {
		t.Errorf("%q has no Enter hint — its worklist action must be runnable as printed", ospResidualRef)
	}
	if !strings.Contains(m.Enter, "steer prs") {
		t.Errorf("%q Enter = %q, want the steer-prs verb that retires the debt it measures", ospResidualRef, m.Enter)
	}

	// The no-drift half: the ref must be a key the control pane really folds, else
	// the walk would send an operator at a scorecard that does not exist and the
	// pinned baseline could never carry a number for it.
	var card scorecardpane.Card
	for _, c := range scorecardpane.Cards {
		if c.Key == ospResidualRef {
			card = c
		}
	}
	if card.Key == "" {
		t.Fatalf("%q is not a control-pane card key (scorecardpane.Cards) — `scorecard_control_pane.py --pin` could never baseline it", ospResidualRef)
	}
	if card.Debt != "residual_count" {
		t.Errorf("card debt key = %q, want residual_count — debt IS the count of RESIDUAL units", card.Debt)
	}
}

// TestOSPResidualIsWalkedByExactlyOneIntent is the once-only witness the done
// condition names: no scorecard key may be walked by two intents, or the root fold
// counts its debt twice and distorts the worst-first ranking.
func TestOSPResidualIsWalkedByExactlyOneIntent(t *testing.T) {
	var holders []string
	for _, s := range Registry() {
		for _, m := range s.Members {
			if m.Kind == KindScorecard && m.Ref == ospResidualRef {
				holders = append(holders, s.Name)
			}
		}
	}
	if len(holders) != 1 {
		t.Fatalf("%q is walked by %d intent(s) %v, want exactly 1", ospResidualRef, len(holders), holders)
	}
	if holders[0] != "tend-scoreboards" {
		t.Errorf("%q is walked by %q, want tend-scoreboards — the steering feed is its home", ospResidualRef, holders[0])
	}
	if rep := Graph(); !rep.OnceOnly {
		t.Errorf("registering the card broke the once-only fold: %v", rep.DoubleCounted)
	}
}

// TestOSPResidualMeasuredFoldsARealDebtNumber is the "posted number" witness: a
// residual pile carries real debt, out-ranks the clean cards beside it, and its
// worklist action is the runnable verb.
func TestOSPResidualMeasuredFoldsARealDebtNumber(t *testing.T) {
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	card := ospResidualMember(t)
	rep := Walk(sb, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "product"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "release"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "steer"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "milestone"}, Measured: true, Debt: 0},
		{Member: card, Measured: true, Debt: 4},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if rep.TotalDebt != 4 {
		t.Errorf("total debt = %d, want the card's 4 folded in", rep.TotalDebt)
	}
	if rep.Satisfied {
		t.Error("a residual pile must keep the intent UNSATISFIED — four units owe an operator a look")
	}
	if len(rep.Worklist) == 0 {
		t.Fatal("empty worklist: a debt-bearing card must be enterable")
	}
	top := rep.Worklist[0]
	if top.Member.Ref != ospResidualRef {
		t.Fatalf("worst-first head = %q, want %q — the only card in debt", top.Member.Ref, ospResidualRef)
	}
	if top.Debt != 4 {
		t.Errorf("worklist debt = %d, want the folded 4 (a real number, not a placeholder)", top.Debt)
	}
	if !strings.Contains(top.Action, "steer prs") {
		t.Errorf("action = %q, want it to name the runnable steer-prs verb", top.Action)
	}
}

// TestOSPResidualCleanOverlayHoldsTheWalkClean pins the declared debt semantics: a
// clean overlay (nothing the kernel could not witness) holds the card at 0 and the
// reporting family reads Satisfied.
func TestOSPResidualCleanOverlayHoldsTheWalkClean(t *testing.T) {
	sb, _ := Lookup("tend-scoreboards")
	rep := Walk(sb, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "product"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "release"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "steer"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "milestone"}, Measured: true, Debt: 0},
		{Member: ospResidualMember(t), Measured: true, Debt: 0},
		{Member: steerprOverlayMember(t, sb), Measured: true, Debt: 0},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if !rep.Satisfied {
		t.Errorf("a clean overlay at 0 must satisfy the reporting family, got %+v", rep)
	}
	if rep.TotalDebt != 0 {
		t.Errorf("total debt = %d, want 0", rep.TotalDebt)
	}
}

// TestOSPResidualUnreadSurfacesUnmeasuredNotZero is the acceptance gate's honesty
// half. Break the payload source and the card cannot be pinned, so the walk's
// scorecard read misses it and reports UNMEASURED: the member blocks Satisfied and
// carries NO fabricated number. A clean 0 here would tell an operator the residual
// pile is empty when in truth nobody read it.
func TestOSPResidualUnreadSurfacesUnmeasuredNotZero(t *testing.T) {
	sb, _ := Lookup("tend-scoreboards")
	card := ospResidualMember(t)
	rep := Walk(sb, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "product"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "release"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "steer"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "milestone"}, Measured: true, Debt: 0},
		// The shell's KindScorecard miss path: the card errored at pin time, so its
		// key is absent from the pinned baseline (see the scorecardpane fence in
		// TestOSPResidualUnreadableIsUnmeasuredNeverZero).
		{Member: card, Measured: false, Detail: `key "osp_residual" not in pinned baseline @abc123`},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if rep.Unmeasured != 1 {
		t.Errorf("Unmeasured = %d, want 1 — an unreadable payload is a known gap", rep.Unmeasured)
	}
	if rep.Satisfied {
		t.Error("an UNMEASURED card must block Satisfied — an unread surface can never read clean")
	}
	if len(rep.Worklist) == 0 {
		t.Fatal("empty worklist: an unmeasured card must still be surfaced for entry")
	}
	top := rep.Worklist[0]
	if top.Member.Ref != ospResidualRef {
		t.Fatalf("worst-first head = %q, want the unmeasured %q — an unread member ranks with the worst", top.Member.Ref, ospResidualRef)
	}
	if top.Debt != 0 {
		t.Errorf("unmeasured debt = %d, want 0 carried with the UNMEASURED flag — never a fabricated count", top.Debt)
	}
}
