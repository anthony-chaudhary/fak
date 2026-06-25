package advmodel

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// TestTrainedArtifactEndToEnd is the Go/Python parity + end-to-end witness. It
// loads the committed trained artifact (testdata/adjudicator.json, produced by
// train.py) and asserts the Go scorer reproduces the trained decision on EVERY
// corpus row: it denies the floor's deny-worthy calls and defers the allows. A
// drift between Go's Tokens() and train.py's featurizer, or a stale/corrupt
// artifact, fails here. It also re-checks the held-out split train.py used
// (every 5th row) so the printed F1 is reproducible from Go too.
func TestTrainedArtifactEndToEnd(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		t.Skipf("no trained artifact yet (run python internal/advmodel/train.py): %v", err)
	}
	art, err := LoadBytes(b)
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}
	rows := LabelCalls(CorpusCalls())

	var denyCorrect, allowCorrect, total int
	var heldDenyCorrect, heldAllowCorrect, heldTotal int
	for i, r := range rows {
		total++
		denies := art.Denies(r.Tool, []byte(r.Args))
		if r.Deny == denies {
			if r.Deny {
				denyCorrect++
			} else {
				allowCorrect++
			}
		} else {
			t.Errorf("row %d (%s %s): model.Denies=%v floor.Deny=%v (Go/Python parity or artifact drift)",
				i, r.Tool, r.Args, denies, r.Deny)
		}
		// train.py's held-out split: every 5th row (index 4,9,14,...).
		if i%5 == 4 {
			heldTotal++
			if r.Deny == denies {
				if r.Deny {
					heldDenyCorrect++
				} else {
					heldAllowCorrect++
				}
			}
		}
	}
	if total > 0 && denyCorrect+allowCorrect != total {
		t.Fatalf("trained artifact misclassified %d/%d corpus rows in Go (parity drift)",
			total-denyCorrect-allowCorrect, total)
	}
	t.Logf("Go scorer reproduces trained model on all %d corpus rows (parity OK)", total)
	t.Logf("held-out split: %d rows, %d/%d correct (precision/recall both %d/%d in Go)",
		heldTotal, heldDenyCorrect+heldAllowCorrect, heldTotal, heldDenyCorrect, heldDenyCorrect)
	t.Logf("artifact meta: trained f1=%v vs stock_ref_f1=%v (majority_f1=%v, train_rows=%d held_rows=%d)",
		art.Meta.F1, art.Meta.StockF1, art.Meta.MajorityF1, art.Meta.TrainRows, art.Meta.HeldRows)
}

// kindName renders a verdict kind for readable test failures (abi has no
// exported Stringer on VerdictKind).
func kindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "Allow"
	case abi.VerdictDeny:
		return "Deny"
	case abi.VerdictTransform:
		return "Transform"
	case abi.VerdictQuarantine:
		return "Quarantine"
	case abi.VerdictRequireWitness:
		return "RequireWitness"
	case abi.VerdictDefer:
		return "Defer"
	case abi.VerdictIndeterminate:
		return "Indeterminate"
	default:
		return "Unknown"
	}
}

// allowAdj is a floor stand-in that affirmatively allows everything; used to
// isolate the advisory model's effect under the kernel fold.
type allowAdj struct{}

func (allowAdj) Caps() []abi.Capability { return nil }
func (allowAdj) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "floor"}
}

// denyAdj is a floor stand-in that denies everything.
type denyAdj struct{}

func (denyAdj) Caps() []abi.Capability { return nil }
func (denyAdj) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "floor"}
}

// miniArtifact is a hand-built classifier used to exercise the WIRING (the
// fail-closed contract + the fold) independent of whether training has produced
// a good artifact. bias -1, threshold 0: a call whose token weights sum >= 1 is
// denied; everything else defers.
func miniArtifact() *Artifact {
	return &Artifact{
		Schema:    ArtifactSchema,
		Bias:      -1.0,
		Threshold: 0.0,
		Features:  map[string]float64{"refund_payment": 5.0, "delete_account": 5.0},
	}
}

func TestTokensDeterministic(t *testing.T) {
	// The featurizer is the Go/Python parity contract. Pin a known call to known
	// tokens so a drift in tokenRe is caught here, not in a silent mis-score.
	got := Tokens("refund_payment", []byte(`{"order":"o-1"}`))
	want := map[string]bool{"refund_payment": true, "order": true, "o": true, "1": true}
	for _, tok := range got {
		if !want[tok] {
			t.Errorf("unexpected token %q (featurizer drift — train.py would score the wrong feature)", tok)
		}
	}
	for k := range want {
		found := false
		for _, tok := range got {
			if tok == k {
				found = true
			}
		}
		if !found {
			t.Errorf("missing expected token %q (got %v)", k, got)
		}
	}
}

func TestArtifactRoundTrip(t *testing.T) {
	a := miniArtifact()
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := LoadBytes(b)
	if err != nil {
		t.Fatalf("LoadBytes: %v", err)
	}
	if got.Bias != a.Bias || got.Threshold != a.Threshold || len(got.Features) != len(a.Features) {
		t.Fatalf("round-trip lost data: got %+v", got)
	}
	// An unknown schema is rejected, not coerced (a future v2 artifact can never
	// be silently mis-scored by this v1 loader).
	bad, _ := json.Marshal(&Artifact{Schema: "fak-advmodel/v9", Features: map[string]float64{"x": 1}})
	if _, err := LoadBytes(bad); err == nil {
		t.Fatal("LoadBytes accepted an unknown schema")
	}
}

// TestFailClosedNeverAllows is the load-bearing security contract: across every
// input — deny-worthy, benign, empty args, an unknown tool, and crucially a NIL
// (inert/unloaded) artifact — the advisory adjudicator NEVER returns VerdictAllow.
// Under the kernel fold (most-restrictive non-Defer wins) that means it can only
// ever add a deny, never weaken the deterministic floor. A nil artifact defers
// on everything (a pure no-op), so a mis-loaded model is safe.
func TestFailClosedNeverAllows(t *testing.T) {
	calls := []struct {
		tool string
		args string
	}{
		{"refund_payment", `{"order":"o-1"}`},         // deny-worthy (known token)
		{"delete_account", `{"account":"a"}`},         // deny-worthy (known token)
		{"search_kb", `{"q":"returns"}`},              // benign
		{"read_customer_record", `{"id":"c-1"}`},      // benign
		{"", ``},                                      // empty
		{"unknown_tool_xyz", `{"k":"v"}`},             // unseen
		{"create_support_ticket", `{"body":"hello"}`}, // benign
	}
	ctx := context.Background()

	// Trained (mini) artifact: must deny the two tokens it knows, defer the rest,
	// and NEVER allow. (Floor-coverage of every deny is the trained-artifact
	// test's job — this test is the WIRING contract, with a deliberately minimal
	// 2-token artifact.)
	adj := NewAdjudicator(miniArtifact())
	knownDeny := map[string]bool{"refund_payment": true, "delete_account": true}
	for _, c := range calls {
		v := adj.Adjudicate(ctx, &abi.ToolCall{Tool: c.tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(c.args)}})
		if v.Kind == abi.VerdictAllow {
			t.Errorf("call %s %s: advisory returned Allow (violates fail-closed)", c.tool, c.args)
		}
		if knownDeny[c.tool] {
			if v.Kind != abi.VerdictDeny {
				t.Errorf("known-token %s: advisory did not corroborate (got %s) — wiring broken", c.tool, kindName(v.Kind))
			}
		}
	}

	// Nil (inert) artifact: pure no-op — defers on everything, never allows/denies.
	inert := NewAdjudicator(nil)
	for _, c := range calls {
		v := inert.Adjudicate(ctx, &abi.ToolCall{Tool: c.tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(c.args)}})
		if v.Kind != abi.VerdictDefer {
			t.Errorf("inert artifact on %s returned %s, want Defer (no-op)", c.tool, kindName(v.Kind))
		}
	}
}

// TestFoldSafety proves the fail-closed claim UNDER the real kernel fold
// (kernel.Fold: most-restrictive non-Defer verdict wins). The advisory model can
// only TIGHTEN a decision: with a floor that allows, an advisory Defer leaves it
// allowed (benign passes through) while an advisory Deny turns it into a deny
// (corroborate). With a floor that denies, the floor's deny always stands
// regardless of the advisory — the deterministic capability floor governs.
func TestFoldSafety(t *testing.T) {
	ctx := context.Background()
	benignCall := &abi.ToolCall{Tool: "search_kb", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"x"}`)}}
	denyCall := &abi.ToolCall{Tool: "refund_payment", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"o":"1"}`)}}

	corroborate := NewAdjudicator(miniArtifact()) // denies refund_payment, defers search_kb

	// floor allows + advisory defers (benign) -> stays allowed: the model does not
	// block benign traffic.
	v := kernel.Fold(ctx, []abi.Adjudicator{allowAdj{}, corroborate}, benignCall)
	if v.Kind != abi.VerdictAllow {
		t.Errorf("benign under allow-floor + advisory: got %s, want Allow (advisory must not block benign)", kindName(v.Kind))
	}
	// floor allows + advisory denies (corroborate) -> tightened to deny.
	v = kernel.Fold(ctx, []abi.Adjudicator{allowAdj{}, corroborate}, denyCall)
	if v.Kind != abi.VerdictDeny {
		t.Errorf("deny-worthy under allow-floor + advisory: got %s, want Deny (advisory corroborates)", kindName(v.Kind))
	}
	// floor denies + advisory defers -> floor's deny stands (floor governs).
	v = kernel.Fold(ctx, []abi.Adjudicator{denyAdj{}, corroborate}, benignCall)
	if v.Kind != abi.VerdictDeny {
		t.Errorf("floor-deny + advisory-defer: got %s, want Deny (floor governs)", kindName(v.Kind))
	}
	// An advisory-only chain (no floor) that defers on a benign call collapses to
	// the fail-closed default deny (empty/all-defer policy) — never an allow.
	v = kernel.Fold(ctx, []abi.Adjudicator{corroborate}, benignCall)
	if v.Kind != abi.VerdictDeny {
		t.Errorf("advisory-only benign: got %s, want Deny (fail-closed default)", kindName(v.Kind))
	}
}
