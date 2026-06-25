package ifc

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// gateCall builds a call whose Meta declares a share_target scope ("" = absent).
func gateCall(target string) *abi.ToolCall {
	c := &abi.ToolCall{Tool: "share", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	if target != "" {
		c.Meta = map[string]string{"share_target": target}
	}
	return c
}

// scopeResult builds a result tagged at scope carrying a recognizable payload the
// witness must NEVER disclose.
func scopeResult(scope abi.ShareScope) *abi.Result {
	return &abi.Result{
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("SECRET-PAYLOAD-MUST-NOT-LEAK"), Scope: scope},
	}
}

// foldResult mirrors kernel.admitResult's fold: most-restrictive verdict wins by
// FoldRank, defaulting to Allow (rank 0). It is the result-side analogue of
// kernel.Fold, kept local so the proof has no kernel import.
func foldResult(chain []abi.ResultAdmitter, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(best.Kind)
	for _, ra := range chain {
		v := ra.Admit(context.Background(), c, r)
		if rk := abi.FoldRank(v.Kind); rk > bestRank {
			bestRank, best = rk, v
		}
	}
	return best
}

// TestScopeCeilingGateConfinesWiderShare is the core acceptance table (criteria
// 2, 3, 5): a result is confined iff its declared scope is ≤ the declared share
// target scope; a wider result into a narrower target is Quarantined with
// ReasonTrustViolation attributed to the gate, and the witness discloses only the
// two scopes — never the payload.
func TestScopeCeilingGateConfinesWiderShare(t *testing.T) {
	gate := ScopeCeilingGate{}
	cases := []struct {
		name       string
		result     abi.ShareScope
		target     string // "" => no share_target tag
		wantKind   abi.VerdictKind
		wantReason abi.ReasonCode
	}{
		// Rung 0 — ScopeAgent (the private default) is a no-op regardless of target.
		{"agent result, no target", abi.ScopeAgent, "", abi.VerdictAllow, abi.ReasonNone},
		{"agent result, agent target", abi.ScopeAgent, "agent", abi.VerdictAllow, abi.ReasonNone},
		{"agent result, tenant target", abi.ScopeAgent, "tenant", abi.VerdictAllow, abi.ReasonNone},

		// Rung 1 — ScopeFleet result: confined iff target is fleet-or-wider.
		{"fleet result, agent target (crossing)", abi.ScopeFleet, "agent", abi.VerdictQuarantine, abi.ReasonTrustViolation},
		{"fleet result, fleet target (in-bounds)", abi.ScopeFleet, "fleet", abi.VerdictAllow, abi.ReasonNone},
		{"fleet result, tenant target (in-bounds)", abi.ScopeFleet, "tenant", abi.VerdictAllow, abi.ReasonNone},

		// Rung 1 — ScopeTenant result: confined iff target is tenant (the widest).
		{"tenant result, agent target (crossing)", abi.ScopeTenant, "agent", abi.VerdictQuarantine, abi.ReasonTrustViolation},
		{"tenant result, fleet target (crossing)", abi.ScopeTenant, "fleet", abi.VerdictQuarantine, abi.ReasonTrustViolation},
		{"tenant result, tenant target (in-bounds)", abi.ScopeTenant, "tenant", abi.VerdictAllow, abi.ReasonNone},
	}
	ctx := context.Background()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := gate.Admit(ctx, gateCall(tc.target), scopeResult(tc.result))
			if v.Kind != tc.wantKind {
				t.Fatalf("kind = %v, want %v", v.Kind, tc.wantKind)
			}
			if tc.wantKind == abi.VerdictQuarantine {
				if v.By != scopeCeilingBy {
					t.Fatalf("By = %q, want %q (the gate id)", v.By, scopeCeilingBy)
				}
				if v.Reason != tc.wantReason {
					t.Fatalf("Reason = %s, want %s", abi.ReasonName(v.Reason), abi.ReasonName(tc.wantReason))
				}
				// Criterion 5 — bounded disclosure: the witness names only the two
				// scopes; the payload bytes never appear in the Claim or the Meta.
				if wp, ok := v.Payload.(abi.WitnessPayload); ok {
					if strings.Contains(wp.Claim, "SECRET-PAYLOAD") {
						t.Fatalf("witness Claim discloses payload: %q", wp.Claim)
					}
				} else {
					t.Fatalf("quarantine verdict must carry a WitnessPayload, got %T", v.Payload)
				}
				for k, val := range v.Meta {
					if strings.Contains(val, "SECRET-PAYLOAD") {
						t.Fatalf("verdict Meta[%q] discloses payload: %q", k, val)
					}
				}
				if v.Meta["result_scope"] != scopeName(tc.result) {
					t.Fatalf("result_scope Meta = %q, want %q", v.Meta["result_scope"], scopeName(tc.result))
				}
			}
		})
	}
}

// TestScopeCeilingDefaultIsNoop is criterion 1: a ScopeAgent result folds
// verdict-identically with and without the gate registered — the "smallest rung
// does no work" guarantee. Mirrors the fold-equivalence pattern of
// kernel.TestScopedFoldEquivalentToFullChain.
func TestScopeCeilingDefaultIsNoop(t *testing.T) {
	gate := ScopeCeilingGate{}
	ctx := context.Background()
	r := scopeResult(abi.ScopeAgent)

	// Direct: the gate's identity verdict on a private result is admit-as-is.
	if v := gate.Admit(ctx, gateCall(""), r); v.Kind != abi.VerdictAllow {
		t.Fatalf("ScopeAgent result: gate verdict = %v, want Allow (identity)", v.Kind)
	}
	// Fold-equivalence: folding the chain WITH the gate equals folding it WITHOUT
	// for a ScopeAgent result (a rank-0 Allow never displaces the fold default).
	without := foldResult(nil, gateCall(""), r)
	with := foldResult([]abi.ResultAdmitter{gate}, gateCall(""), r)
	if without.Kind != with.Kind {
		t.Fatalf("ScopeAgent fold differs: without-gate=%v, with-gate=%v (gate must be a no-op)",
			without.Kind, with.Kind)
	}
}

// TestScopeCeilingUnknownTargetQuarantines is criterion 4: a wider-than-Agent
// result whose call carries NO readable share_target is INDETERMINATE and is
// confined (fail-closed) with a distinct share_target=unknown marker.
func TestScopeCeilingUnknownTargetQuarantines(t *testing.T) {
	gate := ScopeCeilingGate{}
	ctx := context.Background()

	for _, scope := range []abi.ShareScope{abi.ScopeFleet, abi.ScopeTenant} {
		// No Meta at all.
		v := gate.Admit(ctx, &abi.ToolCall{Tool: "share"}, scopeResult(scope))
		if v.Kind != abi.VerdictQuarantine || v.Reason != abi.ReasonTrustViolation {
			t.Fatalf("scope %s with nil Meta: verdict = %+v, want Quarantine/TRUST_VIOLATION", scopeName(scope), v)
		}
		if v.Meta["share_target"] != "unknown" {
			t.Fatalf("scope %s nil Meta: share_target = %q, want \"unknown\"", scopeName(scope), v.Meta["share_target"])
		}
		// Meta present but the tag absent / blank / unrecognized.
		for _, raw := range []string{"", "   ", "garbage"} {
			c := &abi.ToolCall{Tool: "share", Meta: map[string]string{"share_target": raw}}
			v := gate.Admit(ctx, c, scopeResult(scope))
			if v.Kind != abi.VerdictQuarantine || v.Meta["share_target"] != "unknown" {
				t.Fatalf("scope %s raw %q: verdict = %+v, want Quarantine/share_target=unknown", scopeName(scope), raw, v)
			}
		}
	}
}
