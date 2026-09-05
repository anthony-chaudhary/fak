package engine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// retentionCases is the shared contract for the OpenRouter-style zero-data-retention
// knobs: for each {key: value} preference, does it request zero retention (=> the
// payload is sensitive => it must NOT take a remote route)? BOTH the engine
// adjudication floor (sensitiveRoute, via Adjudicate) and the modelroute resolution
// floor (Registry.Resolve) must agree on this table — that agreement is the mirror
// pinned by TestRetentionKnobMirror.
var retentionCases = []struct {
	key, value string
	sensitive  bool
}{
	{"zdr", "true", true},
	{"zdr", "1", true},
	{"zdr", "false", false},
	{"zdr", "0", false},
	{"data_collection", "deny", true},
	{"data_collection", "none", true},
	{"data_collection", "allow", false},
	{"retention", "none", true},
	{"retention", "zero", true},
	{"retention", "zdr", true},
	{"retention", "allow", false},
	{"retention", "any", false},
}

// TestResidencyGateDeniesZeroRetentionRemoteRoute: a zero-data-retention Meta knob
// makes a fleet-scoped call to a REMOTE engine DENY, exactly like a sensitivity tag,
// while a retention-allowed value defers.
func TestResidencyGateDeniesZeroRetentionRemoteRoute(t *testing.T) {
	for _, c := range retentionCases {
		v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
			Tool:   "summarize",
			Engine: "openai-primary", // remote (fail-closed classifier)
			Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet},
			Meta:   map[string]string{c.key: c.value},
		})
		gotDeny := v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonScopeCrossing
		if gotDeny != c.sensitive {
			t.Fatalf("%s=%q remote: gotDeny=%v want=%v (v=%v/%s)",
				c.key, c.value, gotDeny, c.sensitive, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestResidencyGateDefersZeroRetentionLocalRoute: a zero-retention request is only
// DENIED when the route is remote; a local/in-kernel route is fine (the floor blocks
// egress, not local compute).
func TestResidencyGateDefersZeroRetentionLocalRoute(t *testing.T) {
	v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
		Tool:   "summarize",
		Engine: "inkernel",
		Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet},
		Meta:   map[string]string{"zdr": "true"},
	})
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("zero-retention to a local engine should defer, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestRetentionKnobMirror pins the cross-package invariant: the engine adjudication
// floor and the modelroute resolution floor classify every retention knob identically.
// If one package learns a spelling the other does not, this fails — so the two
// mirrored floors can never silently disagree about what "zero data retention" means.
func TestRetentionKnobMirror(t *testing.T) {
	reg, err := modelroute.NewRegistry([]modelroute.ResolvedModel{
		{Alias: "remote", EngineID: "gpt-4o-mini", Provider: "openai", Remote: true},
	})
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	for _, c := range retentionCases {
		// engine floor: a remote route denies iff the knob marks the call sensitive.
		v := (residencyGate{}).Adjudicate(context.Background(), &abi.ToolCall{
			Tool:   "summarize",
			Engine: "openai-primary",
			Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeFleet},
			Meta:   map[string]string{c.key: c.value},
		})
		engineSensitive := v.Kind == abi.VerdictDeny

		// modelroute floor: a remote alias is refused iff the knob marks the subject sensitive.
		_, rerr := reg.Resolve(modelroute.Subject{Labels: map[string]string{c.key: c.value}}, "remote")
		routeSensitive := rerr != nil

		if engineSensitive != routeSensitive {
			t.Fatalf("mirror drift on %s=%q: engineDeny=%v modelrouteRefuse=%v",
				c.key, c.value, engineSensitive, routeSensitive)
		}
		if engineSensitive != c.sensitive {
			t.Fatalf("%s=%q: both floors agree (%v) but disagree with the contract (%v)",
				c.key, c.value, engineSensitive, c.sensitive)
		}
	}
}
