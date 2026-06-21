package kernel

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fakeWitness returns a scripted outcome for the require-witness gate test.
type fakeWitness struct{ out abi.WitnessOutcome }

func (f fakeWitness) Resolve(ctx context.Context, c *abi.ToolCall, claim string) abi.WitnessOutcome {
	return f.out
}

func requireWitness(claim string) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictRequireWitness, By: "test",
		Payload: abi.WitnessPayload{Claim: claim}}
}

// TestRequireWitnessConfirmedOpensGate — a CONFIRMED witness turns the gate into an
// Allow and the call reaches dispatch. The kernel did not take the claim on faith;
// it required corroboration and got it.
func TestRequireWitnessConfirmedOpensGate(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{requireWitness("ancestor:shipped")})
	abi.RegisterWitnessResolver("test", fakeWitness{abi.WitnessConfirmed})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")

	r, v := k.Syscall(context.Background(), call("deploy", "{}"))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("a confirmed witness must open the gate (Allow), got %v", v.Kind)
	}
	if r.Status != abi.StatusOK || atomic.LoadInt64(&eng.n) != 1 {
		t.Fatalf("a witnessed call must reach dispatch exactly once (engine n=%d)", eng.n)
	}
	if v.Meta["witness"] != "confirmed" {
		t.Fatalf("verdict should record the witness outcome, got %v", v.Meta)
	}
}

// TestRequireWitnessRefutedStaysClosed — a REFUTED claim (the agent lied about the
// effect) is a provable trust violation; the call never dispatches.
func TestRequireWitnessRefutedStaysClosed(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{requireWitness("ancestor:never-shipped")})
	abi.RegisterWitnessResolver("test", fakeWitness{abi.WitnessRefuted})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")

	_, v := k.Syscall(context.Background(), call("deploy", "{}"))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("a refuted claim must DENY/TRUST_VIOLATION, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("a refuted call must NOT dispatch (engine n=%d)", eng.n)
	}
}

// TestRequireWitnessAbstainFailsClosed — an ABSTAIN (no evidence either way) keeps
// the gate closed with ReasonUnwitnessed. The witness never blocks on its own
// uncertainty; the kernel's fail-closed default turns abstain into deny.
func TestRequireWitnessAbstainFailsClosed(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{requireWitness("grep:maybe")})
	abi.RegisterWitnessResolver("test", fakeWitness{abi.WitnessAbstain})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")

	_, v := k.Syscall(context.Background(), call("deploy", "{}"))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonUnwitnessed {
		t.Fatalf("an abstaining witness must DENY/UNWITNESSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("an unwitnessed call must NOT dispatch (engine n=%d)", eng.n)
	}
}

// TestRequireWitnessNoResolverPreservesV01 — with NO witness resolver registered,
// a require-witness verdict denies exactly as v0.1 did (fail-closed). This is the
// backward-compat guarantee: the wiring changes behavior only when a resolver is
// present.
func TestRequireWitnessNoResolverPreservesV01(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{requireWitness("ancestor:x")})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")

	_, v := k.Syscall(context.Background(), call("deploy", "{}"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("require-witness with no resolver must DENY (fail-closed), got %v", v.Kind)
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("must not dispatch without a witness (engine n=%d)", eng.n)
	}
	if c := k.Counters(); c.Denies != 1 {
		t.Fatalf("a require-witness deny must count as a deny, got %d", c.Denies)
	}
}
