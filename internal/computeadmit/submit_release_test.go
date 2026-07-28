package computeadmit

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// The engine routes this file's ensemble members are routed onto (modelroute
// writes ToolCall.Engine pre-Submit); each is bound to the compute region it
// occupies so the gate can price it. seamRouteIdle is the kernel default a
// claim-less call falls back to.
const (
	seamRouteA    = "computeadmit-seam-member-a"
	seamRouteB    = "computeadmit-seam-member-b"
	seamRouteIdle = "computeadmit-seam-idle"
)

// seamEngine is the completion stub Reap dispatches to. The release seam under
// test is the kernel's Reap-time ResultAdmitter fold, so the engine only has to
// produce a result; it computes nothing.
type seamEngine struct{}

func (seamEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Call: c, Status: abi.StatusOK}, nil
}
func (seamEngine) Caps() []abi.Capability { return nil }

func init() {
	for _, id := range []string{seamRouteA, seamRouteB, seamRouteIdle} {
		abi.RegisterEngine(id, seamEngine{})
	}
}

// seamRelay is the process-global ResultAdmitter registration these tests share.
// abi.RegisterResultAdmitter has no unregister, so the tests register ONE relay
// and point it at the gate the running test installed — registering each test's
// own gate directly would leave every earlier gate in the global chain.
var seamRelay struct {
	mu   sync.Mutex
	ra   abi.ResultAdmitter
	once sync.Once
}

type seamRelayAdmitter struct{}

func (seamRelayAdmitter) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	seamRelay.mu.Lock()
	ra := seamRelay.ra
	seamRelay.mu.Unlock()
	if ra == nil {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "seam-relay"}
	}
	return ra.Admit(ctx, c, r)
}
func (seamRelayAdmitter) Caps() []abi.Capability { return nil }

// installReleaseSeam wires the gate into the kernel's Reap-time ResultAdmitter
// chain exactly the way a production wiring layer does — abi.RegisterResultAdmitter,
// a driver-blind registry seam, because internal/kernel is the driver-blind
// integrator and may never import this leaf (architest TestKernelImportsOnlyAbi).
//
// The interface assertion is deliberately a RUNTIME one: it IS the claim under
// test. A *SubmitAdmitter that cannot answer abi.ResultAdmitter has no release
// seam at all, and a compile-time `var _ abi.ResultAdmitter = ...` would turn
// that missing half into a build error instead of a legible failure.
func installReleaseSeam(t *testing.T, gate *SubmitAdmitter) {
	t.Helper()
	ra, ok := any(gate).(abi.ResultAdmitter)
	if !ok {
		t.Fatalf("*SubmitAdmitter does not implement abi.ResultAdmitter: nothing frees a compute " +
			"region when its holder's call completes, so the first admitted ensemble member holds " +
			"its device for the life of the process and every co-targeted peer refuses forever — " +
			"that is a wedge, not serialization")
	}
	seamRelay.once.Do(func() { abi.RegisterResultAdmitter(90, seamRelayAdmitter{}) })
	seamRelay.mu.Lock()
	seamRelay.ra = ra
	seamRelay.mu.Unlock()
	t.Cleanup(func() {
		seamRelay.mu.Lock()
		seamRelay.ra = nil
		seamRelay.mu.Unlock()
	})
}

// Acceptance (#3269), the Submit half made whole: two co-targeted ensemble
// members are SERIALIZED at Kernel.Submit with a COLLISION_RISK — the second
// refuses while the first holds the device, and admits once the first's call
// COMPLETES, with no scheduler, no queue, and no out-of-band Release poke. The
// handoff rides the kernel's own Reap-time result-admission fold, so the
// serialization is real in a shipped process and not only in a test that pokes
// the gate by hand.
func TestSubmitSeamFreesTheRegionWhenTheHolderReaps(t *testing.T) {
	gate := NewSubmitAdmitter(Taxonomy{})
	device0 := dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0"}
	gate.BindRoute(seamRouteA, device0)
	gate.BindRoute(seamRouteB, device0)
	installReleaseSeam(t, gate)

	k := kernel.New(seamRouteIdle, kernel.WithAdjudicators([]abi.Adjudicator{gate, allowTail{}}))
	ctx := context.Background()

	hA, vA := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: seamRouteA, TraceID: "seam-member-1"})
	if vA.Kind != abi.VerdictAllow {
		t.Fatalf("first member verdict = %v (reason %s), want allow", vA.Kind, abi.ReasonName(vA.Reason))
	}

	_, vB := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: seamRouteB, TraceID: "seam-member-2"})
	if vB.Kind != abi.VerdictDeny || vB.Reason != ReasonComputeCollision {
		t.Fatalf("co-targeted second member: kind=%v reason=%s, want deny COLLISION_RISK",
			vB.Kind, abi.ReasonName(vB.Reason))
	}
	if vB.Meta["rung"] != RungRegionCollision || vB.Meta["conflict"] != "seam-member-1" {
		t.Fatalf("deny meta = %v, want rung=%s conflict=seam-member-1", vB.Meta, RungRegionCollision)
	}

	// The holder's call has not completed, so its region is still held.
	if live := gate.Live(); len(live) != 1 || live[0].ID != "seam-member-1" {
		t.Fatalf("live leases before completion = %+v, want [seam-member-1]", live)
	}

	rA, err := k.Reap(ctx, hA)
	if err != nil {
		t.Fatalf("reap first member: %v", err)
	}
	if rA == nil || rA.Status != abi.StatusOK {
		t.Fatalf("first member result = %+v, want a StatusOK completion", rA)
	}
	// Acceptance item 3 on the result side: the release seam is no-opinion, so
	// the result-admission verdict of every existing chain is unchanged.
	if got := rA.Meta["admit"]; got != "" {
		t.Fatalf("release seam changed result admission: admit=%q, want unchanged (no opinion)", got)
	}

	if live := gate.Live(); len(live) != 0 {
		t.Fatalf("the holder's compute region is STILL held after its call completed: %+v — "+
			"the Reap-time release seam never fired, so the serialized member can never admit", live)
	}

	_, vB2 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: seamRouteB, TraceID: "seam-member-2"})
	if vB2.Kind != abi.VerdictAllow {
		t.Fatalf("serialized member after the holder completed: kind=%v reason=%s, want allow",
			vB2.Kind, abi.ReasonName(vB2.Reason))
	}
	if live := gate.Live(); len(live) != 1 || live[0].ID != "seam-member-2" {
		t.Fatalf("live leases after the handoff = %+v, want [seam-member-2]", live)
	}
}

// A claim-less call that merely SHARES the holder's trace id must never hand the
// holder's device away underneath it: the release is keyed on the completing
// call's own compute claim, not on its identity alone. Without this the very
// first ordinary tool call an ensemble member makes would silently un-serialize
// its peers.
func TestSubmitSeamReleaseIgnoresAClaimlessCompletion(t *testing.T) {
	gate := NewSubmitAdmitter(Taxonomy{})
	gate.BindRoute(seamRouteA, dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0"})
	installReleaseSeam(t, gate)

	k := kernel.New(seamRouteIdle, kernel.WithAdjudicators([]abi.Adjudicator{gate, allowTail{}}))
	ctx := context.Background()

	if _, v := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: seamRouteA, TraceID: "seam-holder"}); v.Kind != abi.VerdictAllow {
		t.Fatalf("holder verdict = %v (reason %s), want allow", v.Kind, abi.ReasonName(v.Reason))
	}

	h, v := k.Submit(ctx, &abi.ToolCall{Tool: "read", TraceID: "seam-holder"})
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("claim-less call verdict = %v, want allow via the chain tail", v.Kind)
	}
	if _, err := k.Reap(ctx, h); err != nil {
		t.Fatalf("reap claim-less call: %v", err)
	}

	if live := gate.Live(); len(live) != 1 || live[0].ID != "seam-holder" {
		t.Fatalf("a claim-less completion under the holder's trace id handed its compute region "+
			"away: live=%+v, want the holder still on device:0", live)
	}
}
