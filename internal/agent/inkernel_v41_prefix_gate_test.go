package agent

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_v41_prefix_gate_test.go is the fak#13335 acceptance witness: the serving
// PLANNER's eligibility seam must consume the complete-snapshot capabilities the model
// package exposes for V4.1, and must keep the bare-KV refusal intact.
//
// Before this leaf the planner asked only the legacy bare-KV predicate
// (model.Config.KVPrefixReuseSupported), which is FALSE for V4.1 by design — the bare
// *KVCache omits the committed token history and per-layer temporal state. So the host
// V4.1 route could never build a radix tree even though a complete HOST PrefixSnapshot
// (HostCompletePrefixSnapshotSupported, fak#13338) and a qualified DEVICE snapshot
// (InKernelBackendPrefixReuseSupportedFor, fak#13334/#13338) now exist.
//
// The gate is a pure predicate over (model, backend), so the witness drives it directly
// with counting fixtures rather than a full V4.1 forward: this is the planner's
// eligibility accounting, and it must reach the MODEL predicate for the exact backend in
// hand. Realized native first-request behavior is witnessed separately by the cmd-level
// leaves (fak#13326), per the leaf's own definition of done.

// v41GateBackend wraps a real backend and changes ONLY its Name(). It is transparent to
// every operation; the distinct identity is what proves the gate compares backend
// IDENTITY (the qualified "cpu-ref") rather than something the wrapped arithmetic shares.
type v41GateBackend struct {
	compute.Backend
	mu   sync.Mutex
	name string
}

func (b *v41GateBackend) Name() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.name
}

// tinyV41Cfg is a minimal config that IsDeepSeekV41() recognizes: the identity alone is
// what routes the planner's gate to the V4.1 branch, so the geometry is irrelevant here.
func tinyV41Cfg() model.Config {
	cfg := tinyCfg()
	cfg.ModelType = "deepseek_v41"
	return cfg
}

// TestV41PlannerPrefixGate is the named fak#13335 witness. It pins the four-way matrix at
// the planner seam and the constructor-level consequence (tree built or not).
func TestV41PlannerPrefixGate(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	v41 := tinyV41Cfg()
	if !v41.IsDeepSeekV41() {
		t.Fatal("precondition: tinyV41Cfg must be recognized as V4.1")
	}
	m := model.NewSynthetic(v41)

	t.Run("bare-KV predicate stays refused for V4.1", func(t *testing.T) {
		// The legacy incomplete-bare-cache guard the whole warming program rests on: this
		// leaf must NOT weaken it to enable the host route.
		if v41.KVPrefixReuseSupported() {
			t.Fatal("legacy KVPrefixReuseSupported now admits V4.1")
		}
	})

	t.Run("host V4.1 route is admitted through the complete-snapshot capability", func(t *testing.T) {
		if !v41.HostCompletePrefixSnapshotSupported() {
			t.Fatal("precondition: host V4.1 complete-snapshot capability is not present")
		}
		if !inKernelPlannerPrefixReuseSupported(m, nil) {
			t.Fatal("host V4.1 planner route was refused despite a complete host snapshot capability")
		}
	})

	cpuRef := compute.Pick("cpu-ref")
	if cpuRef == nil {
		t.Skip("cpu-ref backend is not registered in this build")
	}

	t.Run("qualified backend is admitted", func(t *testing.T) {
		if !inKernelPlannerPrefixReuseSupported(m, cpuRef) {
			t.Fatalf("qualified backend %q was refused for the V4.1 planner route", cpuRef.Name())
		}
	})

	t.Run("unqualified backend identity is refused", func(t *testing.T) {
		unqualified := &v41GateBackend{Backend: cpuRef, name: "v41-unqualified-planner-device"}
		if inKernelPlannerPrefixReuseSupported(m, unqualified) {
			t.Fatal("unqualified backend identity was admitted to the V4.1 planner route")
		}
	})

	t.Run("non-V4.1 architectures keep their existing admission", func(t *testing.T) {
		// A plain dense host model is unchanged; a generic device model stays refused (its
		// bare cache is incomplete on a device); a Qwen3.5 hybrid keeps its backend reuse.
		if !inKernelPlannerPrefixReuseSupported(model.NewSynthetic(tinyCfg()), nil) {
			t.Fatal("a plain host model lost prefix reuse")
		}
		generic := model.NewSynthetic(tinyCfg())
		if inKernelPlannerPrefixReuseSupported(generic, cpuRef) {
			t.Fatal("a generic device model was admitted; its bare cache is not a complete device prefix")
		}
		qwen := model.Config{ModelType: "qwen35", LayerTypes: []string{"linear_attention", "full_attention"}}
		if !inKernelPlannerPrefixReuseSupported(model.NewSynthetic(qwen), cpuRef) {
			t.Fatal("Qwen3.5 hybrid lost its backend prefix reuse")
		}
	})

	t.Run("constructor builds the tree for an admitted host V4.1 route only", func(t *testing.T) {
		// The constructor consequence: eligibility must actually turn on the radix tree the
		// rest of the planner consults, and an unqualified backend must leave it off.
		host := NewInKernelPlanner(m, nil, "v41-host", false, nil, false)
		if host.tree == nil {
			t.Fatal("an admitted host V4.1 planner built no radix tree")
		}
		if got := host.kvPrefixEligiblePromptTokens(32); got != 0 {
			// First cold turn: nothing admitted yet, so zero eligible is correct and proves
			// the eligibility witness runs the same predicate consistently.
			t.Fatalf("cold host V4.1 planner reports %d eligible prompt tokens, want 0", got)
		}

		unqualified := &v41GateBackend{Backend: cpuRef, name: "v41-unqualified-planner-device"}
		device := NewInKernelPlanner(m, nil, "v41-device", false, unqualified, false)
		if device.tree != nil {
			t.Fatal("an unqualified backend V4.1 planner built a radix tree it cannot safely serve")
		}
	})
}
