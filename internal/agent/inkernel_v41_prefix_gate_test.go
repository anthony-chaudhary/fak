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

// v41CountingBackend is the CW-28 counting capability fixture. It wraps a real backend and
// counts how many times its Name() identity is consulted, so the witness can prove the
// planner seam reaches the MODEL predicate FOR THIS BACKEND rather than admitting on a
// model-side capability that ignores the device. It changes ONLY the reported name.
type v41CountingBackend struct {
	compute.Backend
	mu    sync.Mutex
	name  string
	calls int
}

func (b *v41CountingBackend) Name() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.calls++
	return b.name
}

func (b *v41CountingBackend) nameCalls() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls
}

// TestV41BackendPlannerPrefixGate is the named fak#13330 (agent-startup-cache-warm CW-28)
// witness. fak#13335 landed the V4.1 planner branch on the complete-snapshot capabilities;
// this leaf owns the BACKEND-AWARE half: the selected backend's IDENTITY must reach the
// model predicate (Config.InKernelBackendPrefixReuseSupportedFor), a second unqualified
// backend with the SAME model configuration must be refused, host/non-V4.1 eligibility is
// unchanged, and constructor success alone must not fabricate a warm/hit receipt.
func TestV41BackendPlannerPrefixGate(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	v41 := tinyV41Cfg()
	if !v41.IsDeepSeekV41() {
		t.Fatal("precondition: tinyV41Cfg must be recognized as V4.1")
	}
	m := model.NewSynthetic(v41)

	cpuRef := compute.Pick("cpu-ref")
	if cpuRef == nil {
		t.Skip("cpu-ref backend is not registered in this build")
	}

	// sameConfigQualified and sameConfigUnqualified share ONE model configuration (v41,
	// identical geometry); only the backend descriptor identity differs. Any admission
	// divergence between them can therefore only come from the backend, never the config.
	qualified := &v41CountingBackend{Backend: cpuRef, name: cpuRef.Name()}
	unqualified := &v41CountingBackend{Backend: cpuRef, name: "v41-unqualified-planner-device"}

	t.Run("qualified selected backend constructs the scoped prefix cache", func(t *testing.T) {
		p := NewInKernelPlanner(m, nil, "v41-qualified", false, qualified, false)
		if p.tree == nil {
			t.Fatal("qualified selected backend built no scoped prefix cache")
		}
		if p.scopedTree == nil {
			t.Fatal("qualified selected backend built no scoped tree wrapper")
		}
		if qualified.nameCalls() == 0 {
			t.Fatal("qualified backend identity was never consulted by the admission seam")
		}
	})

	t.Run("same-config unqualified backend constructs no cache", func(t *testing.T) {
		p := NewInKernelPlanner(m, nil, "v41-unqualified", false, unqualified, false)
		if p.tree != nil {
			t.Fatal("an unqualified backend with the same model configuration built a cache")
		}
		if unqualified.nameCalls() == 0 {
			t.Fatal("unqualified backend identity was never consulted by the admission seam")
		}
	})

	t.Run("counting fixture proves the selected backend reaches the model predicate", func(t *testing.T) {
		// The predicate is backend-AWARE: the counting identity must be consulted, and the
		// two same-config backends must take different branches.
		before := qualified.nameCalls()
		if !v41.InKernelBackendPrefixReuseSupportedFor(qualified) {
			t.Fatal("the qualified backend did not reach the model predicate as admitted")
		}
		if qualified.nameCalls() <= before {
			t.Fatal("the model predicate answered without consulting the backend identity")
		}
		if v41.InKernelBackendPrefixReuseSupportedFor(unqualified) {
			t.Fatal("the model predicate admitted an unqualified backend identity")
		}
		// And the seam that consumes it must agree, for the exact same config.
		if inKernelPlannerPrefixReuseSupported(m, qualified) == inKernelPlannerPrefixReuseSupported(m, unqualified) {
			t.Fatal("the planner seam did not distinguish two backends under one model configuration")
		}
	})

	t.Run("host V4.1 and existing Qwen/GLM eligibility unchanged", func(t *testing.T) {
		if !v41.HostCompletePrefixSnapshotSupported() || !inKernelPlannerPrefixReuseSupported(m, nil) {
			t.Fatal("host V4.1 complete-snapshot eligibility regressed")
		}
		glm := model.Config{
			ModelType:     "glm_moe_dsa",
			Architectures: []string{"GlmMoeDsaForCausalLM"},
		}
		if !inKernelPlannerPrefixReuseSupported(model.NewSynthetic(glm), nil) {
			t.Fatal("GLM host eligibility unchanged-from-before failed")
		}
		qwen := model.Config{ModelType: "qwen35", LayerTypes: []string{"linear_attention", "full_attention"}}
		if !inKernelPlannerPrefixReuseSupported(model.NewSynthetic(qwen), cpuRef) {
			t.Fatal("Qwen3.5 hybrid backend eligibility regressed")
		}
	})

	t.Run("constructor success alone emits no warm or hit receipt", func(t *testing.T) {
		p := NewInKernelPlanner(m, nil, "v41-qualified-nohit", false, qualified, false)
		if p.tree == nil {
			t.Fatal("precondition: qualified planner must have built a cache")
		}
		if p.WarmClaimLive() {
			t.Fatal("constructor success alone reported a live warm claim")
		}
		if p.kvPrefixEverAdmitted.Load() {
			t.Fatal("constructor success alone flipped the first-demand hit latch")
		}
		if got := p.kvPrefixEligiblePromptTokens(64); got != 0 {
			t.Fatalf("cold planner reported %d eligible prompt tokens, want 0", got)
		}
	})
}
