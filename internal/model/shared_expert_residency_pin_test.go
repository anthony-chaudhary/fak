package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// shared_expert_residency_pin_test.go - the dedicated #1304 witness: the ALWAYS-ON shared expert
// is DEVICE-PINNED and is never host-offloaded or streamed as a residency miss, across a
// MULTI-TOKEN trace, on BOTH arms of the graded spill and at the predicate level AND the live
// kernel routing level.
//
// #1304's contract, in one sentence: the shared expert is active on EVERY token (hit rate 1.0),
// so it is resident-hot by construction; host-offloading it streams a hot tensor out on the same
// path as the sparse COLD routed experts and pays a needless miss on every token. The fix splits
// the previously-conflated predicates: isExpertWeight stays the ACCOUNTING union (routed OR
// shared - the bytes still belong to the expert block for the load-time planner), while
// hostOffloadWeight is the runtime PLACEMENT predicate and is ROUTED-ONLY. gradedExpertSpillPredicate
// returns hostOffloadWeight ungraded and an explicit early-false on isSharedExpertWeight when
// graded. This file pins all three seams so a regression to `isExpertWeight` placement cannot pass.
//
// The witness style mirrors moe_offload_test.go and expert_ring_hal_test.go: literal canonical HF
// names (expertName / layerName), recording kernels/backends that make "which side ran this weight"
// directly observable, and a differential arm proving the split is actually ACTIVE (a routed expert
// really does leave the device) so the shared-expert assertions cannot pass vacuously.

// sharedExpertNames returns every shared-expert projection name the tiny GLM-DSA fixture's forward
// reads on `layer` - the three GEMMs glmSharedExperts (moe.go) issues. Naming all three (not just
// the gate) is deliberate: a predicate that pinned only gate_proj would still stream the up/down
// projections, so the pin must cover the whole shared-expert block.
func sharedExpertNames(layer int) []string {
	return []string{
		layerName(layer, "mlp.shared_experts.gate_proj.weight"),
		layerName(layer, "mlp.shared_experts.up_proj.weight"),
		layerName(layer, "mlp.shared_experts.down_proj.weight"),
	}
}

// TestSharedExpertIsDevicePinnedAcrossTokens is the load-bearing live-routing witness. It drives a
// real MULTI-TOKEN decode trace (Prefill of a prompt, then several Steps) through the actual
// GLM-DSA forward with CPUOffloadExperts on and a recordingBackend as the device side. The shared
// expert is DEVICE-PINNED: on every token of the trace it reaches the device backend and the host
// never sees it. The split is proven ACTIVE, not vacuously passing, by the differential arm: the
// routed experts do the OPPOSITE - they are host-offloaded, so none of them reaches the device -
// which is exactly the placement #1304 preserves for the routed class. A regression that routed the
// shared expert through isExpertWeight (the pre-#1304 bug) makes the shared name appear on the host
// side / vanish from the device side and fails here.
func TestSharedExpertIsDevicePinnedAcrossTokens(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() || !m.Cfg.IsMoE() || m.Cfg.NSharedExperts == 0 {
		t.Fatalf("fixture is not a glm_moe_dsa MoE model with shared experts (isGLMMoeDsa=%v IsMoE=%v NSharedExperts=%d)",
			m.Cfg.isGLMMoeDsa(), m.Cfg.IsMoE(), m.Cfg.NSharedExperts)
	}
	sharedLayers := []int{0, 1}
	routedNames := []string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 1, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
		expertName(1, 0, "gate_proj.weight"),
		expertName(1, 1, "up_proj.weight"),
		expertName(1, 0, "down_proj.weight"),
	}

	rec := newRecordingBackend(compute.Default())
	s := m.NewBackendSession(rec)
	s.CPUOffloadExperts = true // drives the splitKernel: routed -> host, shared -> device

	prompt := []int{3, 17, 5, 23}
	s.Prefill(prompt)
	// Every shared expert name this fixture carries must have run on the device at least once by the
	// end of the prompt, and the routed experts must NOT have - the split is active.
	seenShared := 0
	for _, l := range sharedLayers {
		for _, name := range sharedExpertNames(l) {
			if !rec.sawName(name) {
				t.Errorf("prefill: shared expert %q never reached the DEVICE backend - it must be device-pinned (#1304)", name)
			} else {
				seenShared++
			}
		}
	}
	if seenShared == 0 {
		t.Fatalf("prefill ran NO shared expert on the device - the name probe is broken (seen=%v)", rec.namesSeen())
	}

	// The differential arm: at least one routed expert was host-routed, so the split is genuinely
	// active rather than a no-op. Because the host side is not observable through a backend, the
	// ROUTED-ON-DEVICE ABSENCE is the observable half: an active split with CPUOffloadExperts on
	// sends every routed expert to host, so none may reach the device.
	for _, name := range routedNames {
		if rec.sawName(name) {
			t.Errorf("offload-hybrid reached routed expert %q on the DEVICE backend; the routed class must be host-offloaded under --n-cpu-moe", name)
		}
	}
	// And the placement predicate itself says the routed names go host while the shared names do not:
	// this is the direct proof that the split is ACTIVE and the absence above is meaningful.
	onHost := s.expertSpillOnHost()
	routedOnHost := 0
	for _, l := range sharedLayers {
		for _, name := range sharedExpertNames(l) {
			if onHost(name) {
				t.Errorf("session placement predicate routed shared expert %q to HOST - it must be device-pinned (#1304)", name)
			}
		}
	}
	for _, name := range routedNames {
		if onHost(name) {
			routedOnHost++
		}
	}
	if routedOnHost == 0 {
		t.Fatalf("no routed expert is host-routed by the session predicate - the split is inert, so the shared-expert assertions would pass vacuously")
	}

	// The MULTI-TOKEN trace: several decode steps after the prompt. The invariant is per-token and
	// cumulative: the shared expert must keep running on the device on each token (its device count
	// strictly increases across a step) while the routed experts never appear there.
	sharedGate := sharedExpertNames(0)[0]
	prevShared := rec.names[sharedGate]
	for step := 0; step < 4; step++ {
		next := []int{11, 7, 29, 2}[step]
		l := s.Step(next)
		if len(l) != cfg.VocabSize {
			t.Fatalf("step %d: logits shape %d, want vocab %d", step, len(l), cfg.VocabSize)
		}
		nowShared := rec.names[sharedGate]
		if nowShared <= prevShared {
			t.Fatalf("step %d: shared expert %q device MatMul count did not increase (%d -> %d) - it is not being run on the device every token",
				step, sharedGate, prevShared, nowShared)
		}
		prevShared = nowShared
		for _, name := range routedNames {
			if rec.sawName(name) {
				t.Fatalf("step %d: routed expert %q reached the DEVICE backend mid-trace - the routed class must stay host-offloaded", step, name)
			}
		}
	}
	t.Logf("shared expert %q ran on the DEVICE on all %d tokens (%d device MatMuls), routed experts stayed host-offloaded (%d routed names predicated to host)",
		sharedGate, 1+4, rec.names[sharedGate], routedOnHost)
}

// TestGradedSpillNeverHostsSharedExpert pins the GRADED arm explicitly. gradedExpertSpillPredicate
// (expert_spill_placement.go) takes the shared-expert early-false BEFORE the routed layer test, so
// the shared expert is device-pinned for EVERY spill grade - including a grade that spills the exact
// layer the shared expert lives on. The routed experts of the spilled layers still go host (the arm
// is active), and the routed experts of a KEPT layer stay on device. This is the predicate-level
// twin of the live-routing witness above: if the graded arm regressed to the accounting union, the
// shared names would flip to true for the fully-spilled grade and fail here.
func TestGradedSpillNeverHostsSharedExpert(t *testing.T) {
	m := expertSpillTestModel([]int{0, 1, 2}, nil, 2, 64)
	moeLayers := m.MoEExpertLayers()
	if len(moeLayers) != 3 {
		t.Fatalf("expertSpillTestModel MoE layers = %v, want [0 1 2]", moeLayers)
	}
	for _, grade := range []int{0, 1, 2, 3} {
		s := &Session{M: m, CPUOffloadExperts: true, ExpertSpillLayers: grade}
		onHost := s.expertSpillOnHost()
		// The shared expert is device-pinned on EVERY grade, including grade 3 which spills ALL
		// layers (the case the accounting union would wrongly send to host).
		for _, l := range moeLayers {
			for _, name := range sharedExpertNames(l) {
				if onHost(name) {
					t.Errorf("ExpertSpillLayers=%d: onHost(%q) = true, want false - the shared expert is device-pinned on every grade (#1304)", grade, name)
				}
			}
		}
		// The routed experts of the first `grade` MoE layers go host - the graded arm is ACTIVE, so
		// the shared-expert assertions above are not vacuous. Unspilled layers stay on device. Grade
		// 0 is the UNGRADED default (spilledExpertLayers(nil) == hostOffloadWeight), so it spills
		// EVERY routed expert rather than none - the same endpoint the existing
		// TestExpertSpillUngradedMatchesLegacyPredicate pins.
		routedToHost := 0
		for i, l := range moeLayers {
			for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				name := expertName(l, 0, suffix)
				got := onHost(name)
				want := grade == 0 || i < grade
				if got != want {
					t.Errorf("ExpertSpillLayers=%d: onHost(routed %q) = %v, want %v", grade, name, got, want)
				}
				if got {
					routedToHost++
				}
			}
		}
		if grade > 0 && routedToHost == 0 {
			t.Fatalf("ExpertSpillLayers=%d spilled no routed expert - the graded arm is inert, so the shared-expert assertions are vacuous", grade)
		}
	}
}

// TestSharedExpertStaysOutOfRoutedExpertRing pins the third seam: the bounded routed-expert ring
// (expert_ring_hal.go) must never own the shared expert's residency. routedExpertRing returns nil
// for a non-routed name, so a shared-expert stage takes the unchanged PERMANENT halW path and never
// creates or enters the ring, cannot be evicted, and cannot inflate the ring's eviction ledger. The
// routed expert is the contrast arm: it takes ring residency (under a declared budget) and NOT the
// permanent memoizer. A regression that classified the shared expert as a routed expert (so it were
// bounded and evictable) fails the residency assertions here and would show up in ExpertRingStats.
func TestSharedExpertStaysOutOfRoutedExpertRing(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 2)
	// Add a shared-expert projection to the synthetic model (the shared sibling of the routed gate).
	// expertRingTestModel builds routed experts only, so the shared arm needs its own resident tensor
	// with the same known footprint; this reuses its builder + resident-bytes helper rather than
	// inventing a second fixture.
	sharedName := "model.layers.0.mlp.shared_experts.gate_proj.weight"
	m.q4kw[sharedName] = m.q4kw[expertName(0, 0, "gate_proj.weight")]

	perWeight := expertRingWeightBytes(t, m)
	s := expertRingSession(m, perWeight*6) // budget for exactly two whole routed experts

	// Stage the shared expert FIRST: it must take permanent residency and must NOT create the ring.
	s.weightHALQ4K(sharedName, m.q4kw[sharedName])
	if _, ok := s.halW["q4k:"+sharedName]; !ok {
		t.Fatalf("shared expert %q did not take permanent halW residency", sharedName)
	}
	if s.expertRing != nil {
		t.Fatalf("staging the shared expert built a routed-expert ring (resident=%d) - the shared expert must stay out of the bounded ring", s.expertRing.residentCount())
	}

	// Now stage routed experts through the SAME session: the ring is built, and they take ring
	// residency rather than the permanent memoizer - the differential arm proving the ring is real.
	routed := []string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
		expertName(0, 1, "gate_proj.weight"),
		expertName(0, 1, "up_proj.weight"),
		expertName(0, 1, "down_proj.weight"),
	}
	for _, name := range routed {
		s.weightHALQ4K(name, m.q4kw[name])
	}
	if s.expertRing == nil {
		t.Fatalf("staging routed experts under a declared budget did not build the ring")
	}
	for _, name := range routed {
		if !s.expertRing.isResident("q4k:" + name) {
			t.Fatalf("routed expert %q did not take bounded ring residency", name)
		}
		if _, ok := s.halW["q4k:"+name]; ok {
			t.Fatalf("routed expert %q ALSO landed in the permanent halW memoizer", name)
		}
	}
	// The shared expert still owns permanent residency and is still absent from the ring.
	if _, ok := s.halW["q4k:"+sharedName]; !ok {
		t.Fatalf("shared expert %q lost its permanent residency after routed experts entered the ring", sharedName)
	}
	if s.expertRing.isResident("q4k:" + sharedName) {
		t.Fatalf("shared expert %q was admitted to the routed-expert ring after routed staging", sharedName)
	}

	// Re-stage the shared expert repeatedly across many "tokens": it is idempotent permanent
	// residency and must never page in, be evicted, or otherwise perturb the ring's ledger. Capture
	// the ring stats BEFORE and AFTER so any perturbation is visible; a shared expert that wrongly
	// entered the ring would page in on first stage and could be evicted under budget pressure.
	before := s.ExpertRing()
	for token := 0; token < 8; token++ {
		s.weightHALQ4K(sharedName, m.q4kw[sharedName])
	}
	after := s.ExpertRing()
	if after.Evictions != before.Evictions || after.PageIns != before.PageIns || after.Lookups != before.Lookups {
		t.Fatalf("shared-expert staging perturbed the ring ledger: before %+v after %+v", before, after)
	}
	if s.expertRing.isResident("q4k:" + sharedName) {
		t.Fatalf("shared expert %q appeared in the ring after repeated staging", sharedName)
	}
	t.Logf("shared expert stayed permanently resident and out of the routed-expert ring; ring evictions=%d pageIns=%d lookups=%d over the trace", after.Evictions, after.PageIns, after.Lookups)
}
