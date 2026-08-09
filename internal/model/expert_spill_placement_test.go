package model

import (
	"errors"
	"testing"
)

// expert_spill_placement_test.go — the R1 witnesses for #5612 (epic #5606). R0 bounded the activated set
// but left it session-only and unsized; R1 makes the placement GRADED and DERIVED FROM MEASURED
// BYTES. The claims, one test each:
//
//	ordinal      — "the first N" counts the model's real MoE layers, not the first N layers;
//	ungraded     — an unset grade is byte-for-byte the legacy all-or-nothing predicate;
//	placement    — a graded split really does route only the first N layers' experts to host, and
//	               the forward it produces is BIT-EXACT with the plain one (pure placement);
//	sizing       — the plan is computed from the loaded model's own resident bytes, and its ring
//	               budget is the leftover device headroom, capped by what the residents would need;
//	refusal      — an out-of-range operator N is refused, never silently clamped.

// expertSpillTestModel builds a model whose ONLY residency is a known, exactly-divisible set of
// bytes: `moeLayers` routed-expert layers of `experts` experts each (3 projections apiece) plus one
// dense projection per layer and an LM head. Every byte quantity the sizing math reads is therefore
// an exact multiple, so a fit can be asserted to the byte rather than to a tolerance.
func expertSpillTestModel(moeLayers []int, denseLayers []int, experts, bytesPerWeight int) *Model {
	m := &Model{q4kw: map[string]*q4kTensor{}}
	put := func(name string) {
		m.q4kw[name] = &q4kTensor{raw: make([]byte, bytesPerWeight)}
	}
	for _, l := range append(append([]int{}, denseLayers...), moeLayers...) {
		put(layerName(l, "self_attn.q_proj.weight"))
	}
	for _, l := range moeLayers {
		put(routerName(l))
		for e := 0; e < experts; e++ {
			for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				put(expertName(l, e, suffix))
			}
		}
	}
	put("lm_head.weight")
	return m
}

// TestExpertSpillLayersAreMoEOrdinals is the ordinal witness. `--n-cpu-moe N` moves the first N MoE
// layers — and on a hybrid checkpoint whose first layers are DENSE (FirstKDenseReplace, moe.go) those
// are not layers 0..N-1. Counting layers instead of MoE layers would silently spill N-k of them.
func TestExpertSpillLayersAreMoEOrdinals(t *testing.T) {
	m := expertSpillTestModel([]int{2, 3, 4, 5}, []int{0, 1}, 2, 128)

	got := m.MoEExpertLayers()
	want := []int{2, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("MoEExpertLayers() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MoEExpertLayers() = %v, want %v (ascending MoE ordinals, dense prefix excluded)", got, want)
		}
	}

	// Spilling 2 must move MoE layers 2 and 3 — NOT layers 0 and 1, which carry no experts at all.
	s := &Session{M: m, CPUOffloadExperts: true, ExpertSpillLayers: 2}
	onHost := s.expertSpillOnHost()
	cases := []struct {
		name string
		want bool
	}{
		{expertName(2, 0, "gate_proj.weight"), true},                  // first spilled MoE layer
		{expertName(3, 1, "down_proj.weight"), true},                  // second spilled MoE layer
		{expertName(4, 0, "up_proj.weight"), false},                   // kept on the device
		{expertName(5, 1, "gate_proj.weight"), false},                 // kept on the device
		{"model.layers.2.mlp.shared_experts.gate_proj.weight", true},  // the spilled layer's whole MoE block
		{"model.layers.5.mlp.shared_experts.gate_proj.weight", false}, // ...and the kept layer's stays
		{routerName(2), false},                                        // the router is dense: never spills
		{layerName(0, "self_attn.q_proj.weight"), false},
		{layerName(2, "self_attn.q_proj.weight"), false},
		{"lm_head.weight", false},
	}
	for _, c := range cases {
		if got := onHost(c.name); got != c.want {
			t.Errorf("graded onHost(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestExpertSpillUngradedMatchesLegacyPredicate is the default-unchanged witness: an unset grade —
// every session in the tree today — and a grade at or above the MoE layer count both resolve to
// exactly isExpertWeight, so the split placement is byte-for-byte what it was before #5612.
func TestExpertSpillUngradedMatchesLegacyPredicate(t *testing.T) {
	m := expertSpillTestModel([]int{0, 1, 2}, nil, 2, 64)
	names := []string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(2, 1, "down_proj.weight"),
		"model.layers.1.mlp.shared_experts.up_proj.weight",
		routerName(0),
		layerName(1, "self_attn.q_proj.weight"),
		"lm_head.weight",
		"model.embed_tokens.weight", // no layer ordinal at all
	}
	for _, n := range []int{0, -1, 3, 99} {
		s := &Session{M: m, CPUOffloadExperts: true, ExpertSpillLayers: n}
		onHost := s.expertSpillOnHost()
		for _, name := range names {
			if got, want := onHost(name), isExpertWeight(name); got != want {
				t.Errorf("ExpertSpillLayers=%d: onHost(%q) = %v, want the ungraded isExpertWeight = %v", n, name, got, want)
			}
		}
	}
}

// TestGradedSpillRoutesOnlyTheFirstLayersToHost is the live-placement witness, driven through the
// kernel decodeBandGLMDsa actually runs (glmDsaMatKernel). With a 2-MoE-layer checkpoint graded at
// N=1, layer 0's expert GEMMs must reach the HOST kernel and layer 1's must NOT — the state the
// all-or-nothing predicate could not express — while the dense router stays on the device either way.
func TestGradedSpillRoutesOnlyTheFirstLayersToHost(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if layers := m.MoEExpertLayers(); len(layers) != 2 {
		t.Fatalf("fixture has MoE layers %v, want exactly 2 so a grade of 1 is a real interior split", layers)
	}

	s := m.NewSession()
	s.CPUOffloadExperts = true
	s.ExpertSpillLayers = 1
	host := newRecordingKernel(residentKernel{m})
	device := newRecordingKernel(residentKernel{m})
	k := splitKernel{host: host, device: device, onHost: s.expertSpillOnHost()}

	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(i)*0.01 - 0.1
	}
	prepped := k.prep(x)
	cases := []struct {
		name       string
		out, in    int
		wantOnHost bool
	}{
		{expertName(0, 0, "gate_proj.weight"), m.Cfg.IntermediateSize, H, true},
		{expertName(0, 1, "up_proj.weight"), m.Cfg.IntermediateSize, H, true},
		{expertName(1, 0, "gate_proj.weight"), m.Cfg.IntermediateSize, H, false},
		{expertName(1, 1, "up_proj.weight"), m.Cfg.IntermediateSize, H, false},
		{routerName(0), m.Cfg.NumExperts, H, false},
		{routerName(1), m.Cfg.NumExperts, H, false},
	}
	for _, c := range cases {
		got := k.mul(c.name, prepped, c.out, c.in)
		want := residentKernel{m}.mul(c.name, prepped, c.out, c.in)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: graded split changed the arithmetic at %d (%v != %v)", c.name, i, got[i], want[i])
			}
		}
		if c.wantOnHost && (!host.saw(c.name) || device.saw(c.name)) {
			t.Errorf("%s: want HOST (spilled layer), host=%v device=%v", c.name, host.saw(c.name), device.saw(c.name))
		}
		if !c.wantOnHost && (host.saw(c.name) || !device.saw(c.name)) {
			t.Errorf("%s: want DEVICE (kept layer), host=%v device=%v", c.name, host.saw(c.name), device.saw(c.name))
		}
	}

	// Placement invariance, the same honesty gate the ungraded split keeps: with no backend the
	// graded split is split(host=resident, device=resident), so the full forward must be BIT-EXACT
	// with the plain one. A grade changes WHERE a GEMM runs, never WHAT it computes.
	prompt := []int{3, 17, 5, 23}
	lPlain := m.NewSession().Prefill(prompt)
	sG := m.NewSession()
	sG.CPUOffloadExperts = true
	sG.ExpertSpillLayers = 1
	lGraded := sG.Prefill(prompt)
	if d, at := maxAbsDiff(lPlain, lGraded); d != 0 {
		t.Fatalf("graded-spill forward max|Δ|=%.3e at %d != 0 — the grade is not a pure placement decision", d, at)
	}
}

// TestPlanExpertSpillSizesFromResidentBytes is the sizing witness — the rung's point. The plan is
// computed from the model's OWN resident bytes (MoEResidentWeightBytes), so N and the ring budget
// answer "what fits THIS device", and applying it makes R0's ring operator-reachable for the first
// time: the session's ring budget is the plan's, to the byte.
func TestPlanExpertSpillSizesFromResidentBytes(t *testing.T) {
	const (
		perWeight = 1024
		experts   = 2
		moe       = 4
	)
	m := expertSpillTestModel([]int{0, 1, 2, 3}, nil, experts, perWeight)
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok {
		t.Fatal("MoEResidentWeightBytes reports nothing resident on a model built with expert weights")
	}
	perLayer := int64(experts * 3 * perWeight)
	if expert != perLayer*moe {
		t.Fatalf("expert bytes = %d, want %d (%d layers x %d)", expert, perLayer*moe, moe, perLayer)
	}

	b, ok := m.ExpertSpillBudgetFor(replicated + expert)
	if !ok {
		t.Fatal("ExpertSpillBudgetFor: not sizeable on a loaded MoE model")
	}
	if b.MoELayers != moe || b.ExpertBytesPerLayer != perLayer || b.DeviceBaseBytes != replicated {
		t.Fatalf("budget = %+v, want MoELayers=%d perLayer=%d base=%d", b, moe, perLayer, replicated)
	}

	// (a) A device that fits everything: nothing spills, and the ring is sized to the whole expert
	// set — a bound that is never reached, which is the honest answer for a model that fits.
	fits, ok, err := m.ResolveExpertSpillPlacement(replicated+expert, -1)
	if err != nil || !ok {
		t.Fatalf("ResolveExpertSpillPlacement(auto, generous): ok=%v err=%v", ok, err)
	}
	if fits.Fit.SpillLayers != 0 || !fits.Fit.Fits {
		t.Fatalf("generous budget: spill=%d fits=%v, want 0/true", fits.Fit.SpillLayers, fits.Fit.Fits)
	}
	if fits.RingBytes != expert {
		t.Fatalf("generous budget ring = %d, want the full expert bulk %d", fits.RingBytes, expert)
	}

	// (b) A device that holds the dense base plus TWO expert layers: auto spills the other two, and
	// the ring gets exactly the leftover headroom — the bytes the kept layers must share.
	tight := replicated + 2*perLayer
	p, ok, err := m.ResolveExpertSpillPlacement(tight, -1)
	if err != nil || !ok {
		t.Fatalf("ResolveExpertSpillPlacement(auto, tight): ok=%v err=%v", ok, err)
	}
	if p.Fit.SpillLayers != 2 || !p.Fit.Fits {
		t.Fatalf("tight budget: spill=%d fits=%v, want 2/true", p.Fit.SpillLayers, p.Fit.Fits)
	}
	if p.RingBytes != 2*perLayer {
		t.Fatalf("tight budget ring = %d, want the leftover headroom %d", p.RingBytes, 2*perLayer)
	}
	if p.Fit.DeviceResidentBytes > tight {
		t.Fatalf("planned device residency %d exceeds the budget %d", p.Fit.DeviceResidentBytes, tight)
	}

	// Applying it wires all three coupled knobs at once — this is what makes R0 reachable.
	s := &Session{M: m}
	s.ApplyExpertSpillPlacement(p)
	if !s.CPUOffloadExperts || s.ExpertSpillLayers != 2 || s.ExpertRingBytes != p.RingBytes {
		t.Fatalf("applied plan: offload=%v spill=%d ring=%d, want true/2/%d", s.CPUOffloadExperts, s.ExpertSpillLayers, s.ExpertRingBytes, p.RingBytes)
	}
	// A spill of zero must NOT switch the split on: an operator who sized a ring did not ask for
	// every expert GEMM to move to the host.
	sz := &Session{M: m}
	sz.ApplyExpertSpillPlacement(fits)
	if sz.CPUOffloadExperts {
		t.Fatal("a zero-layer spill turned the CPU-offload split on")
	}
	if sz.ExpertRingBytes != fits.RingBytes {
		t.Fatalf("zero-layer plan ring = %d, want %d", sz.ExpertRingBytes, fits.RingBytes)
	}

	// (c) A device too small even for the dense base: the maximal spill, Fits=false (the caller
	// refuses), and NO ring — a nonsensically small ring would thrash rather than help.
	starved, ok, err := m.ResolveExpertSpillPlacement(replicated/2, -1)
	if err != nil || !ok {
		t.Fatalf("ResolveExpertSpillPlacement(auto, starved): ok=%v err=%v", ok, err)
	}
	if starved.Fit.SpillLayers != moe || starved.Fit.Fits {
		t.Fatalf("starved budget: spill=%d fits=%v, want %d/false", starved.Fit.SpillLayers, starved.Fit.Fits, moe)
	}
	if starved.RingBytes != 0 {
		t.Fatalf("starved budget ring = %d, want 0", starved.RingBytes)
	}

	// A dense model has no gradeable expert residency: ok=false, and the caller leaves placement alone.
	if _, ok, err := (&Model{q4kw: map[string]*q4kTensor{"lm_head.weight": {raw: make([]byte, 32)}}}).ResolveExpertSpillPlacement(1<<30, -1); ok || err != nil {
		t.Fatalf("dense model: ok=%v err=%v, want false/nil", ok, err)
	}
}

// TestPlanExpertSpillRefusesOutOfRangeN pins the fail-closed operator path: an explicit N past the
// MoE layer count is a typo, and a typo must be refused with the typed range error rather than
// clamped into a residency nobody asked for.
func TestPlanExpertSpillRefusesOutOfRangeN(t *testing.T) {
	m := expertSpillTestModel([]int{0, 1, 2}, nil, 2, 256)
	if _, _, err := m.ResolveExpertSpillPlacement(1<<30, 4); err == nil {
		t.Fatal("ResolveExpertSpillPlacement(N=4) on a 3-MoE-layer model returned no error")
	} else {
		var rangeErr *ExpertSpillRangeError
		if !errors.As(err, &rangeErr) {
			t.Fatalf("ResolveExpertSpillPlacement(N=4) error = %v, want *ExpertSpillRangeError", err)
		}
		if rangeErr.N != 4 || rangeErr.Layers != 3 {
			t.Fatalf("range error = %+v, want N=4 Layers=3", rangeErr)
		}
	}
	// An in-range explicit N is honored exactly — the operator's number, not the auto-fit's.
	p, ok, err := m.ResolveExpertSpillPlacement(1<<30, 1)
	if err != nil || !ok {
		t.Fatalf("ResolveExpertSpillPlacement(N=1): ok=%v err=%v", ok, err)
	}
	if p.Fit.SpillLayers != 1 {
		t.Fatalf("explicit N=1 resolved to spill=%d", p.Fit.SpillLayers)
	}
}
