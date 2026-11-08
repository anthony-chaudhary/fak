package model

import (
	"math"
	"testing"
)

// Bounded ABSTAIN record — rank 3 of the V4.1 prefill perf goal (public fak
// #13820 family, "keep per-layer intermediates device-resident").
//
// Objective: remove ONE named mid-layer D2H readback from the DeepSeek-V4.1
// layer dataflow to device-resident form. Candidates named for this file:
// the indexer top-k readback, the mHC collapse readback, and the shared-expert
// readback in v41LayerSharedExpertAndPost.
//
// VERDICT: ABSTAIN. None of the three is safely removable by editing
// v41_forward_layer_moe.go alone, because this file is the ORCHESTRATOR: it
// contains no literal host-read-of-device-buffer statement. Every named
// readback executes inside a helper whose contract (host []float32 return) and
// whose consumer both live outside this file's allowed scope:
//
//   - Shared-expert panel readback (the dominant multi-token prefill path):
//     called at v41_forward_layer_moe.go:398 via
//     st.callbackOwner.v41ColdSharedExpertPanel(l)(panel.flat, seq); the D2H
//     happens inside (*Session).v41SharedExpertPanel
//     (v41_shared_expert_resident.go:243-264, copy(result[...], values)). Its
//     host result feeds v41SharedExpertAddWith (v41_expert_sum_device.go),
//     whose routed operand is itself host-resident (routedByToken from
//     v41LayerRoutedContractionPacked) and whose add RESULT must land on host
//     for the trace record and the mHC post-mix/publish. Making the shared row
//     device-resident requires BOTH a device-tensor panel return AND a batched
//     device shared-add — neither exists; adding either touches out-of-scope
//     files. The only device-resident seam reachable here (the consume callback
//     of v41SharedExpertChain, used by the decode path) serves one token at a
//     time and would regress the panel's single batched GEMM over all tokens.
//
//   - mHC collapse readback: called at v41_forward_layer_moe.go:97/102 via
//     in.resident.ffnInputOwned / ffnInput; the D2H happens inside
//     (*v41MHCResident).collapseRetained (v41_mhc_resident.go:386
//     compute.ReadV41MHCCollapse / :401 s.Backend.Read). The collapsed rows are
//     re-uploaded by scratch.ffnNorm.panel (v41_ffn_norm_binding.go) for the
//     learned RMSNorm, so they round-trip device->host->device. Keeping them
//     resident requires a device-tensor collapse return plus a device-input
//     norm — both in out-of-scope files.
//
//   - Indexer top-k readback: not present in this file at all (the indexer is
//     the attention half, v41_attention.go); its selected ids feed sparse-KV
//     scheduling and are out of scope here.
//
// EXACT NEXT CHECKABLE STEP (for the worker who owns the helpers): extend
// (*Session).v41SharedExpertPanel to expose the final published down-projection
// row as a device tensor (a ...PanelDevice variant or optional out-param) and
// add a batched device shared-add seam taking two device tensors, then point
// v41LayerSharedExpertAndPost's panel branch at them so the seq*H panel never
// crosses D2H; re-pin bit parity against the host reference asserted below.
//
// This test is the GPU-less regression guard for that future change: it pins,
// on a real full-geometry V4.1 model with no backend, that the layer epilogue
// adds the shared expert to the routed sum BEFORE the mHC post-mix and
// publishes the next streams bit-exactly (Float32bits) against an independent
// host recomputation. A device-resident port must keep these bytes identical.
// It models the host-driven source-contract style of
// v41_prefill_mhc_reuse_test.go / v41_shared_down_token_panel_test.go.
// fak-test:runtime fast est=300ms lane=default
func TestV41LayerMoEReadbackContract(t *testing.T) {
	m := v41RawFullFlattenedMHC(t)
	H := m.Cfg.HiddenSize
	const rows = 3
	checkBits := func(t *testing.T, got, want [][]float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("row count %d != %d", len(got), len(want))
		}
		for h := range want {
			if len(got[h]) != len(want[h]) {
				t.Fatalf("row %d width differs", h)
			}
			for d := range want[h] {
				if math.Float32bits(got[h][d]) != math.Float32bits(want[h][d]) {
					t.Fatalf("row %d coordinate %d differs: %g != %g", h, d, got[h][d], want[h][d])
				}
			}
		}
	}
	// Distinct from the sibling residual-reuse fixture: nontrivial cross-stream
	// coefficients and a non-divisible hidden width tail so any reordering of
	// the shared-add vs post-mix boundary changes the published bits.
	post := []float32{.6, -.5, .4, .7}
	comb := make([]float32, 16)
	for i := range comb {
		comb[i] = float32(i%5-2) / 11
	}
	in := v41LayerMoEInput{full: true, H: H, routeCfg: v41DefaultRouterConfig(),
		x: make([][]float32, rows), streams: make([][][]float32, rows),
		attnOut: make([][]float32, rows), hcByPos: make([]v41MHCMix, rows),
		scratch: &v41ProjScratch{mhcCarry: newV41MHCCarry(rows)}}
	for row := 0; row < rows; row++ {
		in.x[row] = make([]float32, H)
		in.attnOut[row] = make([]float32, H)
		for d := range in.x[row] {
			in.x[row][d] = float32((row*7+d)%11+1) / 19
			in.attnOut[row][d] = float32((d*3-row)%7-3) / 23
		}
		var err error
		in.streams[row], err = v41FullInitialStreams(in.x[row])
		if err != nil {
			t.Fatal(err)
		}
		in.hcByPos[row] = v41MHCMix{pre: []float32{.3, .3, .2, .2}, post: post, comb: comb}
	}
	panel, err := m.v41LayerFFNInputs(in)
	if err != nil {
		t.Fatalf("FFN panel: %v", err)
	}
	// A structured, NON-ZERO routed contribution per token: pinning that the
	// shared expert is added into THIS value before the mHC post-mix consumes
	// the sum. Zero routed (as in the sibling test) would leave the add
	// boundary under-exercised.
	routed := make([][]float32, rows)
	for row := range routed {
		routed[row] = make([]float32, H)
		for d := range routed[row] {
			routed[row][d] = float32((row*13+d*5)%17-8) / 29
		}
	}
	// The owned multi-row post-mix recycles panel.residual in place, so the
	// reference must be computed from a pre-epilogue snapshot of the carried
	// residual streams and mixes (exactly as the sibling residual-reuse test
	// orders its work).
	snapResidual := make([][][]float32, rows)
	snapMixes := make([]v41MHCMix, rows)
	for row := range snapResidual {
		snapResidual[row] = make([][]float32, len(panel.residual[row]))
		for h := range panel.residual[row] {
			snapResidual[row][h] = append([]float32(nil), panel.residual[row][h]...)
		}
		snapMixes[row] = panel.mixes[row]
	}
	// Independent host recomputation of the exact epilogue order: shared expert
	// projected from the normalized FFN panel row, added into the routed sum,
	// THEN the mHC post-mix over the carried residual streams.
	wantNext := make([][][]float32, rows)
	for row := range wantNext {
		shared, e := m.v41SharedExpertSwiGLU(0, panel.inputs[row], m.Cfg)
		if e != nil {
			t.Fatal(e)
		}
		moe, e := v41SharedExpertAdd(routed[row], shared, in.routeCfg)
		if e != nil {
			t.Fatal(e)
		}
		wantNext[row], e = v41MHCPostBF16On(nil, 0, moe, snapResidual[row], snapMixes[row])
		if e != nil {
			t.Fatal(e)
		}
	}
	originalCarryNext := in.scratch.mhcCarry.nextLayer
	err = m.v41LayerSharedExpertAndPost(in, panel, routed)
	if err != nil {
		t.Fatalf("epilogue: %v", err)
	}
	if in.scratch.mhcCarry.nextLayer != originalCarryNext+1 {
		t.Fatalf("carry advanced to %d, want %d", in.scratch.mhcCarry.nextLayer, originalCarryNext+1)
	}
	for row := range wantNext {
		checkBits(t, in.streams[row], wantNext[row])
	}
	// Negative control: post-mixing the routed sum WITHOUT first adding the
	// shared expert must differ, proving the add precedes the post-mix and the
	// readback-relevant value crossing the boundary is routed+shared, not routed.
	control := make([][][]float32, rows)
	for row := range control {
		next, e := v41MHCPostBF16On(nil, 0, routed[row], snapResidual[row], snapMixes[row])
		if e != nil {
			t.Fatal(e)
		}
		control[row] = next
	}
	for row := range control {
		same := true
		for h := range control[row] {
			for d := range control[row][h] {
				if math.Float32bits(in.streams[row][h][d]) == math.Float32bits(control[row][h][d]) {
					continue
				}
				same = false
			}
		}
		if same {
			t.Fatalf("row %d: shared expert had no effect on the published streams", row)
		}
	}
}
