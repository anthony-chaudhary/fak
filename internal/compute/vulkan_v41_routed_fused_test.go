//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// Parity contract for the fused routed FFN port (fak-private#3341):
//
//   - every deferred status verdict is EXACT: the fused path publishes the same
//     fault type with the same fields as the five-dispatch id path;
//   - the gate/up GEMV is BF16-ULP BOUNDED against the id GEMV. The fused
//     gate/up kernel is one compiled program producing both projections, while
//     the two id GEMV dispatches compile to different FP32 FMA contraction; the
//     raw pre-rounding values differ by ~1e-6 relative and a few BF16 rounding
//     boundaries turn that into at most a few BF16 ULP. Worst measured relative
//     mid deviation ~1.3e-2;
//   - the down+publication+sum fold is BITWISE equal to the id path given the
//     same activation. A third arm runs the id down GEMV and routed
//     publication+sum over the FUSED arm's `mid`, so both arms consume the same
//     bits and the fold must reproduce the fused two-dispatch arm's `sum`
//     bit-for-bit and its publication verdicts exactly. Consuming one shared
//     mid removes the cancellation sensitivity a sum of six BF16-rounded
//     near-cancelling rows would otherwise amplify;
//   - the end-to-end model logits (TestV41RoutedLayerBatchVulkan, whose batch_id
//     arm engages the fused path) are BITWISE equal between the fused and id
//     arms.
//
// The constants below encode that contract. They are calibrated on the real
// device (strix3, RADV STRIX_HALO) against the exact witness geometry, which
// measured a worst |fused-id| relative deviation of 2.78e-2 and a worst BF16
// distance of 5 representable values at hidden=4096/inter=2048/picks=6.
const (
	// v41FusedMidAbsTol/v41FusedMidRelTol bound |fused-id| for mid rows. The
	// relative term dominates; 5e-2 is under 2x the measured worst case
	// (2.78e-2 at hidden=4096/inter=2048/picks=6) and still far below the
	// deviation a real indexing or activation defect produces.
	v41FusedMidAbsTol = 2e-3
	v41FusedMidRelTol = 5e-2
	// v41FusedMidMaxBF16ULP is the sharper gate on the same rows: every lane
	// whose BF16 rounding differs must be within this many representable BF16
	// values of the id lane. It states the ticket's "bit or ULP parity"
	// directly instead of only bounding the relative error. 8 is under 2x the
	// measured worst 5 BF16 values at hidden=4096/inter=2048/picks=6.
	v41FusedMidMaxBF16ULP = 8
)

// v41RoutedFusedCase exercises the degrees of freedom the fused two-dispatch
// layer exposes beyond the P3 geometry: a non-identity routed-sum order and a
// per-pick activation row. bitwise marks geometries measured to agree with the
// id path bit-for-bit, which the witness then asserts explicitly instead of
// accepting the bounded tolerance.
type v41RoutedFusedCase struct {
	name                   string
	hidden, inter, picks   int
	permute                bool
	nonIdentityOrder       bool
	perPickX               bool
	bitwise                bool
	poisonGate, poisonDown int // pick index, or -1
}

// v41RoutedFusedArm is what the two fused dispatches publish: every pick's mid
// row (the id path's activation row), the summed row over hidden, and the
// interleaved status verdicts (per pick: SwiGLU then publication).
type v41RoutedFusedArm struct {
	mid      [][]float32
	sum      []float32
	verdicts []error
	// idsTensor and midTensor stay live so a third arm can run the id path's
	// down+publication+sum over the exact bits this arm's fused gate/up produced.
	// The caller owns and frees them.
	idsTensor Tensor
	midTensor Tensor
}

// v41RoutedFusedBatched records one routed layer as the two fused dispatches
// (V41RoutedIDs -> V41RoutedGateUpSwiGLUDeferred -> V41RoutedDownSumDeferred)
// and reads both stages' records behind one fence. The slot table and ids image
// are built exactly as v41RoutedIDBatched builds them, so a pick-vs-slot
// indexing mistake in the fused kernel diverges here. The ids image and the mid
// rows are NOT freed: the returned arm holds them so the caller can replay the
// id down+sum fold over the same mid bits.
func v41RoutedFusedBatched(t *testing.T, v *vulkanBackend, gate, up, down []Tensor, x Tensor, weights []float32, order, perm []int, limit float32) v41RoutedFusedArm {
	t.Helper()
	picks := len(gate)
	table := func(ws []Tensor) []Tensor {
		out := make([]Tensor, picks)
		for s, p := range perm {
			out[s] = ws[p]
		}
		return out
	}
	slots := make([]int, picks)
	for s, p := range perm {
		slots[p] = s
	}
	if !v.BeginV41StepBatch() {
		t.Fatal("fused arm could not open a step batch")
	}
	before := v.VulkanDebugDispatchProfileSnapshot()
	ids, err := v.V41RoutedIDs(slots, weights, order)
	if err != nil {
		t.Fatal(err)
	}
	mid, swStatus, err := v.V41RoutedGateUpSwiGLUDeferred(table(gate), table(up), ids, x, picks, limit)
	if err != nil {
		t.Fatal(err)
	}
	sum, pubStatus, err := v.V41RoutedDownSumDeferred(table(down), mid, ids, picks, V41BF16PublishRoutedDown)
	if err != nil {
		t.Fatal(err)
	}
	// The fused layer must expose exactly the id path's status shape: one
	// SwiGLU record per pick of inter lanes, one publication record per pick of
	// hidden lanes at the routed-down site.
	if inter := mid.Shape[0] / picks; swStatus.Kind != V41DeferredRoutedSwiGLU || swStatus.Count != inter || swStatus.Records != picks {
		t.Errorf("fused gate/up status %+v, want kind %d count %d records %d", swStatus, V41DeferredRoutedSwiGLU, inter, picks)
	}
	if hidden := sum.Shape[0]; pubStatus.Kind != V41DeferredBF16Publish || pubStatus.Count != hidden || pubStatus.Site != V41BF16PublishRoutedDown || pubStatus.Records != picks {
		t.Errorf("fused down status %+v, want kind %d count %d site %d records %d", pubStatus, V41DeferredBF16Publish, hidden, V41BF16PublishRoutedDown, picks)
	}
	recorded := v.VulkanDebugDispatchProfileSnapshot()
	verdicts, values, err := v.V41ReadDeferred([]V41DeferredStatus{swStatus, pubStatus}, []Tensor{mid, sum})
	if err != nil {
		t.Fatal(err)
	}
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	if profiled := recorded.ComputeDispatches - before.ComputeDispatches; os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" && profiled != 2 {
		t.Errorf("fused layer recorded %d compute dispatches, want 2", profiled)
	}
	v.Free(sum)
	if len(verdicts) != 2*picks {
		t.Fatalf("fused layer returned %d verdicts, want one per pick per checked stage (%d)", len(verdicts), 2*picks)
	}
	if len(values) != 2 {
		t.Fatalf("fused layer read %d rows, want mid and sum", len(values))
	}
	split := func(row []float32) [][]float32 {
		n := len(row) / picks
		out := make([][]float32, picks)
		for p := range out {
			out[p] = row[p*n : (p+1)*n]
		}
		return out
	}
	arm := v41RoutedFusedArm{mid: split(values[0]), sum: values[1], verdicts: make([]error, 2*picks), idsTensor: ids, midTensor: mid}
	for p := 0; p < picks; p++ {
		arm.verdicts[2*p], arm.verdicts[2*p+1] = verdicts[p], verdicts[picks+p]
	}
	return arm
}

// v41RoutedSlotTable rebuilds the bound slot table from a permutation, exactly
// as the fused and id arms do, so the third arm addresses the same pages.
func v41RoutedSlotTable(ws []Tensor, perm []int) []Tensor {
	out := make([]Tensor, len(ws))
	for s, p := range perm {
		out[s] = ws[p]
	}
	return out
}

// v41RoutedFusedDownArm is the third arm's result: the id path's
// down+publication+sum fold run over a supplied activation, plus its per-pick
// publication verdicts.
type v41RoutedFusedDownArm struct {
	sum      []float32
	verdicts []error
}

// v41RoutedFusedDownSumArm replays the id path's down GEMV, routed publication
// and ordered sum over an activation the caller already holds (the fused arm's
// mid). Because both the fused down+sum kernel and this arm consume the SAME
// mid bits, their sums must agree bitwise and their publication verdicts must be
// identical: that is the exact witness for the down+sum fold, with no
// cancellation-amplified tolerance.
func v41RoutedFusedDownSumArm(t *testing.T, v *vulkanBackend, downs []Tensor, ids Tensor, picks int, mid Tensor) v41RoutedFusedDownArm {
	t.Helper()
	if !v.BeginV41StepBatch() {
		t.Fatal("id down+sum arm could not open a step batch")
	}
	d, err := v.V41MatVecID(downs, ids, picks, mid)
	if err != nil {
		t.Fatal(err)
	}
	sum, status, err := v.V41RoutedPublishSumDeferred(d, ids, picks, V41BF16PublishRoutedDown)
	if err != nil {
		t.Fatal(err)
	}
	verdicts, values, err := v.V41ReadDeferred([]V41DeferredStatus{status}, []Tensor{sum})
	if err != nil {
		t.Fatal(err)
	}
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	v.Free(d)
	v.Free(sum)
	if len(values) != 1 || len(verdicts) != picks {
		t.Fatalf("id down+sum arm read %d rows and %d verdicts, want 1 row and %d verdicts", len(values), len(verdicts), picks)
	}
	return v41RoutedFusedDownArm{sum: values[0], verdicts: verdicts}
}

// v41RoutedFusedHostSum is the P3 host reference generalized to a per-pick
// activation: host Q2_K/Q3_K GEMMs, the host routed SwiGLU lane, BF16
// publication and the ordered F32 sum. xStride 0 means one shared row.
func v41RoutedFusedHostSum(gate, up, down []Tensor, x []float32, xStride int, weights []float32, order []int, limit float32) []float32 {
	ref := Default()
	pub := make([][]float32, len(gate))
	for p := range gate {
		xp := x
		if xStride != 0 {
			xp = x[p*xStride : (p+1)*xStride]
		}
		g, u := ref.Read(ref.MatMul(gate[p], NewF32(ref, []int{len(xp)}, xp))), ref.Read(ref.MatMul(up[p], NewF32(ref, []int{len(xp)}, xp)))
		act := make([]float32, len(g))
		for i := range g {
			pre, _ := v41RoutedSwiGLUHostLane(g[i], u[i], limit, weights[p])
			act[i] = v41RoutedTestBF16(pre)
		}
		d := ref.Read(ref.MatMul(down[p], NewF32(ref, []int{len(act)}, act)))
		pub[p] = make([]float32, len(d))
		for i, v := range d {
			pub[p][i] = v41RoutedTestBF16(v)
		}
	}
	sum := make([]float32, down[0].Shape[0])
	for _, p := range order {
		for i := range sum {
			sum[i] += pub[p][i]
		}
	}
	return sum
}

// bf16ULP is the distance between a and b in representable BF16 values. It
// rounds both operands to BF16 and compares their monotonic unsigned integer
// encodings (negatives inverted, positives sign-offset), so adjacent BF16 values
// differ by 1 regardless of sign or magnitude. Ops that agree after BF16
// rounding yield 0; the ticket's ULP-parity clause bounds the rest.
func bf16ULP(a, b float32) uint32 {
	key := func(x uint32) uint32 {
		if x&0x8000 != 0 {
			return ^x & 0xffff
		}
		return x | 0x8000
	}
	ka := key(math.Float32bits(v41RoutedTestBF16(a)) >> 16)
	kb := key(math.Float32bits(v41RoutedTestBF16(b)) >> 16)
	if ka > kb {
		return ka - kb
	}
	return kb - ka
}

// v41RoutedFusedBF16ULPEqual asserts that every lane whose BF16 rounding differs
// from want is within maxULP representable BF16 values of it, and returns the
// worst distance observed for the receipt. This is stronger than a relative
// tolerance: it fails on any lane that crosses more than maxULP rounding steps,
// independent of magnitude.
func v41RoutedFusedBF16ULPEqual(t *testing.T, what string, got, want []float32, maxULP uint32) uint32 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d, want %d", what, len(got), len(want))
	}
	var worst uint32
	for i := range want {
		if math.Float32bits(got[i]) == math.Float32bits(want[i]) {
			continue
		}
		d := bf16ULP(got[i], want[i])
		if d > maxULP {
			t.Fatalf("%s lane %d: fused=%#08x id=%#08x differ by %d BF16 ULP, want <= %d", what, i, math.Float32bits(got[i]), math.Float32bits(want[i]), d, maxULP)
		}
		if d > worst {
			worst = d
		}
	}
	return worst
}

// v41RoutedFusedMaxRelDiff reports the worst relative deviation without gating.
// The fused-vs-id routed sum is no longer a pass/fail gate (the exact witness is
// the third arm over one shared mid); this only feeds the receipt so a real
// divergence stays visible.
func v41RoutedFusedMaxRelDiff(got, want []float32) float64 {
	var maxRel float64
	for i := range want {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		rel := diff / math.Max(math.Abs(float64(want[i])), 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
	}
	return maxRel
}

// v41RoutedFusedBoundedEqual asserts |got-want| <= absTol + relTol*|want| for every
// lane of a BF16-published row, and returns the observed maximum relative
// deviation so the witness receipt carries the measured number. Unlike the id
// path's five dispatches, the fused gate/up program is one compiled program and
// contracts the two projections' FP32 FMAs differently, so its activation rows
// can cross a BF16 rounding boundary by a few ULP; the bound makes that honest
// and explicit while still failing on a real indexing or activation defect.
func v41RoutedFusedBoundedEqual(t *testing.T, what string, got, want []float32, absTol, relTol float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d, want %d", what, len(got), len(want))
	}
	var maxRel float64
	for i := range want {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		bound := absTol + relTol*math.Abs(float64(want[i]))
		if rel := diff / math.Max(math.Abs(float64(want[i])), 1e-30); rel > maxRel {
			maxRel = rel
		}
		if diff > bound {
			t.Fatalf("%s lane %d: fused=%#08x id=%#08x |diff|=%.6e exceeds tol (abs=%.3e rel=%.3e bound=%.6e)", what, i, math.Float32bits(got[i]), math.Float32bits(want[i]), diff, absTol, relTol, bound)
		}
	}
	return maxRel
}

// TestVulkanV41RoutedFusedMatchesID is the fused-port parity witness
// (fak-private#3341): one layer's routed experts as TWO dispatches
// (gate+up+clamp-SwiGLU+publication, then down+publication+sum) equal the
// five-dispatch id path on identity and permuted slot maps, shared and per-pick
// activations, identity and reversed sum orders. Status verdicts are asserted
// EXACT. The gate/up mid rows are BF16-ULP bounded (the fused gate/up program's
// FP32 FMA contraction differs from the two id GEMV dispatches; see the
// parity-contract constants) AND held to the direct BF16-ULP bound. The
// down+sum fold is proven EXACT by a third arm that replays the id path's
// down+publication+sum over the fused arm's own mid, so the two sums and their
// publication verdicts must agree bit-for-bit; the fused-vs-id sum comparison is
// only a reported diagnostic. The geometries measured to agree bit-for-bit
// carry bitwise=true and are held to the stronger bitwise assertion. A fault in
// one pick still surfaces in exactly that pick's record. The sum also tracks the
// host reference.
// fak-test:runtime integration est=10s lane=optin
func TestVulkanV41RoutedFusedMatchesID(t *testing.T) {
	v := vk(t)
	// The fused path is defined to equal the id path, so the witness is void
	// unless both the fused modules and the id reference kernels are present.
	if !v.V41RoutedFusedAvailable() || !v.V41RoutedIDAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 fused routed modules (registry V16) or the routed id modules the parity witness compares against")
		}
		t.Skip("V4.1 fused routed or routed id modules unavailable")
	}
	const limit = 10
	for _, c := range []v41RoutedFusedCase{
		{name: "v41_flash_geometry", hidden: 4096, inter: 2048, picks: 6, poisonGate: -1, poisonDown: -1},
		{name: "permuted_slots", hidden: 4096, inter: 2048, picks: 6, permute: true, poisonGate: -1, poisonDown: -1},
		{name: "small_tail_three_picks", hidden: 512, inter: 256, picks: 3, permute: true, bitwise: true, poisonGate: -1, poisonDown: -1},
		{name: "eight_picks", hidden: 1024, inter: 512, picks: 8, permute: true, poisonGate: -1, poisonDown: -1},
		{name: "activation_fault_pick_2", hidden: 512, inter: 256, picks: 6, poisonGate: 2, poisonDown: -1},
		{name: "publication_fault_pick_4", hidden: 512, inter: 256, picks: 6, permute: true, poisonGate: -1, poisonDown: 4},
		{name: "reversed_sum_order", hidden: 1024, inter: 512, picks: 4, permute: true, nonIdentityOrder: true, poisonGate: -1, poisonDown: -1},
		{name: "per_pick_activation", hidden: 1024, inter: 512, picks: 4, permute: true, perPickX: true, poisonGate: -1, poisonDown: -1},
		{name: "single_pick", hidden: 512, inter: 256, picks: 1, bitwise: true, poisonGate: -1, poisonDown: -1},
		{name: "per_pick_reversed_permuted", hidden: 2048, inter: 512, picks: 5, permute: true, nonIdentityOrder: true, perPickX: true, poisonGate: -1, poisonDown: -1},
		{name: "fused_down_fault_reversed_per_pick", hidden: 2048, inter: 512, picks: 6, permute: true, nonIdentityOrder: true, perPickX: true, poisonGate: -1, poisonDown: 5},
	} {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(3341 + c.picks*c.hidden)))
			var hostGate, hostUp, hostDown, gate, up, down []Tensor
			weights := make([]float32, c.picks)
			experts := make([]int, c.picks)
			for p := 0; p < c.picks; p++ {
				seed := int64(1000*p + c.hidden)
				hostGate = append(hostGate, NewQ2K(Default(), []int{c.inter, c.hidden}, v41RoutedIDQ2K(seed, c.inter, c.hidden, p == c.poisonGate)))
				hostUp = append(hostUp, NewQ2K(Default(), []int{c.inter, c.hidden}, v41RoutedIDQ2K(seed+1, c.inter, c.hidden, false)))
				hostDown = append(hostDown, NewQ3K(Default(), []int{c.hidden, c.inter}, v41RoutedIDQ3K(seed+2, c.hidden, c.inter, p == c.poisonDown)))
				gate = append(gate, v.Upload(hostGate[p], Q2_K))
				up = append(up, v.Upload(hostUp[p], Q2_K))
				down = append(down, v.Upload(hostDown[p], Q3_K))
				weights[p] = float32(0.05 + 0.3*rng.Float64())
				experts[p] = rng.Intn(256)
			}
			defer func() {
				for _, w := range append(append(append([]Tensor(nil), gate...), up...), down...) {
					v.Free(w)
				}
			}()
			order := make([]int, c.picks)
			for i := range order {
				order[i] = i
			}
			if c.nonIdentityOrder {
				for i := range order {
					order[i] = c.picks - 1 - i
				}
			} else {
				sort.SliceStable(order, func(i, j int) bool { return experts[order[i]] < experts[order[j]] })
			}
			perm := make([]int, c.picks)
			for i := range perm {
				perm[i] = i
			}
			if c.permute {
				rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
			}
			hostX := make([]float32, c.hidden)
			xStride := 0
			if c.perPickX {
				hostX = make([]float32, c.picks*c.hidden)
				xStride = c.hidden
			}
			for i := range hostX {
				hostX[i] = v41RoutedTestBF16(float32(rng.NormFloat64() * 0.5))
			}
			x := v.Upload(NewF32(Default(), []int{len(hostX)}, hostX), F32)
			defer v.Free(x)

			want := v41RoutedIDBatched(t, v, gate, up, down, x, weights, order, perm, limit)
			got := v41RoutedFusedBatched(t, v, gate, up, down, x, weights, order, perm, limit)
			// The fused arm hands back its ids image and mid so the third arm can
			// replay the id down+sum fold over the same activation bits.
			defer v.Free(got.idsTensor)
			defer v.Free(got.midTensor)
			var maxRelMid, maxRelSum float64
			var maxMidULP uint32
			for p := 0; p < c.picks; p++ {
				v41RoutedIDSameVerdict(t, fmt.Sprintf("pick %d SwiGLU", p), got.verdicts[2*p], want.verdicts[2*p])
				v41RoutedIDSameVerdict(t, fmt.Sprintf("pick %d publication", p), got.verdicts[2*p+1], want.verdicts[2*p+1])
				// mid[p] is the fused path's activation row, which the id path
				// stores as its activation output row. A poisoned gate corrupts
				// only its own row; a poisoned down operand leaves every mid row
				// (the gate/up dispatch) untouched.
				if p == c.poisonGate {
					continue
				}
				// The strong property: geometries the device reproduces exactly are
				// asserted bitwise; the rest carry the documented BF16-ULP bound.
				if c.bitwise {
					v41RoutedIDBitsEqual(t, fmt.Sprintf("pick %d mid", p), got.mid[p], want.act[p])
					continue
				}
				if rel := v41RoutedFusedBoundedEqual(t, fmt.Sprintf("pick %d mid", p), got.mid[p], want.act[p], v41FusedMidAbsTol, v41FusedMidRelTol); rel > maxRelMid {
					maxRelMid = rel
				}
				if ulp := v41RoutedFusedBF16ULPEqual(t, fmt.Sprintf("pick %d mid (BF16 ULP)", p), got.mid[p], want.act[p], v41FusedMidMaxBF16ULP); ulp > maxMidULP {
					maxMidULP = ulp
				}
			}
			switch {
			case c.poisonGate >= 0:
				var fault *V41RoutedSwiGLUFault
				if !errors.As(got.verdicts[2*c.poisonGate], &fault) {
					t.Fatalf("poisoned gate of pick %d raised no activation fault", c.poisonGate)
				}
				for p := 0; p < c.picks; p++ {
					if p == c.poisonGate {
						continue
					}
					var other *V41RoutedSwiGLUFault
					if errors.As(got.verdicts[2*p], &other) {
						t.Fatalf("unpoisoned pick %d activation also faulted: %v", p, other)
					}
					if got.verdicts[2*p+1] != nil {
						t.Fatalf("unpoisoned pick %d publication faulted: %v", p, got.verdicts[2*p+1])
					}
				}
				return
			case c.poisonDown >= 0:
				var numeric *V41BF16CheckedArithmeticError
				if !errors.As(got.verdicts[2*c.poisonDown+1], &numeric) {
					t.Fatalf("poisoned down of pick %d raised no publication fault", c.poisonDown)
				}
				for p := 0; p < c.picks; p++ {
					if p == c.poisonDown {
						continue
					}
					var other *V41BF16CheckedArithmeticError
					if errors.As(got.verdicts[2*p+1], &other) {
						t.Fatalf("unpoisoned pick %d publication also faulted: %v", p, other)
					}
					if got.verdicts[2*p] != nil {
						t.Fatalf("unpoisoned pick %d activation faulted: %v", p, got.verdicts[2*p])
					}
				}
				return
			}
			// The exact half of the fusion claim: replay the id path's
			// down+publication+sum over the fused arm's own mid. Both arms consume
			// the same bits, so the sum and every publication verdict must match
			// exactly, independent of the gate/up contraction difference.
			fusedDown := v41RoutedFusedDownSumArm(t, v, v41RoutedSlotTable(down, perm), got.idsTensor, c.picks, got.midTensor)
			v41RoutedIDBitsEqual(t, "routed sum (id down+sum over fused mid)", got.sum, fusedDown.sum)
			for p := 0; p < c.picks; p++ {
				v41RoutedIDSameVerdict(t, fmt.Sprintf("pick %d fused publication", p), got.verdicts[2*p+1], fusedDown.verdicts[p])
			}
			// The fused-vs-id sum is only a reported diagnostic now: the sum of six
			// BF16-rounded rows amplifies the gate/up ULP difference on
			// near-cancelling lanes, which is not a down+sum defect.
			if c.bitwise {
				v41RoutedIDBitsEqual(t, "routed sum", got.sum, want.sum)
			} else {
				maxRelSum = v41RoutedFusedMaxRelDiff(got.sum, want.sum)
			}
			host := v41RoutedFusedHostSum(hostGate, hostUp, hostDown, hostX, xStride, weights, order, limit)
			if cs := cosine(got.sum, host); cs < 0.9999 {
				t.Fatalf("routed sum cosine vs host reference %.8f < 0.9999", cs)
			}
			if math.IsNaN(float64(got.sum[0])) {
				t.Fatal("routed sum is NaN")
			}
			t.Logf("%s picks=%d hidden=%d inter=%d permuted=%t per-pick-x=%t reversed-order=%t bitwise=%t: 2 fused dispatches vs 5 id dispatches; maxRelMid=%.4e maxMidULP=%d maxRelSum(diag)=%.4e", c.name, c.picks, c.hidden, c.inter, c.permute, c.perPickX, c.nonIdentityOrder, c.bitwise, maxRelMid, maxMidULP, maxRelSum)
		})
	}
}
