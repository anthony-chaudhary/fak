//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// v41RoutedIDQ2K is a Q2_K [out,in] fixture; poison makes block 0's d +Inf.
func v41RoutedIDQ2K(seed int64, out, in int, poison bool) []byte {
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < out*(in/q2kSuper); b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], float32(0.01+rng.Float64()*0.05))
		binaryPutFloat16(blk[82:84], float32(0.005+rng.Float64()*0.02))
	}
	if poison {
		binary.LittleEndian.PutUint16(raw[80:82], 0x7c00)
	}
	return raw
}

func v41RoutedIDQ3K(seed int64, out, in int, poison bool) []byte {
	raw := q3kStrixFixture(out, in)
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < len(raw); i += q3kSuperBlock {
		for j := 0; j < 96; j++ {
			raw[i+j] ^= byte(rng.Intn(256))
		}
	}
	if poison {
		binary.LittleEndian.PutUint16(raw[q3kSuperBlock-2:], 0x7c00)
	}
	return raw
}

type v41RoutedIDCase struct {
	name                   string
	hidden, inter, picks   int
	permute                bool
	poisonGate, poisonDown int // pick index, or -1
}

type v41RoutedIDArm struct {
	gate, up, act, down [][]float32
	sum                 []float32
	verdicts            []error // per pick: SwiGLU then publication
}

// v41RoutedIDPerPick records the per-pick path exactly as the batched routed
// layer did before the id kernels: two MatMuls, the deferred SwiGLU, the down
// MatMul, the deferred publication, then AddInPlace in sum order.
func v41RoutedIDPerPick(t *testing.T, v *vulkanBackend, gate, up, down []Tensor, x Tensor, weights []float32, order []int, limit float32) v41RoutedIDArm {
	t.Helper()
	picks := len(gate)
	if !v.BeginV41StepBatch() {
		t.Fatal("per-pick arm could not open a step batch")
	}
	var owned []Tensor
	var statuses []V41DeferredStatus
	var rows []Tensor
	published := make([]Tensor, picks)
	for p := 0; p < picks; p++ {
		g, u := v.MatMul(gate[p], x), v.MatMul(up[p], x)
		a, st, err := v.V41RoutedSwiGLUDeferred(g, u, limit, weights[p])
		if err != nil {
			t.Fatal(err)
		}
		statuses = append(statuses, st)
		d := v.MatMul(down[p], a)
		row, pst, err := v.V41BF16PublishDeferred(d, V41BF16PublishRoutedDown)
		if err != nil {
			t.Fatal(err)
		}
		statuses = append(statuses, pst)
		published[p] = row
		owned = append(owned, g, u, a, d, row)
		rows = append(rows, g, u, a, d)
	}
	acc := v.Upload(NewF32(Default(), []int{x.Shape[0]}, make([]float32, x.Shape[0])), F32)
	owned = append(owned, acc)
	for _, p := range order {
		v.AddInPlace(acc, published[p])
	}
	rows = append(rows, acc)
	verdicts, values, err := v.V41ReadDeferred(statuses, rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	for _, o := range owned {
		v.Free(o)
	}
	arm := v41RoutedIDArm{sum: values[len(values)-1], verdicts: make([]error, 2*picks)}
	for p := 0; p < picks; p++ {
		arm.gate = append(arm.gate, values[4*p])
		arm.up = append(arm.up, values[4*p+1])
		arm.act = append(arm.act, values[4*p+2])
		arm.down = append(arm.down, values[4*p+3])
		arm.verdicts[2*p], arm.verdicts[2*p+1] = verdicts[2*p], verdicts[2*p+1]
	}
	return arm
}

// v41RoutedIDBatched records the same layer as five id dispatches.
func v41RoutedIDBatched(t *testing.T, v *vulkanBackend, gate, up, down []Tensor, x Tensor, weights []float32, order, perm []int, limit float32) v41RoutedIDArm {
	t.Helper()
	picks := len(gate)
	// The bound slot table is perm order; slots maps each pick to its page.
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
		t.Fatal("id arm could not open a step batch")
	}
	before := v.VulkanDebugDispatchProfileSnapshot()
	ids, err := v.V41RoutedIDs(slots, weights, order)
	if err != nil {
		t.Fatal(err)
	}
	g, err := v.V41MatVecID(table(gate), ids, picks, x)
	if err != nil {
		t.Fatal(err)
	}
	u, err := v.V41MatVecID(table(up), ids, picks, x)
	if err != nil {
		t.Fatal(err)
	}
	a, swStatus, err := v.V41RoutedSwiGLUIDDeferred(g, u, ids, picks, limit)
	if err != nil {
		t.Fatal(err)
	}
	d, err := v.V41MatVecID(table(down), ids, picks, a)
	if err != nil {
		t.Fatal(err)
	}
	sum, pubStatus, err := v.V41RoutedPublishSumDeferred(d, ids, picks, V41BF16PublishRoutedDown)
	if err != nil {
		t.Fatal(err)
	}
	recorded := v.VulkanDebugDispatchProfileSnapshot()
	verdicts, values, err := v.V41ReadDeferred([]V41DeferredStatus{swStatus, pubStatus}, []Tensor{g, u, a, d, sum})
	if err != nil {
		t.Fatal(err)
	}
	if err := v.EndV41StepBatch(); err != nil {
		t.Fatal(err)
	}
	if profiled := recorded.ComputeDispatches - before.ComputeDispatches; os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" && profiled != 5 {
		t.Errorf("id layer recorded %d compute dispatches, want 5", profiled)
	}
	for _, o := range []Tensor{ids, g, u, a, d, sum} {
		v.Free(o)
	}
	if len(verdicts) != 2*picks {
		t.Fatalf("id layer returned %d verdicts, want one per pick per checked stage (%d)", len(verdicts), 2*picks)
	}
	split := func(row []float32) [][]float32 {
		n := len(row) / picks
		out := make([][]float32, picks)
		for p := range out {
			out[p] = row[p*n : (p+1)*n]
		}
		return out
	}
	arm := v41RoutedIDArm{gate: split(values[0]), up: split(values[1]), act: split(values[2]), down: split(values[3]), sum: values[4], verdicts: make([]error, 2*picks)}
	for p := 0; p < picks; p++ {
		arm.verdicts[2*p], arm.verdicts[2*p+1] = verdicts[p], verdicts[picks+p]
	}
	return arm
}

func v41RoutedIDBitsEqual(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s lane %d: id=%#08x per-pick=%#08x", what, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func v41RoutedIDSameVerdict(t *testing.T, what string, got, want error) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Fatalf("%s: id verdict %v, per-pick %v", what, got, want)
	}
	if want == nil {
		return
	}
	var gf, wf *V41RoutedSwiGLUFault
	if errors.As(want, &wf) {
		if !errors.As(got, &gf) || *gf != *wf {
			t.Fatalf("%s: id fault %v, per-pick %v", what, got, want)
		}
		return
	}
	var gn, wn *V41BF16CheckedArithmeticError
	if errors.As(want, &wn) {
		if !errors.As(got, &gn) || *gn != *wn {
			t.Fatalf("%s: id fault %v, per-pick %v", what, got, want)
		}
		return
	}
	t.Fatalf("%s: per-pick verdict %v is not a numeric fault", what, want)
}

// v41RoutedIDHostSum is the CPU reference: host Q2_K/Q3_K GEMMs, the host
// routed SwiGLU lane, BF16 publication and the ordered F32 sum.
func v41RoutedIDHostSum(gate, up, down []Tensor, x []float32, weights []float32, order []int, limit float32) []float32 {
	ref := Default()
	xt := NewF32(ref, []int{len(x)}, x)
	pub := make([][]float32, len(gate))
	for p := range gate {
		g, u := ref.Read(ref.MatMul(gate[p], xt)), ref.Read(ref.MatMul(up[p], xt))
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
	sum := make([]float32, len(x))
	for _, p := range order {
		for i := range sum {
			sum[i] += pub[p][i]
		}
	}
	return sum
}

// TestVulkanV41RoutedIDMatchesPerPick is the P3 port witness (fak#13820): one
// layer's routed experts as five id dispatches (gate, up, SwiGLU, down,
// publish+sum) are bitwise the per-pick kernels' rows, records and sum, with
// identity and permuted slot maps, and a fault in one pick surfaces in that
// pick's record only. The sum also tracks the host reference.
// fak-test:runtime integration est=10s lane=optin
func TestVulkanV41RoutedIDMatchesPerPick(t *testing.T) {
	v := vk(t)
	if !v.V41RoutedIDAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 routed id modules (registry V15) or binary32 float controls")
		}
		t.Skip("V4.1 routed id modules unavailable")
	}
	const limit = 10
	for _, c := range []v41RoutedIDCase{
		{name: "v41_flash_geometry", hidden: 4096, inter: 2048, picks: 6, poisonGate: -1, poisonDown: -1},
		{name: "permuted_slots", hidden: 4096, inter: 2048, picks: 6, permute: true, poisonGate: -1, poisonDown: -1},
		{name: "small_tail_three_picks", hidden: 512, inter: 256, picks: 3, permute: true, poisonGate: -1, poisonDown: -1},
		{name: "eight_picks", hidden: 1024, inter: 512, picks: 8, permute: true, poisonGate: -1, poisonDown: -1},
		{name: "activation_fault_pick_2", hidden: 512, inter: 256, picks: 6, poisonGate: 2, poisonDown: -1},
		{name: "publication_fault_pick_4", hidden: 512, inter: 256, picks: 6, permute: true, poisonGate: -1, poisonDown: 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(13820 + c.picks*c.hidden)))
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
			sort.SliceStable(order, func(i, j int) bool { return experts[order[i]] < experts[order[j]] })
			perm := make([]int, c.picks)
			for i := range perm {
				perm[i] = i
			}
			if c.permute {
				rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
			}
			hostX := make([]float32, c.hidden)
			for i := range hostX {
				hostX[i] = v41RoutedTestBF16(float32(rng.NormFloat64() * 0.5))
			}
			x := v.Upload(NewF32(Default(), []int{c.hidden}, hostX), F32)
			defer v.Free(x)

			want := v41RoutedIDPerPick(t, v, gate, up, down, x, weights, order, limit)
			got := v41RoutedIDBatched(t, v, gate, up, down, x, weights, order, perm, limit)
			faulted := c.poisonGate >= 0 || c.poisonDown >= 0
			for p := 0; p < c.picks; p++ {
				v41RoutedIDSameVerdict(t, fmt.Sprintf("pick %d SwiGLU", p), got.verdicts[2*p], want.verdicts[2*p])
				v41RoutedIDSameVerdict(t, fmt.Sprintf("pick %d publication", p), got.verdicts[2*p+1], want.verdicts[2*p+1])
				if faulted {
					continue
				}
				v41RoutedIDBitsEqual(t, fmt.Sprintf("pick %d gate", p), got.gate[p], want.gate[p])
				v41RoutedIDBitsEqual(t, fmt.Sprintf("pick %d up", p), got.up[p], want.up[p])
				v41RoutedIDBitsEqual(t, fmt.Sprintf("pick %d activation", p), got.act[p], want.act[p])
				v41RoutedIDBitsEqual(t, fmt.Sprintf("pick %d down", p), got.down[p], want.down[p])
			}
			switch {
			case c.poisonGate >= 0:
				var fault *V41RoutedSwiGLUFault
				if !errors.As(got.verdicts[2*c.poisonGate], &fault) {
					t.Fatalf("poisoned gate of pick %d raised no activation fault", c.poisonGate)
				}
				return
			case c.poisonDown >= 0:
				var numeric *V41BF16CheckedArithmeticError
				if !errors.As(got.verdicts[2*c.poisonDown+1], &numeric) {
					t.Fatalf("poisoned down of pick %d raised no publication fault", c.poisonDown)
				}
				return
			}
			v41RoutedIDBitsEqual(t, "routed sum", got.sum, want.sum)
			host := v41RoutedIDHostSum(hostGate, hostUp, hostDown, hostX, weights, order, limit)
			if cs := cosine(got.sum, host); cs < 0.9999 {
				t.Fatalf("routed sum cosine vs host reference %.8f < 0.9999", cs)
			}
			t.Logf("%s picks=%d hidden=%d inter=%d permuted=%t: 5 id dispatches bitwise equal to %d per-pick dispatches", c.name, c.picks, c.hidden, c.inter, c.permute, 6*c.picks)
		})
	}
}

// Geometry and id-image refusals happen before any native call.
// fak-test:runtime fast est=1s lane=optin
func TestVulkanV41RoutedIDRefusesGeometry(t *testing.T) {
	v := vk(t)
	if _, err := v.V41RoutedIDs([]int{0, 0}, []float32{1, 1}, []int{0, 0}); err == nil {
		t.Fatal("a non-permutation sum order was accepted")
	}
	if _, err := v.V41RoutedIDs([]int{0, 2}, []float32{1, 1}, []int{0, 1}); err == nil {
		t.Fatal("a slot outside the table was accepted")
	}
	if _, err := v.V41RoutedIDs(make([]int, V41RoutedIDMaxPicks+1), make([]float32, V41RoutedIDMaxPicks+1), make([]int, V41RoutedIDMaxPicks+1)); err == nil {
		t.Fatal("more picks than slots was accepted")
	}
	var geometry *BackendError
	if _, err := v.V41MatVecID(nil, Tensor{}, 1, Tensor{}); !errors.As(err, &geometry) || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("empty slot table: %v", err)
	}
}
