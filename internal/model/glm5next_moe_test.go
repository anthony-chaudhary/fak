package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute/slotstream"
)

// cylceSource is a deterministic mock NVMe expert source: it fabricates a
// serialized expert record for any (layer, expert) without touching disk, so the
// streaming logic can be witnessed software-side ([SW-VERIFIED]).
type cycleSource struct {
	reads int
}

func (s *cycleSource) ExpertBytes(layer, expert int) ([]byte, error) {
	s.reads++
	buf := make([]byte, slotstream.ExpertChunkBytes)
	// Encode the first few f32 weights deterministically; the codec reads them.
	seed := float32(expert+1) * 0.01
	for i := 0; i < 3*16 && i*4+4 <= len(buf); i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(seed))
	}
	return buf, nil
}

// headerCodec decodes the leading bytes of a slot buffer into a small SwiGLU
// expert, treating (inDim, interDim) as the geometry to fill.
type headerCodec struct{}

func (headerCodec) Decode(slotData []byte, inDim, interDim int) (GLM5NextExpertWeight, error) {
	g := make([]float32, interDim*inDim)
	u := make([]float32, interDim*inDim)
	d := make([]float32, inDim*interDim)
	read := func(i int) float32 {
		off := i * 4
		if off+4 > len(slotData) {
			return 0
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(slotData[off:]))
	}
	for i := range g {
		v := read(i)
		g[i] = v
		u[i] = v
	}
	for i := range d {
		d[i] = read(i) * 0.5
	}
	return GLM5NextExpertWeight{WAct: g, WUp: u, WDown: d}, nil
}

// TestGLM5NextSlotStreamTopology asserts criterion 1: the plan serves the
// 288-expert routed topology from 54 resident slots per layer with LRU
// replacement, within the VRAM budget.
func TestGLM5NextSlotStreamTopology(t *testing.T) {
	plan, err := NewGLM5NextSlotStreamPlan(42)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.SlotsPerLayer != slotstream.DefaultGLM5NextSlotsPerLayer {
		t.Fatalf("SlotsPerLayer = %d, want %d", plan.SlotsPerLayer, slotstream.DefaultGLM5NextSlotsPerLayer)
	}
	if plan.NumLayers != 42 {
		t.Fatalf("NumLayers = %d, want 42", plan.NumLayers)
	}
	if slotstream.DefaultGLM5NextSlotsPerLayer >= slotstream.GLM5NextNumExperts {
		t.Fatalf("slots %d must be < experts %d to exercise streaming",
			slotstream.DefaultGLM5NextSlotsPerLayer, slotstream.GLM5NextNumExperts)
	}
	if plan.Pool.TotalVRAMBytes > slotstream.DefaultVRAMBudgetBytes {
		t.Fatalf("pool VRAM %d exceeds budget %d", plan.Pool.TotalVRAMBytes, slotstream.DefaultVRAMBudgetBytes)
	}

	// LRU via the real streaming path: fill all 54 slots, warm expert 0 with a
	// later access, then page a 55th expert and confirm the oldest (expert 1)
	// is the victim.
	ids := make([]int, slotstream.DefaultGLM5NextSlotsPerLayer)
	for e := range ids {
		ids[e] = e
	}
	hits, misses, _, err := plan.Engine.StreamPicks(3, ids, 1)
	if err != nil {
		t.Fatalf("fill: %v", err)
	}
	if hits != 0 || misses != slotstream.DefaultGLM5NextSlotsPerLayer {
		t.Fatalf("fill hits=%d misses=%d, want 0/%d", hits, misses, slotstream.DefaultGLM5NextSlotsPerLayer)
	}
	if _, _, _, err := plan.Engine.StreamPicks(3, []int{0}, 1000); err != nil {
		t.Fatalf("warm 0: %v", err)
	}
	if hits, _, _, err := plan.Engine.StreamPicks(3, []int{0}, 1001); err != nil || hits != 1 {
		t.Fatalf("expert 0 warm-read hits=%d err=%v, want 1 hit", hits, err)
	}
	if _, _, _, err := plan.Engine.StreamPicks(3, []int{999}, 2000); err != nil {
		t.Fatalf("page 999: %v", err)
	}
	if hits, _, _, err := plan.Engine.StreamPicks(3, []int{999}, 2001); err != nil || hits != 1 {
		t.Fatalf("expert 999 warm-read hits=%d err=%v, want 1 hit", hits, err)
	}
	// Expert 1 was the least-recently-used and must have been evicted; a bare
	// pool inspection (not AcquireSlot, which overwrites a victim) proves it.
	residentExperts := map[int]bool{}
	for _, s := range plan.Pool.Slots[3] {
		if s.Resident {
			residentExperts[s.ExpertID] = true
		}
	}
	if residentExperts[1] {
		t.Fatalf("expert 1 should have been the LRU victim; resident set still holds it")
	}
	if !residentExperts[999] {
		t.Fatalf("expert 999 should be resident after page-in")
	}
}

// TestGLM5NextStreamedMoEForward asserts criterion 2: the forward pass returns
// finite hidden states with mock NVMe page-in, counts hits/misses truthfully,
// and never streams more experts than the resident-slot ceiling.
func TestGLM5NextStreamedMoEForward(t *testing.T) {
	const inDim = 8
	const interDim = 16
	const numExperts = slotstream.GLM5NextNumExperts

	plan, err := NewGLM5NextSlotStreamPlan(42)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	src := &cycleSource{}

	// Shared expert stays resident (not paged).
	shared := GLM5NextExpertWeight{
		WAct:  make([]float32, interDim*inDim),
		WUp:   make([]float32, interDim*inDim),
		WDown: make([]float32, inDim*interDim),
	}
	for i := range shared.WAct {
		shared.WAct[i] = 0.01
		shared.WUp[i] = 0.01
	}
	for i := range shared.WDown {
		shared.WDown[i] = 0.01
	}

	x := make([]float32, inDim)
	for i := range x {
		x[i] = 1.0
	}

	route := GLM5NextMoERouteResult{
		ExpertIndices: []int{0, 1, 2, 3, 4, 5, 6, 7},
		Weights:       []float32{0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1},
	}

	res, err := ExecuteGLM5NextStreamedMoE(3, x, route, shared, inDim, interDim, plan, src, headerCodec{}, 100)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(res.Out) != inDim {
		t.Fatalf("len(out) = %d, want %d", len(res.Out), inDim)
	}
	for i, v := range res.Out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("out[%d] = %v, want finite", i, v)
		}
	}
	if res.Misses != 8 || res.Hits != 0 {
		t.Fatalf("first call hits=%d misses=%d, want 0/8", res.Hits, res.Misses)
	}
	if src.reads != 8 {
		t.Fatalf("source reads = %d, want 8 (one per miss)", src.reads)
	}
	if res.Slots != slotstream.DefaultGLM5NextSlotsPerLayer {
		t.Fatalf("reported slots = %d, want %d", res.Slots, slotstream.DefaultGLM5NextSlotsPerLayer)
	}

	// Second call with the same top-8 is all hits and re-reads nothing.
	src.reads = 0
	res2, err := ExecuteGLM5NextStreamedMoE(3, x, route, shared, inDim, interDim, plan, src, headerCodec{}, 200)
	if err != nil {
		t.Fatalf("forward 2: %v", err)
	}
	if res2.Hits != 8 || res2.Misses != 0 {
		t.Fatalf("second call hits=%d misses=%d, want 8/0", res2.Hits, res2.Misses)
	}
	if src.reads != 0 {
		t.Fatalf("warm slot re-read the source: reads=%d, want 0", src.reads)
	}
	for i := range res.Out {
		if math.Abs(float64(res.Out[i]-res2.Out[i])) > 1e-6 {
			t.Fatalf("out[%d] drifted warm: %g vs %g", i, res.Out[i], res2.Out[i])
		}
	}

	// A slot count below the expert count bounds the resident working set: per
	// layer the pool holds SlotsPerLayer expert records, never all 288.
	allResident := int64(numExperts) * slotstream.ExpertChunkBytes
	slotted := int64(plan.SlotsPerLayer) * slotstream.ExpertChunkBytes
	if slotted >= allResident {
		t.Fatalf("slotted bytes %d not below all-resident bytes %d", slotted, allResident)
	}
}
