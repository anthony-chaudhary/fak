//go:build darwin && arm64 && cgo

package metalgemm

import (
	"os"
	"slices"
	"sort"
	"strconv"
	"testing"
	"time"
)

// q4kSmallPShapes are the parity shapes for the 2<=P<=20 route (fak#13694): a Qwen3.5-4B square
// projection, its FFN up shape, and a row count that is neither a multiple of the GEMM's 64-row
// tile nor of the multi-token GEMV's 8-row threadgroup.
var q4kSmallPShapes = []struct {
	name    string
	out, in int
}{
	{"hidden-2560x2560", 2560, 2560},
	{"ffn-up-9728x2560", 9728, 2560},
	{"ragged-1003x2560", 1003, 2560},
}

var q4kSmallPPrompts = []int{2, 3, 6, 7, 9, 13, 16, 17, 20}

const (
	q4kSmallPMinCosine = 0.99999
	q4kSmallPMaxRel    = 1e-3
)

func q4kSmallPAssertParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	cosine, maxRel := q4kTestCosineMaxRel(want, got)
	if cosine < q4kSmallPMinCosine || maxRel > q4kSmallPMaxRel {
		t.Fatalf("%s: cosine=%.9f maxRel=%.3g, want cosine >= %g and maxRel <= %g",
			label, cosine, maxRel, q4kSmallPMinCosine, q4kSmallPMaxRel)
	}
}

// TestQ4KSmallPSelectorModeForPrompt pins the darwin production selector without touching the
// GPU: 2<=P<=20 requests the small-P GEMV under the default flags, P=1 and P>20 stay scalar,
// and the opt-out returns every P to scalar. MM32 (exact P32) precedence is unchanged.
// fak-test:runtime fast est=1ms lane=default
func TestQ4KSmallPSelectorModeForPrompt(t *testing.T) {
	priorSmallP, priorMM, priorM5 := q4kUseSmallP.Load(), q4kUseMM.Swap(false), q4kUseM5.Swap(false)
	defer func() {
		q4kUseSmallP.Store(priorSmallP)
		q4kUseMM.Store(priorMM)
		q4kUseM5.Store(priorM5)
	}()
	q4kUseSmallP.Store(true)
	for P, want := range map[int]Q4KGEMMMode{
		1: Q4KGEMMModeScalar, 2: Q4KGEMMModeSmallPGEMV, 16: Q4KGEMMModeSmallPGEMV,
		20: Q4KGEMMModeSmallPGEMV, 21: Q4KGEMMModeScalar, 32: Q4KGEMMModeScalar, 64: Q4KGEMMModeScalar,
	} {
		if got := q4kGEMMModeForPrompt(P); got != want {
			t.Errorf("default P=%d mode=%v, want %v", P, got, want)
		}
	}
	if got := Q4KGEMMRequestedExecution(9); got != Q4KGEMMExecutedSmallPGEMV {
		t.Errorf("default P=9 requested=%v, want %v", got, Q4KGEMMExecutedSmallPGEMV)
	}
	q4kUseMM.Store(true)
	if got := q4kGEMMModeForPrompt(32); got != Q4KGEMMModeMM32 {
		t.Errorf("FAK_Q4K_MM P=32 mode=%v, want mm32", got)
	}
	q4kUseMM.Store(false)
	q4kUseSmallP.Store(false)
	for _, P := range []int{1, 2, 16, 20, 21, 32, 64} {
		if got := q4kGEMMModeForPrompt(P); got != Q4KGEMMModeScalar {
			t.Errorf("opt-out P=%d mode=%v, want scalar", P, got)
		}
	}
	// An explicit small-P request outside the band is credited to the scalar kernel.
	for _, P := range []int{1, 21, 64} {
		if got := q4kGEMMRequestedExecution(P, Q4KGEMMModeSmallPGEMV); got != Q4KGEMMExecutedScalar {
			t.Errorf("explicit small-P P=%d requested=%v, want scalar", P, got)
		}
	}
}

// TestQ4KSmallPGemvParity is the fak#13694 red/green gate: under the default selector every
// 2<=P<=20 Q4_K GEMM must execute the batched multi-token GEMV (typed identity) and match the
// explicit scalar q4k_gemm kernel, including the ragged 9/16-token chunk split and a row count
// that is not a multiple of 8. The fail-closed witness pins that a missing multi-token pipeline
// executes (and is credited to) the scalar kernel, never NotExecuted.
// fak-test:runtime integration est=5s lane=default
func TestQ4KSmallPGemvParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	prior := q4kUseSmallP.Swap(true)
	defer q4kUseSmallP.Store(prior)
	smallP := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedSmallPGEMV, Executed: Q4KGEMMExecutedSmallPGEMV}
	scalar := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedScalar, Executed: Q4KGEMMExecutedScalar}
	for _, shape := range q4kSmallPShapes {
		w := UploadQ4K(q4kTestRaw(shape.out, shape.in, 0x13694), shape.out, shape.in)
		if w == nil {
			t.Fatalf("%s: UploadQ4K returned nil", shape.name)
		}
		for _, P := range q4kSmallPPrompts {
			x := q4kTestVector(P*shape.in, int64(13694+P))
			want := make([]float32, P*shape.out)
			got := make([]float32, P*shape.out)
			if id := w.GEMMWithEventsMode(x, P, want, nil, Q4KGEMMModeScalar); id != scalar {
				t.Fatalf("%s P=%d scalar control identity=%+v, want %+v", shape.name, P, id, scalar)
			}
			observation := NewExecutionObservation(ExecutionQ4KGEMM)
			id, err := w.GEMMWithEventsModeErr(x, P, got, observation, q4kGEMMModeForPrompt(P))
			if err != nil {
				t.Fatalf("%s P=%d small-P: %v", shape.name, P, err)
			}
			if id != smallP {
				t.Fatalf("%s P=%d default-selector identity=%+v (requested %v / executed %v), want small-P GEMV",
					shape.name, P, id, id.Requested, id.Executed)
			}
			if snap, err := observation.Snapshot(); err != nil || len(snap.Events) != 1 || !snap.Events[0].CompletedWait {
				t.Fatalf("%s P=%d events=%+v err=%v, want one completed command buffer", shape.name, P, snap.Events, err)
			}
			q4kSmallPAssertParity(t, shape.name+" P="+strconv.Itoa(P), want, got)
		}
		// Fail-closed: the unavailable witness requests small-P, executes the scalar kernel, and
		// its output is bit-identical to the scalar control.
		const P = 6
		x := q4kTestVector(P*shape.in, 136946)
		want := make([]float32, P*shape.out)
		got := make([]float32, P*shape.out)
		w.GEMMWithEventsMode(x, P, want, nil, Q4KGEMMModeScalar)
		fallback := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedSmallPGEMV, Executed: Q4KGEMMExecutedScalar}
		if id := w.GEMMWithEventsMode(x, P, got, nil, Q4KGEMMModeSmallPGEMVUnavailable); id != fallback {
			t.Fatalf("%s unavailable small-P identity=%+v, want %+v", shape.name, id, fallback)
		}
		if !slices.Equal(want, got) {
			t.Fatalf("%s unavailable small-P fallback is not the scalar kernel", shape.name)
		}
		// An explicit small-P request outside the band (P=1, P=21) also executes scalar.
		for _, P := range []int{1, 21} {
			x := q4kTestVector(P*shape.in, int64(136947+P))
			want := make([]float32, P*shape.out)
			got := make([]float32, P*shape.out)
			w.GEMMWithEventsMode(x, P, want, nil, Q4KGEMMModeScalar)
			if id := w.GEMMWithEventsMode(x, P, got, nil, Q4KGEMMModeSmallPGEMV); id != scalar {
				t.Fatalf("%s out-of-band P=%d small-P identity=%+v, want scalar", shape.name, P, id)
			}
			if !slices.Equal(want, got) {
				t.Fatalf("%s out-of-band P=%d small-P request did not run the scalar kernel", shape.name, P)
			}
		}
		w.Release()
	}
}

// TestQ4KSmallPGemvGroupAndGraphParity covers the two other prefill entry points: a grouped GEMM
// (one command buffer, shared activation panel, per-weight Y offsets) and a ProjectionGraph encode
// (EncodeQ4K and a device-chained EncodeQ4KFrom) at the ragged P=9 split. Group members must be
// bit-identical to the single small-P GEMM and both must match the scalar kernel.
// fak-test:runtime integration est=3s lane=default
func TestQ4KSmallPGemvGroupAndGraphParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	prior := q4kUseSmallP.Swap(true)
	defer q4kUseSmallP.Store(prior)
	const P, in = 9, 2560
	smallP := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedSmallPGEMV, Executed: Q4KGEMMExecutedSmallPGEMV}
	ws := make([]*Q4KWeight, 0, len(q4kSmallPShapes))
	for i, shape := range q4kSmallPShapes {
		w := UploadQ4K(q4kTestRaw(shape.out, in, uint64(0x13694+i)), shape.out, in)
		if w == nil {
			t.Fatalf("%s: UploadQ4K returned nil", shape.name)
		}
		ws = append(ws, w)
	}
	x := q4kTestVector(P*in, 1369409)
	scalarOf := func(w *Q4KWeight, x []float32) []float32 {
		y := make([]float32, P*w.Out)
		w.GEMMWithEventsMode(x, P, y, nil, Q4KGEMMModeScalar)
		return y
	}

	total := 0
	for _, w := range ws {
		total += P * w.Out
	}
	observation := NewExecutionObservation(ExecutionQ4KGEMMGroup)
	group, id, err := GEMMGroupIntoWithEventsModeErr(ws, x, P, make([]float32, total), observation, Q4KGEMMModeForPrompt(P))
	if err != nil || group == nil {
		t.Fatalf("group: out=%v err=%v", group, err)
	}
	if id != smallP {
		t.Fatalf("group identity=%+v, want small-P GEMV", id)
	}
	for i, w := range ws {
		single := make([]float32, P*w.Out)
		if sid := w.GEMMWithEventsMode(x, P, single, nil, Q4KGEMMModeSmallPGEMV); sid != smallP {
			t.Fatalf("group member %d single identity=%+v", i, sid)
		}
		if !slices.Equal(group[i], single) {
			t.Fatalf("group member %d differs from its single small-P GEMM", i)
		}
		q4kSmallPAssertParity(t, "group "+q4kSmallPShapes[i].name, scalarOf(w, x), group[i])
	}

	g, err := BeginProjectionGraph(x, nil, nil, P, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	if !g.SetQ4KGEMMMode(Q4KGEMMModeForPrompt(P)) {
		t.Fatal("graph refused the small-P mode at P=9")
	}
	if got := g.Q4KGEMMMode(); got != Q4KGEMMModeSmallPGEMV {
		t.Fatalf("graph mode=%v, want small-P GEMV", got)
	}
	first, err := g.EncodeQ4K(ws[0])
	if err != nil {
		t.Fatal(err)
	}
	chained, err := g.EncodeQ4KFrom(ws[1], first)
	if err != nil {
		t.Fatal(err)
	}
	ragged, err := g.EncodeQ4K(ws[2])
	if err != nil {
		t.Fatal(err)
	}
	outs, receipt, err := g.FinishRead(first, chained, ragged)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("graph receipt=%+v", receipt)
	}
	if !slices.Equal(outs[0], group[0]) || !slices.Equal(outs[2], group[2]) {
		t.Fatal("graph small-P projection differs from the direct small-P GEMM")
	}
	firstScalar := scalarOf(ws[0], x)
	q4kSmallPAssertParity(t, "graph first", firstScalar, outs[0])
	q4kSmallPAssertParity(t, "graph chained", scalarOf(ws[1], firstScalar), outs[1])
	q4kSmallPAssertParity(t, "graph ragged", scalarOf(ws[2], x), outs[2])

	// A graph whose P is outside the band refuses the mode and keeps the scalar kernel.
	g21, err := BeginProjectionGraph(q4kTestVector(21*in, 1369417), nil, nil, 21, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g21.Free()
	if g21.SetQ4KGEMMMode(Q4KGEMMModeSmallPGEMV) {
		t.Fatal("P=21 graph accepted the small-P mode")
	}
	if got := g21.Q4KGEMMMode(); got != Q4KGEMMModeScalar {
		t.Fatalf("P=21 graph mode=%v, want scalar", got)
	}
}

// q6kSmallPTestRaw returns deterministic Q6_K rows: random ql/qh/scale bytes with a modest,
// finite f16 super-block scale so every block decodes to bounded weights.
func q6kSmallPTestRaw(out, in int, seed uint64) []byte {
	const blockBytes = 210
	raw := make([]byte, out*(in/256)*blockBytes)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state >> 56)
	}
	for base := 0; base < len(raw); base += blockBytes {
		raw[base+209] = 0x20 | raw[base+209]&0x03
	}
	return raw
}

// TestQ6KSmallPGemvParity is the Q6_K half of the fak#13694 gate: q4_k_m GGUFs keep ffn_down and
// attn_v in Q6_K, so a 2<=P<=20 prefill must route those through q6k_gemv_multiN too. The default
// selector must execute the small-P GEMV and match q6k_gemm; the unavailable witness and
// out-of-band prompts must execute (and be credited to) q6k_gemm bit-exactly; and a small-P
// ProjectionGraph must encode the same kernel for EncodeQ6K / EncodeQ6KFrom.
// fak-test:runtime integration est=5s lane=default
func TestQ6KSmallPGemvParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	prior := q4kUseSmallP.Swap(true)
	defer q4kUseSmallP.Store(prior)
	shapes := []struct {
		name    string
		out, in int
	}{{"ffn-down-2560x9728", 2560, 9728}, {"ragged-1003x2560", 1003, 2560}}
	ws := make([]*Q6KWeight, len(shapes))
	for i, shape := range shapes {
		w := UploadQ6K(q6kSmallPTestRaw(shape.out, shape.in, uint64(0x613694+i)), shape.out, shape.in)
		if w == nil {
			t.Fatalf("%s: UploadQ6K returned nil", shape.name)
		}
		ws[i] = w
		for _, P := range q4kSmallPPrompts {
			x := q4kTestVector(P*shape.in, int64(613694+P))
			want := make([]float32, P*shape.out)
			got := make([]float32, P*shape.out)
			if ex := w.GEMMWithEventsMode(x, P, want, nil, Q4KGEMMModeScalar); ex != Q4KGEMMExecutedScalar {
				t.Fatalf("%s P=%d scalar control executed=%v", shape.name, P, ex)
			}
			observation := NewExecutionObservation(ExecutionQ6KGEMM)
			if ex := w.GEMMWithEventsMode(x, P, got, observation, q4kGEMMModeForPrompt(P)); ex != Q4KGEMMExecutedSmallPGEMV {
				t.Fatalf("%s P=%d default-selector executed=%v, want small-P GEMV", shape.name, P, ex)
			}
			if snap, err := observation.Snapshot(); err != nil || len(snap.Events) != 1 || !snap.Events[0].CompletedWait {
				t.Fatalf("%s P=%d events=%+v err=%v, want one completed command buffer", shape.name, P, snap.Events, err)
			}
			q4kSmallPAssertParity(t, "q6k "+shape.name+" P="+strconv.Itoa(P), want, got)
			// The legacy entry point uses the same selector.
			legacy := make([]float32, P*shape.out)
			w.GEMMWithEvents(x, P, legacy, nil)
			if !slices.Equal(legacy, got) {
				t.Fatalf("%s P=%d GEMMWithEvents differs from the small-P route", shape.name, P)
			}
		}
		for _, tc := range []struct {
			P    int
			mode Q4KGEMMMode
		}{{6, Q4KGEMMModeSmallPGEMVUnavailable}, {1, Q4KGEMMModeSmallPGEMV}, {21, Q4KGEMMModeSmallPGEMV}} {
			x := q4kTestVector(tc.P*shape.in, int64(613700+tc.P))
			want := make([]float32, tc.P*shape.out)
			got := make([]float32, tc.P*shape.out)
			w.GEMMWithEventsMode(x, tc.P, want, nil, Q4KGEMMModeScalar)
			if ex := w.GEMMWithEventsMode(x, tc.P, got, nil, tc.mode); ex != Q4KGEMMExecutedScalar {
				t.Fatalf("%s P=%d mode=%v executed=%v, want scalar fallback", shape.name, tc.P, tc.mode, ex)
			}
			if !slices.Equal(want, got) {
				t.Fatalf("%s P=%d mode=%v fallback is not q6k_gemm", shape.name, tc.P, tc.mode)
			}
		}
	}

	// Graph: a P=9 small-P graph projects the ffn-down shape from the panel and the ragged shape
	// from a device-resident Q4_K result (the hybrid prefill's chained form).
	const P = 9
	down, ragged := ws[0], ws[1]
	x := q4kTestVector(P*down.In, 6136949)
	up := UploadQ4K(q4kTestRaw(ragged.In, down.In, 0x613699), ragged.In, down.In)
	if up == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	g, err := BeginProjectionGraph(x, nil, nil, P, down.In)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	if !g.SetQ4KGEMMMode(Q4KGEMMModeForPrompt(P)) {
		t.Fatal("graph refused the small-P mode at P=9")
	}
	r1, err := g.EncodeQ6K(down)
	if err != nil {
		t.Fatal(err)
	}
	mid, err := g.EncodeQ4K(up)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := g.EncodeQ6KFrom(ragged, mid)
	if err != nil {
		t.Fatal(err)
	}
	outs, receipt, err := g.FinishRead(r1, mid, r2)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("graph receipt=%+v", receipt)
	}
	direct := make([]float32, P*down.Out)
	if ex := down.GEMMWithEventsMode(x, P, direct, nil, Q4KGEMMModeSmallPGEMV); ex != Q4KGEMMExecutedSmallPGEMV {
		t.Fatalf("direct small-P executed=%v", ex)
	}
	if !slices.Equal(outs[0], direct) {
		t.Fatal("graph EncodeQ6K differs from the direct small-P Q6_K GEMM")
	}
	scalarDown := make([]float32, P*down.Out)
	down.GEMMWithEventsMode(x, P, scalarDown, nil, Q4KGEMMModeScalar)
	q4kSmallPAssertParity(t, "graph q6k panel", scalarDown, outs[0])
	scalarRagged := make([]float32, P*ragged.Out)
	ragged.GEMMWithEventsMode(outs[1], P, scalarRagged, nil, Q4KGEMMModeScalar)
	q4kSmallPAssertParity(t, "graph q6k chained", scalarRagged, outs[2])
}

// TestQ4KSmallPGemvTiming is the moment-in-time scalar vs small-P timing probe for fak#13694.
// It is opt-in (FAK_Q4K_SMALLP_BENCH=1) because it occupies the single-tenant GPU for a while;
// it logs the median wall and GPU milliseconds per call at P in {2,6,9,16} over the Qwen3.5-4B
// and Qwen2.5-7B projection shapes.
// fak-test:runtime medium est=30s lane=optin
func TestQ4KSmallPGemvTiming(t *testing.T) {
	if os.Getenv("FAK_Q4K_SMALLP_BENCH") != "1" {
		t.Skip("set FAK_Q4K_SMALLP_BENCH=1 to time scalar vs small-P GEMV")
	}
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const iters = 10
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	for _, shape := range []struct {
		name    string
		out, in int
	}{
		{"4B-hidden", 2560, 2560}, {"4B-ffn-up", 9728, 2560}, {"4B-ffn-down", 2560, 9728},
		{"7B-hidden", 3584, 3584}, {"7B-ffn-up", 18944, 3584},
	} {
		w := UploadQ4K(q4kTestRaw(shape.out, shape.in, 0x13694), shape.out, shape.in)
		if w == nil {
			t.Fatalf("%s: UploadQ4K returned nil", shape.name)
		}
		for _, P := range q4kSmallPPrompts {
			x := q4kTestVector(P*shape.in, int64(P))
			y := make([]float32, P*shape.out)
			for _, arm := range []struct {
				name string
				mode Q4KGEMMMode
			}{{"scalar", Q4KGEMMModeScalar}, {"smallp", Q4KGEMMModeSmallPGEMV}} {
				w.GEMMWithEventsMode(x, P, y, nil, arm.mode) // warm the pipeline and caches
				wall, gpu := make([]float64, iters), make([]float64, iters)
				for i := range wall {
					start := time.Now()
					w.GEMMWithEventsMode(x, P, y, nil, arm.mode)
					wall[i] = float64(time.Since(start).Microseconds()) / 1000
					gpu[i] = LastGEMMGPUMs()
				}
				t.Logf("q4k %-12s [%5d,%5d] P=%2d %-6s wall_ms=%.3f gpu_ms=%.3f",
					shape.name, shape.out, shape.in, P, arm.name, median(wall), median(gpu))
			}
		}
		w.Release()
	}
	for _, shape := range []struct {
		name    string
		out, in int
	}{{"4B-ffn-down", 2560, 9728}, {"7B-ffn-down", 3584, 18944}} {
		w := UploadQ6K(q6kSmallPTestRaw(shape.out, shape.in, 0x613694), shape.out, shape.in)
		if w == nil {
			t.Fatalf("%s: UploadQ6K returned nil", shape.name)
		}
		for _, P := range q4kSmallPPrompts {
			x := q4kTestVector(P*shape.in, int64(P))
			y := make([]float32, P*shape.out)
			for _, arm := range []struct {
				name string
				mode Q4KGEMMMode
			}{{"scalar", Q4KGEMMModeScalar}, {"smallp", Q4KGEMMModeSmallPGEMV}} {
				w.GEMMWithEventsMode(x, P, y, nil, arm.mode)
				wall, gpu := make([]float64, iters), make([]float64, iters)
				for i := range wall {
					observation := NewExecutionObservation(ExecutionQ6KGEMM)
					start := time.Now()
					w.GEMMWithEventsMode(x, P, y, observation, arm.mode)
					wall[i] = float64(time.Since(start).Microseconds()) / 1000
					if snap, err := observation.Snapshot(); err == nil && len(snap.Events) == 1 {
						gpu[i] = snap.Events[0].GPUMilliseconds
					}
				}
				t.Logf("q6k %-12s [%5d,%5d] P=%2d %-6s wall_ms=%.3f gpu_ms=%.3f",
					shape.name, shape.out, shape.in, P, arm.name, median(wall), median(gpu))
			}
		}
		w.Release()
	}
}
