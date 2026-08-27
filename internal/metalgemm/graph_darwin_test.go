//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"time"
)

func TestProjectionGraphMixedQuantizedSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()
	const P, in, out = 2, 256, 32
	xf := q4kTestVector(P*in, 9267)
	q4 := UploadQ4K(q4kTestRaw(out, in, 9267), out, in)
	if q4 == nil {
		t.Fatal("q4 upload")
	}
	// Q6 remains a valid zero matrix in this mixed packet; Q4/Q8 carry non-zero parity.
	q6 := UploadQ6K(make([]byte, out*(in/256)*210), out, in)
	if q6 == nil {
		t.Fatal("q6 upload")
	}
	q8codes := make([]int8, out*in)
	q8scales := make([]float32, out*(in/32))
	for i := range q8codes {
		q8codes[i] = int8(i%15 - 7)
	}
	for i := range q8scales {
		q8scales[i] = 0.02
	}
	q8 := UploadQ8(q8codes, q8scales, out, in)
	if q8 == nil {
		t.Fatal("q8 upload")
	}
	xq := make([]int8, P*in)
	xd := make([]float32, P*(in/32))
	for row := 0; row < P; row++ {
		for b := 0; b < in/32; b++ {
			xd[row*(in/32)+b] = 0.01
			for j := 0; j < 32; j++ {
				xq[row*in+b*32+j] = int8((row+b+j)%17 - 8)
			}
		}
	}

	want4 := make([]float32, P*out)
	q4.GEMM(xf, P, want4)
	want6 := make([]float32, P*out)
	q6.GEMM(xf, P, want6)
	want8 := make([]float32, P*out)
	q8.GEMM(xq, xd, P, want8)
	g, err := BeginProjectionGraph(xf, xq, xd, P, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	r4, err := g.EncodeQ4K(q4)
	if err != nil {
		t.Fatal(err)
	}
	r6, err := g.EncodeQ6K(q6)
	if err != nil {
		t.Fatal(err)
	}
	r8, err := g.EncodeQ8(q8)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := g.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 3 || receipt.HostReadbacks != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
	for i, pair := range []struct {
		r    *GraphResult
		want []float32
	}{{r4, want4}, {r6, want6}, {r8, want8}} {
		got, err := g.Read(pair.r)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(pair.want) {
			t.Fatalf("result %d len=%d", i, len(got))
		}
		for j := range got {
			d := math.Abs(float64(got[j] - pair.want[j]))
			if d > 1e-5 {
				t.Fatalf("result %d[%d] got=%g want=%g delta=%g", i, j, got[j], pair.want[j], d)
			}
		}
	}
}

func TestProjectionGraphDeviceResultChainingSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()
	const P, hidden, value, intermediate, output = 32, 256, 512, 768, 256
	x := q4kTestVector(P*hidden, 9456)
	first := UploadQ4K(q4kTestRaw(value, hidden, 9456), value, hidden)
	if first == nil {
		t.Fatal("first Q4_K upload")
	}
	q8Weight := func(out, in, phase int) *Q8Weight {
		codes := make([]int8, out*in)
		scales := make([]float32, out*(in/32))
		for i := range codes {
			codes[i] = int8((i+phase)%15 - 7)
		}
		for i := range scales {
			scales[i] = 0.02
		}
		return UploadQ8(codes, scales, out, in)
	}
	second := q8Weight(intermediate, value, 3)
	third := q8Weight(output, intermediate, 9)
	if second == nil || third == nil {
		t.Fatal("width-changing Q8 uploads")
	}
	quantize := func(values []float32) ([]int8, []float32) {
		q, d := make([]int8, len(values)), make([]float32, len(values)/32)
		for block := range d {
			var amax float32
			for i := 0; i < 32; i++ {
				v := values[block*32+i]
				if v < 0 {
					v = -v
				}
				if v > amax {
					amax = v
				}
			}
			d[block] = amax / 127
			if d[block] == 0 {
				continue
			}
			for i := 0; i < 32; i++ {
				v := values[block*32+i] / d[block]
				if v < 0 {
					v = float32(math.Ceil(float64(v - .5)))
				} else {
					v = float32(math.Floor(float64(v + .5)))
				}
				q[block*32+i] = int8(v)
			}
		}
		return q, d
	}

	g, err := BeginProjectionGraph(x, nil, nil, P, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	firstResult, err := g.EncodeQ4K(first)
	if err != nil {
		t.Fatal(err)
	}
	quantized, err := g.QuantizeQ8(firstResult)
	if err != nil {
		t.Fatal(err)
	}
	secondResult, err := g.EncodeQ8From(second, quantized)
	if err != nil {
		t.Fatal(err)
	}
	quantizedAgain, err := g.QuantizeQ8(secondResult)
	if err != nil {
		t.Fatal(err)
	}
	thirdResult, err := g.EncodeQ8From(third, quantizedAgain)
	if err != nil {
		t.Fatal(err)
	}
	outputs, receipt, err := g.FinishRead(thirdResult)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || len(outputs[0]) != P*output {
		t.Fatalf("outputs shape=%d/%d, want 1/%d", len(outputs), len(outputs[0]), P*output)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 5 || receipt.HostReadbacks != 1 {
		t.Fatalf("receipt=%+v, want one command buffer, five encoders, one terminal wait/readback", receipt)
	}
	firstHost := make([]float32, P*value)
	first.GEMM(x, P, firstHost)
	q, d := quantize(firstHost)
	secondHost := make([]float32, P*intermediate)
	second.GEMM(q, d, P, secondHost)
	q, d = quantize(secondHost)
	want := make([]float32, P*output)
	third.GEMM(q, d, P, want)
	cosine, maxRel := q4kTestCosineMaxRel(want, outputs[0])
	if cosine < 0.999999 || maxRel > 1e-5 {
		t.Fatalf("width-changing device chain cosine=%g maxRel=%g", cosine, maxRel)
	}
}

func graphGDNPanel(g GDNGeometry, tokens int) GDNPanel {
	panel := GDNPanel{
		Tokens: tokens, Mixed: make([]float32, tokens*g.convDim()),
		Z: make([]float32, tokens*g.valueDim()), B: make([]float32, tokens*g.NumValueHeads),
		A: make([]float32, tokens*g.NumValueHeads), Conv1D: make([]float32, g.convDim()*g.ConvKernel),
		ALog: make([]float32, g.NumValueHeads), DTBias: make([]float32, g.NumValueHeads),
		Norm: make([]float32, g.ValueHeadDim), RMSNormEpsilon: 1e-5,
	}
	for i := range panel.Norm {
		panel.Norm[i] = 1
	}
	for i := range panel.ALog {
		panel.ALog[i] = -2
		panel.DTBias[i] = .1
	}
	return panel
}

func newProjectionGraphGDNLeaseFixture(t *testing.T) (*ProjectionGraph, *GDNState, *GraphResult, GDNGeometry) {
	t.Helper()
	const P, input = 32, 256
	geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
	g, err := BeginProjectionGraph(q4kTestVector(P*input, 945603), nil, nil, P, input)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(width, seed int) *GraphResult {
		t.Helper()
		weight := UploadQ4K(q4kTestRaw(width, input, uint64(seed)), width, input)
		if weight == nil {
			t.Fatalf("GDN fixture Q4_K upload width=%d", width)
		}
		result, encodeErr := g.EncodeQ4K(weight)
		if encodeErr != nil {
			t.Fatalf("GDN fixture projection width=%d: %v", width, encodeErr)
		}
		return result
	}
	mixed := encode(geometry.convDim(), 945604)
	z := encode(geometry.valueDim(), 945605)
	b := encode(geometry.NumValueHeads, 945606)
	a := encode(geometry.NumValueHeads, 945607)
	state, err := NewGDNState(geometry)
	if err != nil {
		g.Free()
		t.Fatal(err)
	}
	core, err := g.GDN(state, mixed, z, b, a, graphGDNPanel(geometry, P))
	if err != nil {
		state.Close()
		g.Free()
		t.Fatal(err)
	}
	return g, state, core, geometry
}

func waitForGDNGraphWaiters(t *testing.T, state *GDNState, want int) {
	t.Helper()
	for i := 0; i < 100000; i++ {
		state.mu.Lock()
		got := state.graphWaiters
		state.mu.Unlock()
		if got == want {
			return
		}
		runtime.Gosched()
	}
	state.mu.Lock()
	got := state.graphWaiters
	state.mu.Unlock()
	t.Fatalf("GDN graph waiters=%d, want %d", got, want)
}

func TestProjectionGraphGDNLeaseSerializesOwnerLifecycle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	baseline := GDNLiveBufferCount()
	g, state, core, geometry := newProjectionGraphGDNLeaseFixture(t)

	type operationResult struct {
		name string
		err  error
	}
	results := make(chan operationResult, 4)
	seedConv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	seedRecurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	go func() { results <- operationResult{name: "reset", err: state.Reset()} }()
	go func() { results <- operationResult{name: "seed", err: state.Seed(seedConv, seedRecurrent)} }()
	go func() {
		_, _, _, err := state.Run(graphGDNPanel(geometry, 1))
		results <- operationResult{name: "run", err: err}
	}()
	go func() { state.Close(); results <- operationResult{name: "close"} }()
	waitForGDNGraphWaiters(t, state, 4)
	if got := GDNLiveBufferCount(); got != baseline+2 {
		t.Fatalf("in-flight graph owner buffers=%d, want %d", got, baseline+2)
	}
	outputs, receipt, err := g.FinishRead(core)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || len(outputs[0]) != 32*geometry.valueDim() || !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 5 || receipt.HostReadbacks != 1 {
		t.Fatalf("GDN graph terminal result=%d receipt=%+v", len(outputs), receipt)
	}
	for i := 0; i < 4; i++ {
		select {
		case result := <-results:
			var declined *GDNDeclinedError
			if result.err != nil && !errors.As(result.err, &declined) {
				t.Fatalf("queued %s after graph completion: %v", result.name, result.err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("queued GDN owner operation deadlocked after terminal fence")
		}
	}
	g.Free()
	state.Close()
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("serialized graph owner leaked buffers=%d, baseline=%d", got, baseline)
	}
}

func TestProjectionGraphGDNLeaseReleasesOnFailureAndFree(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	baseline := GDNLiveBufferCount()

	t.Run("post-submit failure", func(t *testing.T) {
		g, state, _, _ := newProjectionGraphGDNLeaseFixture(t)
		g.InjectPostSubmitFailureForTest()
		receipt, err := g.Finish()
		var post *GraphPostSubmitError
		if !errors.As(err, &post) || !receipt.Committed || !receipt.CompletedWait {
			t.Fatalf("injected graph failure receipt=%+v err=%T %v", receipt, err, err)
		}
		state.Close()
		g.Free()
	})
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("failed graph leaked buffers=%d, baseline=%d", got, baseline)
	}

	t.Run("unsubmitted free", func(t *testing.T) {
		g, state, _, _ := newProjectionGraphGDNLeaseFixture(t)
		g.Free()
		g.Free()
		state.Close()
		state.Close()
	})
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("freed graph leaked buffers=%d, baseline=%d", got, baseline)
	}
}
