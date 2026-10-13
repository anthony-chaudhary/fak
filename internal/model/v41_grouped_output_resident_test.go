package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU-backed operation/ownership recorder, not a physical accelerator witness.
type v41GroupedResidentRecorder struct {
	futureWeight compute.Tensor
	malformed    string
	lastWeight   compute.Tensor
	priorRecord  compute.Tensor
	compute.Backend
	*v41DeferredLedger
	uploads, reads, matmuls, gathers, rowReads int
	matmulFaultAt                              int
	fault                                      error
	deny                                       bool
	created, freed                             map[compute.Buffer]int
}

func newV41GroupedResidentRecorder() *v41GroupedResidentRecorder {
	b := &v41GroupedResidentRecorder{Backend: compute.Default(), created: map[compute.Buffer]int{}, freed: map[compute.Buffer]int{}}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41GroupedResidentRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41GroupedResidentRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}
func (b *v41GroupedResidentRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, shape, append([]float32(nil), values...))
	b.created[x.Buf()]++
	return x
}
func (b *v41GroupedResidentRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.uploads++
	return b.tensor(x.Shape, b.Backend.Read(x))
}
func (b *v41GroupedResidentRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	b.lastWeight = w
	if b.matmuls == b.matmulFaultAt {
		panic(b.fault)
	}
	y := b.Backend.MatMul(w, x)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41GroupedResidentRecorder) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(x)
}
func (b *v41GroupedResidentRecorder) Free(x compute.Tensor) {
	b.freed[x.Buf()]++
	delete(b.deferredPending, x.Buf())
	b.Backend.Free(x)
}
func (b *v41GroupedResidentRecorder) SupportsV41BF16Publish(n int) bool { return n > 0 }
func (b *v41GroupedResidentRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	p, st, err := b.V41BF16PublishDeferred(x, site)
	if err != nil {
		return p, err
	}
	verdict, _, err := b.V41ReadDeferred([]compute.V41DeferredStatus{st}, nil)
	if err != nil {
		return p, err
	}
	return p, verdict[0]
}
func (b *v41GroupedResidentRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	p, st, err := b.v41DeferredLedger.V41BF16PublishDeferred(x, site)
	b.created[p.Buf()]++
	b.created[st.Record.Buf()]++
	old := st.Record
	active := b.deferredPublishes == 1 || (b.malformed == "record-prior" && b.deferredPublishes == 2)
	if active {
		switch b.malformed {
		case "publication-partial":
			err = b.fault
		case "nil-output":
			b.Free(p)
			p = compute.Tensor{}
		case "aliased-output":
			b.Free(p)
			p = x
		case "record-output":
			b.Free(old)
			st.Record = p
		case "record-input":
			b.Free(old)
			st.Record = x
		case "output-future-weight":
			b.Free(p)
			p = b.futureWeight
		case "record-future-weight":
			b.Free(old)
			st.Record = b.futureWeight
		case "record-weight":
			b.Free(old)
			st.Record = b.lastWeight
		case "record-prior":
			if b.deferredPublishes == 2 {
				b.Free(old)
				st.Record = b.priorRecord
			}
		case "record-metadata":
			st.Kind = 0
		}
	}
	if b.deferredPublishes == 1 {
		b.priorRecord = st.Record
	}
	return p, st, err
}
func (b *v41GroupedResidentRecorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.rowReads += len(rows)
	return b.v41DeferredLedger.V41ReadDeferred(st, rows)
}
func (b *v41GroupedResidentRecorder) V41RowLogAvailable() bool { return !b.deny }
func (b *v41GroupedResidentRecorder) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	return b.tensor([]int{rows, width}, make([]float32, rows*width)), nil
}
func (b *v41GroupedResidentRecorder) V41RowLogCopy(compute.Tensor, int, compute.Tensor, int, int) error {
	panic("resident grouped output needs gather only")
}
func (b *v41GroupedResidentRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	b.gathers++
	rows, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	var values []float32
	for _, span := range spans {
		// Backend-internal copies; deliberately do not call the host Read API.
		x := b.Backend.Read(span.Src)
		values = append(values, x[span.Row*width:(span.Row+span.Rows)*width]...)
	}
	out := b.tensor([]int{rows, width}, values)
	if b.malformed == "gather-future-weight" {
		b.Free(out)
		return b.futureWeight, nil
	}
	if b.malformed == "gather-partial" {
		return out, b.fault
	}
	return out, nil
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedResidentChain(t *testing.T) {
	for _, fault := range []string{"none", "group-status", "later-matmul", "earlier-status-before-later-matmul", "output-status", "transport", "earlier-status-before-later-weight", "failed-group-before-output-weight", "publication-partial", "nil-output", "aliased-output", "record-output", "record-input", "record-weight", "record-prior", "record-metadata", "gather-partial", "output-future-weight", "record-future-weight", "gather-future-weight"} {
		t.Run(fault, func(t *testing.T) {
			b := newV41GroupedResidentRecorder()
			const groups, width, rank, dim = 2, 4, 3, 2
			input := []float32{1, 2, 3, 4, -1, 2, -3, 4}
			a := []float32{.1, .2, .3, .4, -.2, .3, -.4, .5, .3, .5, -.7, .2}
			bb := []float32{.2, .3, .4, .5, .6, .7, -.1, .2, -.3, .4, -.5, .6}
			weights := []compute.Tensor{compute.NewF32(b, []int{rank, width}, a), compute.NewF32(b, []int{rank, width}, a)}
			output := compute.NewF32(b, []int{dim, groups * rank}, bb)
			cause, later := errors.New("earlier group arithmetic"), errors.New("later native failure")
			b.malformed, b.fault = fault, later
			switch fault {
			case "group-status", "earlier-status-before-later-matmul", "earlier-status-before-later-weight", "failed-group-before-output-weight":
				b.deferredFaultAt, b.deferredFault = 1, cause
			case "output-status":
				b.deferredFaultAt, b.deferredFault = groups+1, cause
			case "transport":
				b.deferredReadErr = cause
			}
			if fault == "later-matmul" || fault == "earlier-status-before-later-matmul" {
				b.matmulFaultAt, b.fault = 2, later
			}
			ops, ok := v41GroupedResidentCapabilities(b, rank, dim)
			if !ok {
				t.Fatal("recorder capability refused")
			}
			b.futureWeight = output
			s := &Session{Backend: b, halW: map[string]compute.Tensor{"future-output": output}}
			stage := ""
			var stats v41GroupedResidentStats
			var got []float32
			var err error
			resolvers := make([]func() compute.Tensor, len(weights))
			for g := range resolvers {
				resolvers[g] = func() compute.Tensor {
					if g == 1 && fault == "earlier-status-before-later-weight" {
						panic(later)
					}
					return weights[g]
				}
			}
			resolveOutput := func() compute.Tensor {
				if fault == "failed-group-before-output-weight" {
					t.Fatal("output weight staged after failed group")
				}
				return output
			}
			panicked := panicAsError(func() {
				got, err = s.v41GroupedResident(ops, resolvers, resolveOutput, input, rank, dim, []compute.Tensor{output}, &stage, &stats)
			})
			if panicked != nil {
				err = panicked
			}
			if fault == "none" {
				var joined []float32
				for g := range groups {
					x := compute.NewF32(b.Backend, []int{width}, input[g*width:(g+1)*width])
					y := b.Backend.Read(b.Backend.MatMul(weights[g], x))
					for _, v := range y {
						joined = append(joined, v41RoundBF16(v))
					}
				}
				want := b.Backend.Read(b.Backend.MatMul(output, compute.NewF32(b.Backend, []int{len(joined)}, joined)))
				for i := range want {
					want[i] = v41RoundBF16(want[i])
				}
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("parity got=%v want=%v err=%v", got, want, err)
				}
				if b.gathers != 1 || b.matmuls != groups+1 || b.uploads != groups || b.reads != 0 || b.rowReads != 1 || b.deferredFences != 2 {
					t.Fatalf("wrong operations: gather=%d matmul=%d uploads=%d reads=%d rows=%d fences=%d", b.gathers, b.matmuls, b.uploads, b.reads, b.rowReads, b.deferredFences)
				}
				if stats.upload != int64(len(input)*4) || stats.readback != dim*4 || stats.calls != groups+1 {
					t.Fatalf("stats=%+v", stats)
				}
			} else {
				if err == nil || got != nil {
					t.Fatalf("failed chain emitted output: %v %v", got, err)
				}
				if fault != "output-status" && (b.matmuls > groups || (fault != "gather-partial" && fault != "gather-future-weight" && b.gathers != 0)) {
					t.Fatal("failed group reached concat/wo_b")
				}
				if (fault == "earlier-status-before-later-matmul" || fault == "earlier-status-before-later-weight") && (!errors.Is(err, cause) || !errors.Is(err, later) || stage != "BF16 publication") {
					t.Fatalf("lost primary verdict: %v stage=%s", err, stage)
				}
			}
			for buf := range b.created {
				if b.freed[buf] != 1 {
					t.Fatalf("temporary released %d times", b.freed[buf])
				}
			}
			for _, w := range append(weights, output) {
				if b.freed[w.Buf()] != 0 {
					t.Fatal("borrowed weight freed")
				}
			}
			if len(b.deferredPending) != 0 {
				t.Fatal("deferred status leaked")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41GroupedResidentDeclinesBeforeMutation(t *testing.T) {
	b := newV41GroupedResidentRecorder()
	b.deny = true
	if _, ok := v41GroupedResidentCapabilities(b, 3, 2); ok {
		t.Fatal("unsupported row log admitted")
	}
	b.deny = false
	t.Setenv(v41DeferredBF16PublishEnv, "0")
	if _, ok := v41GroupedResidentCapabilities(b, 3, 2); ok {
		t.Fatal("disabled deferred publication admitted")
	}
	if b.uploads+b.reads+b.matmuls+b.gathers != 0 || len(b.created) != 0 {
		t.Fatal("decline mutated device")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedResidentSessionBridge(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "selected-failure"}[fail], func(t *testing.T) {
			const heads, hd, groups, rank, dim = 2, 512, 2, 3, 2
			manifest, raw := synthBuildRaw([]synthTensor{
				{layerName(0, "attn.wo_a.weight"), []int{groups * rank, heads * hd / groups}},
				{layerName(0, "attn.wo_b.weight"), []int{dim, groups * rank}},
			}, synthMatmulFill)
			m := &Model{Cfg: Config{HeadDim: hd, NumLayers: 1, HiddenSize: dim}, manifest: manifest, raw: raw}
			b := newV41GroupedResidentRecorder()
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			input := make([]float32, heads*hd)
			for i := range input {
				input[i] = float32(i%7 - 3)
			}
			cause := errors.New("selected resident group arithmetic")
			if fail {
				b.deferredFaultAt, b.deferredFault = 1, cause
			}
			operation := s.v41GroupedOutputFunc()
			if operation == nil {
				t.Fatal("production grouped callback absent")
			}
			got, outcome, err := operation(0, input, heads, hd, groups, rank, dim)
			if fail {
				if got != nil || outcome != v41ProjectionError || !errors.Is(err, cause) || !s.BackendSessionClosed() || b.matmuls != groups || b.gathers != 0 {
					t.Fatalf("selected failure escaped: outcome=%v err=%v closed=%t matmul=%d gather=%d", outcome, err, s.BackendSessionClosed(), b.matmuls, b.gathers)
				}
				return
			}
			a, _ := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
			bb, _ := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
			want, werr := v41FullGroupedOutput(0, input, a, bb, heads, hd, groups, rank, dim)
			if err != nil || werr != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) || b.gathers != 1 || b.rowReads != 1 {
				t.Fatalf("production bridge parity/reachability got=%v want=%v err=%v ref=%v gathers=%d reads=%d", got, want, err, werr, b.gathers, b.rowReads)
			}
		})
	}
}
