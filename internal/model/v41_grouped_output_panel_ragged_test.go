package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedRaggedOp struct {
	kind  string
	dtype compute.Dtype
	count int
}

// Reuse the CPU ownership/status ledger while retaining this recorder as the
// owner of both packed-panel and scalar-tail results. This is a wiring witness.
type v41GroupedRaggedRecorder struct {
	*v41GroupedResidentRecorder
	ops         []v41GroupedRaggedOp
	weights     []compute.Tensor
	denyQ3Rows  int
	scalarFault error
}

func (b *v41GroupedRaggedRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, shape, append([]float32(nil), values...))
	b.created[x.Buf()]++
	return x
}

func (b *v41GroupedRaggedRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || dt == compute.Q2_K || dt == compute.Q3_K
}

func (b *v41GroupedRaggedRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.uploads++
	if dt == compute.F32 {
		return b.tensor(x.Shape, b.Backend.Read(x))
	}
	packed := x.Buf().(compute.HostBuffer).I8()
	raw := make([]byte, len(packed))
	for i := range packed {
		raw[i] = byte(packed[i])
	}
	var out compute.Tensor
	switch dt {
	case compute.Q2_K:
		out = compute.NewQ2K(b, x.Shape, raw)
	case compute.Q3_K:
		out = compute.NewQ3K(b, x.Shape, raw)
	default:
		panic("unexpected ragged fixture weight dtype")
	}
	b.created[out.Buf()]++
	b.weights = append(b.weights, out)
	return out
}

func (b *v41GroupedRaggedRecorder) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	b.ops = append(b.ops, v41GroupedRaggedOp{"support", w.Dtype, rows})
	return w.Backend() == b && w.Dtype == compute.Q2_K && (rows == 2 || rows == 4 || rows == 8)
}

func (b *v41GroupedRaggedRecorder) SupportsV41Q3KTokenPanel(w compute.Tensor, rows int) bool {
	b.ops = append(b.ops, v41GroupedRaggedOp{"support", w.Dtype, rows})
	return rows != b.denyQ3Rows && w.Backend() == b && w.Dtype == compute.Q3_K && (rows == 2 || rows == 4 || rows == 8)
}

func (b *v41GroupedRaggedRecorder) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	return b.project(w, x, rows), nil
}

func (b *v41GroupedRaggedRecorder) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	return b.project(w, x, rows), nil
}

func (b *v41GroupedRaggedRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.project(w, x, rows)
}

func (b *v41GroupedRaggedRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.project(w, x, 1)
}

func (b *v41GroupedRaggedRecorder) project(w, x compute.Tensor, rows int) compute.Tensor {
	b.ops = append(b.ops, v41GroupedRaggedOp{"project", w.Dtype, rows})
	b.matmuls++
	var y compute.Tensor
	if rows == 1 {
		if b.scalarFault != nil {
			panic(b.scalarFault)
		}
		y = b.Backend.MatMul(w, x)
	} else {
		y = b.Backend.BatchedMatMul(w, x, rows)
	}
	return b.tensor(y.Shape, b.Backend.Read(y))
}

func (b *v41GroupedRaggedRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.ops = append(b.ops, v41GroupedRaggedOp{"publish", compute.F32, x.Numel()})
	return b.v41GroupedResidentRecorder.V41BF16PublishDeferred(x, site)
}

func (b *v41GroupedRaggedRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	b.gathers++
	rows, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	values := make([]float32, 0, rows*width)
	for _, span := range spans {
		x := b.Backend.Read(span.Src)
		values = append(values, x[span.Row*width:(span.Row+span.Rows)*width]...)
	}
	return b.tensor([]int{rows, width}, values), nil
}

func (b *v41GroupedRaggedRecorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	count := 0
	for _, row := range rows {
		count += row.Numel()
	}
	b.ops = append(b.ops, v41GroupedRaggedOp{"read", compute.F32, count})
	return b.v41GroupedResidentRecorder.V41ReadDeferred(st, rows)
}

func v41GroupedRaggedOwnership(t *testing.T, s *Session, b *v41GroupedRaggedRecorder) {
	t.Helper()
	if len(b.deferredPending) != 0 {
		t.Fatalf("unconsumed status records: %d", len(b.deferredPending))
	}
	cached := make(map[compute.Buffer]bool)
	for _, w := range b.weights {
		cached[w.Buf()] = true
	}
	if len(cached) != s.M.Cfg.OGroups+1 {
		t.Fatalf("cached weights=%d, want %d", len(cached), s.M.Cfg.OGroups+1)
	}
	s.Close()
	for buf, created := range b.created {
		wantFrees := 1
		if cached[buf] {
			wantFrees = 0
		}
		if created != 1 || b.freed[buf] != wantFrees {
			t.Fatalf("tensor ownership: created=%d freed=%d cached=%t", created, b.freed[buf], cached[buf])
		}
	}
	for buf := range b.freed {
		if b.created[buf] != 1 {
			t.Fatal("freed an unowned tensor")
		}
	}
	if err := s.M.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	for buf := range b.created {
		if b.freed[buf] != 1 {
			t.Fatalf("model close freed tensor %d times, want once", b.freed[buf])
		}
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41GroupedOutputPanelPackedRagged(t *testing.T) {
	for _, mode := range []string{"parity", "tail-support-refusal", "scalar-tail-failure"} {
		t.Run(mode, func(t *testing.T) {
			m := v41GroupedFixture(t, "Q2_K", true, false)
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			c := m.Cfg
			m.kqw[layerName(0, "attn.wo_b.weight")] = q3kFixtureTensor(c.HiddenSize, c.OGroups*c.OLoraRank)
			b := &v41GroupedRaggedRecorder{v41GroupedResidentRecorder: newV41GroupedResidentRecorder()}
			b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
			s := v41DenseTestSession(t, m, b)
			const rows = 11
			width, groupIn := c.NumHeads*c.HeadDim, c.NumHeads*c.HeadDim/c.OGroups
			input := v41GroupedPanelInput(rows, width)
			fault := errors.New("ragged scalar tail")
			if mode == "tail-support-refusal" {
				b.denyQ3Rows = 2
			} else if mode == "scalar-tail-failure" {
				b.scalarFault = &compute.BackendError{Backend: "ragged-recorder", Site: "MatMul", Err: fault}
			}
			got, outcome, err := s.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			var wantOps []v41GroupedRaggedOp
			for _, n := range []int{8, 2} {
				for g := 0; g < c.OGroups; g++ {
					wantOps = append(wantOps, v41GroupedRaggedOp{"support", compute.Q2_K, n})
				}
				wantOps = append(wantOps, v41GroupedRaggedOp{"support", compute.Q3_K, n})
			}
			if mode == "tail-support-refusal" {
				if got != nil || err != nil || outcome != v41ProjectionDeclined || s.BackendSessionClosed() || b.uploads != c.OGroups+1 || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 {
					t.Fatalf("ragged refusal outcome=%d err=%v closed=%t uploads=%d matmuls=%d gathers=%d publications=%d", outcome, err, s.BackendSessionClosed(), b.uploads, b.matmuls, b.gathers, b.deferredPublishes)
				}
			} else {
				for _, n := range []int{8, 2, 1} {
					if mode == "scalar-tail-failure" && n == 1 {
						wantOps = append(wantOps, v41GroupedRaggedOp{"publish", compute.F32, groupIn}, v41GroupedRaggedOp{"project", compute.Q2_K, 1}, v41GroupedRaggedOp{"read", compute.F32, 0})
						break
					}
					for g := 0; g < c.OGroups; g++ {
						wantOps = append(wantOps, v41GroupedRaggedOp{"publish", compute.F32, n * groupIn}, v41GroupedRaggedOp{"project", compute.Q2_K, n}, v41GroupedRaggedOp{"publish", compute.F32, n * c.OLoraRank})
					}
					wantOps = append(wantOps, v41GroupedRaggedOp{"project", compute.Q3_K, n}, v41GroupedRaggedOp{"publish", compute.F32, n * c.HiddenSize}, v41GroupedRaggedOp{"read", compute.F32, n * c.HiddenSize})
				}
				if mode == "scalar-tail-failure" {
					var selected *BackendForwardOperationError
					if got != nil || outcome != v41ProjectionError || !errors.As(err, &selected) || selected.Path != "v41-grouped-output-panel" || !errors.Is(err, fault) || !s.BackendSessionClosed() {
						t.Fatalf("scalar-tail failure outcome=%d err=%v closed=%t", outcome, err, s.BackendSessionClosed())
					}
				} else {
					if err != nil || outcome != v41ProjectionHandled || len(got) != rows {
						t.Fatalf("ragged panel rows=%d outcome=%d err=%v", len(got), outcome, err)
					}
					a, aOK := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
					bb, bOK := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
					if !aOK || !bOK {
						t.Fatal("packed scalar oracle weights absent")
					}
					for row := 0; row < rows; row++ {
						x := append([]float32(nil), input[row*width:(row+1)*width]...)
						for i := range x {
							x[i] = v41RoundBF16(x[i])
						}
						want, err := v41FullGroupedOutput(0, x, a, bb, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
						if err != nil {
							t.Fatal(err)
						}
						v41DeferredBitsEqual(t, "packed ragged row "+itoa(row), got[row], want)
					}
				}
			}
			if !reflect.DeepEqual(b.ops, wantOps) {
				t.Fatalf("ragged publication/projection/read order:\ngot  %v\nwant %v", b.ops, wantOps)
			}
			for _, site := range b.deferredSites {
				if site != compute.V41BF16PublishGroupedOutputProjection {
					t.Fatalf("publication site=%d", site)
				}
			}
			v41GroupedRaggedOwnership(t, s, b)
		})
	}
}
