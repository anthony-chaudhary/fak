package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// BatchedMatMul keeps the CPU recorder's result owned by the selected backend.
func (b *v41GroupedResidentRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.matmuls++
	b.lastWeight = w
	if b.matmuls == b.matmulFaultAt {
		panic(b.fault)
	}
	y := b.Backend.BatchedMatMul(w, x, rows)
	return b.tensor(y.Shape, b.Backend.Read(y))
}

type v41GroupedPanelQuantRecorder struct {
	*v41GroupedResidentRecorder
	q2Supports, q3Supports  int
	q2Calls, q3Calls        int
	denyQ3, earlyArithmetic bool
}

func newV41GroupedPanelQuantRecorder() *v41GroupedPanelQuantRecorder {
	b := &v41GroupedPanelQuantRecorder{v41GroupedResidentRecorder: newV41GroupedResidentRecorder()}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}

func (b *v41GroupedPanelQuantRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || dt == compute.Q2_K || dt == compute.Q3_K
}

func (b *v41GroupedPanelQuantRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, shape, append([]float32(nil), values...))
	b.created[x.Buf()]++
	return x
}

func (b *v41GroupedPanelQuantRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.uploads++
	if dt == compute.Q2_K || dt == compute.Q3_K {
		host, ok := x.Buf().(compute.HostBuffer)
		if !ok {
			panic("quant panel fixture requires host packed bytes")
		}
		src := host.I8()
		raw := make([]byte, len(src))
		for i := range src {
			raw[i] = byte(src[i])
		}
		var out compute.Tensor
		if dt == compute.Q2_K {
			out = compute.NewQ2K(b, x.Shape, raw)
		} else {
			out = compute.NewQ3K(b, x.Shape, raw)
		}
		b.created[out.Buf()]++
		return out
	}
	return b.tensor(x.Shape, b.Backend.Read(x))
}

func (b *v41GroupedPanelQuantRecorder) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	b.q2Supports++
	return w.Backend() == b && w.Dtype == compute.Q2_K && w.Layout == compute.RowMajor && len(w.Shape) == 2 && w.Shape[0] > 0 && w.Shape[1] > 0 && w.Shape[1]%256 == 0 && (rows == 2 || rows == 4 || rows == 8)
}

func (b *v41GroupedPanelQuantRecorder) SupportsV41Q3KTokenPanel(w compute.Tensor, rows int) bool {
	b.q3Supports++
	return !b.denyQ3 && w.Backend() == b && w.Dtype == compute.Q3_K && w.Layout == compute.RowMajor && len(w.Shape) == 2 && w.Shape[0] > 0 && w.Shape[1] > 0 && w.Shape[1]%256 == 0 && (rows == 2 || rows == 4 || rows == 8)
}

func (b *v41GroupedPanelQuantRecorder) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	if b.q2Supports == 0 || b.q3Supports == 0 {
		b.earlyArithmetic = true
	}
	b.q2Calls++
	return b.quantBatchedMatMul(w, x, rows), nil
}

func (b *v41GroupedPanelQuantRecorder) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	if b.q2Supports == 0 || b.q3Supports == 0 {
		b.earlyArithmetic = true
	}
	b.q3Calls++
	return b.quantBatchedMatMul(w, x, rows), nil
}

func (b *v41GroupedPanelQuantRecorder) quantBatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.matmuls++
	b.lastWeight = w
	y := b.Backend.BatchedMatMul(w, x, rows)
	return b.tensor(y.Shape, b.Backend.Read(y))
}

func (b *v41GroupedPanelQuantRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	switch w.Dtype {
	case compute.Q2_K:
		y, _ := b.V41Q2KTokenPanel(w, x, rows)
		return y
	case compute.Q3_K:
		y, _ := b.V41Q3KTokenPanel(w, x, rows)
		return y
	default:
		y := b.Backend.BatchedMatMul(w, x, rows)
		return b.tensor(y.Shape, b.Backend.Read(y))
	}
}

func (b *v41GroupedPanelQuantRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
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

func v41GroupedPanelQuantFixture(t *testing.T) (*Model, *Session, *v41GroupedPanelQuantRecorder) {
	t.Helper()
	m := v41GroupedFixture(t, "Q2_K", true, false)
	c := m.Cfg
	bName := layerName(0, "attn.wo_b.weight")
	m.kqw[bName] = q3kFixtureTensor(c.HiddenSize, c.OGroups*c.OLoraRank)
	b := newV41GroupedPanelQuantRecorder()
	s := v41DenseTestSession(t, m, b)
	return m, s, b
}

func v41GroupedPanelFixture(t *testing.T) (*Model, *Session, *v41GroupedResidentRecorder) {
	t.Helper()
	m := v41GroupedBF16Fixture(t)
	b := newV41GroupedResidentRecorder()
	s := v41DenseTestSession(t, m, b)
	return m, s, b
}

func v41GroupedPanelInput(rows, width int) []float32 {
	out := make([]float32, rows*width)
	for i := range out {
		out[i] = float32((i%13)-6) / 8
	}
	if len(out) > 3 {
		out[1] = math.Float32frombits(0x3f818000)
		out[3] = math.Float32frombits(0x80000000)
	}
	return out
}

// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputPanelParityAndInvocation(t *testing.T) {
	for _, rows := range []int{2, 4, 8, 10, 11} {
		t.Run(itoa(rows), func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			c := m.Cfg
			width := c.NumHeads * c.HeadDim
			input := v41GroupedPanelInput(rows, width)
			a := cpuOracleTensor(t, m, layerName(0, "attn.wo_a.weight"))
			bb := cpuOracleTensor(t, m, layerName(0, "attn.wo_b.weight"))
			want := make([][]float32, rows)
			for row := 0; row < rows; row++ {
				x := append([]float32(nil), input[row*width:(row+1)*width]...)
				for i := range x {
					x[i] = v41RoundBF16(x[i])
				}
				var err error
				want[row], err = v41FullGroupedOutput(0, x, a, bb, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
				if err != nil {
					t.Fatal(err)
				}
			}
			got, outcome, err := s.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
				t.Fatalf("rows=%d outcome=%d err=%v\ngot=%v\nwant=%v", rows, outcome, err, got, want)
			}
			wantCalls := c.OGroups + 1
			if b.matmuls != wantCalls || b.gathers != 1 || b.rowReads != 1 {
				t.Fatalf("rows=%d matmuls/gathers/reads=%d/%d/%d want %d/1/1", rows, b.matmuls, b.gathers, b.rowReads, wantCalls)
			}
			s.Close()
			live := 0
			for tensor, created := range b.created {
				live += created - b.freed[tensor]
			}
			if live != c.OGroups+1 {
				t.Fatalf("rows=%d live tensors=%d, want %d immutable cached weights", rows, live, c.OGroups+1)
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedOutputPanelPackedColdSelection(t *testing.T) {
	m, s, b := v41GroupedPanelQuantFixture(t)
	c := m.Cfg
	rows, width := 8, c.NumHeads*c.HeadDim
	input := v41GroupedPanelInput(rows, width)
	a, aOK := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
	bb, bOK := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
	if !aOK || !bOK {
		t.Fatal("packed oracle weights absent")
	}
	want := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		x := append([]float32(nil), input[row*width:(row+1)*width]...)
		for i := range x {
			x[i] = v41RoundBF16(x[i])
		}
		var err error
		want[row], err = v41FullGroupedOutput(0, x, a, bb, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(s.halW) != 0 {
		t.Fatal("cold session unexpectedly has staged grouped weights")
	}
	got, outcome, err := s.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
	if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
		t.Fatalf("cold packed outcome=%d err=%v parity=%t", outcome, err, reflect.DeepEqual(got, want))
	}
	if b.uploads != 2*c.OGroups+1 || b.q2Supports != c.OGroups || b.q3Supports != 1 || b.q2Calls != c.OGroups || b.q3Calls != 1 || b.earlyArithmetic {
		t.Fatalf("cold packed uploads=%d supports=%d/%d calls=%d/%d early=%t", b.uploads, b.q2Supports, b.q3Supports, b.q2Calls, b.q3Calls, b.earlyArithmetic)
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedOutputPanelPackedSupportRefusal(t *testing.T) {
	m, s, b := v41GroupedPanelQuantFixture(t)
	b.denyQ3 = true
	c := m.Cfg
	rows, width := 8, c.NumHeads*c.HeadDim
	got, outcome, err := s.v41GroupedOutputPanel(0, v41GroupedPanelInput(rows, width), rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
	if got != nil || err != nil || outcome != v41ProjectionDeclined {
		t.Fatalf("packed refusal got=%v outcome=%d err=%v", got, outcome, err)
	}
	if b.uploads != c.OGroups+1 || b.q2Calls != 0 || b.q3Calls != 0 || b.gathers != 0 || b.deferredPublishes != 0 {
		t.Fatalf("packed refusal staging=%d arithmetic=%d/%d gathers=%d publishes=%d", b.uploads, b.q2Calls, b.q3Calls, b.gathers, b.deferredPublishes)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputPanelDeclinesBeforeActivation(t *testing.T) {
	m, s, b := v41GroupedPanelFixture(t)
	b.deny = true
	c := m.Cfg
	rows, width := 8, c.NumHeads*c.HeadDim
	got, outcome, err := s.v41GroupedOutputPanel(0, v41GroupedPanelInput(rows, width), rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
	if got != nil || err != nil || outcome != v41ProjectionDeclined || b.uploads != 0 || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 {
		t.Fatalf("decline got=%v outcome=%d err=%v effects=%d/%d/%d/%d", got, outcome, err, b.uploads, b.matmuls, b.gathers, b.deferredPublishes)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputPanelDeferredDisabledBeforeActivation(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "0")
	m, s, b := v41GroupedPanelFixture(t)
	c := m.Cfg
	rows, width := 4, c.NumHeads*c.HeadDim
	got, outcome, err := s.v41GroupedOutputPanel(0, v41GroupedPanelInput(rows, width), rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
	if got != nil || err != nil || outcome != v41ProjectionDeclined {
		t.Fatalf("disabled deferred publication got=%v outcome=%d err=%v", got, outcome, err)
	}
	if b.uploads != 0 || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 || len(b.created) != 0 || s.BackendSessionClosed() {
		t.Fatalf("disabled panel effects=%d/%d/%d/%d created=%d closed=%t", b.uploads, b.matmuls, b.gathers, b.deferredPublishes, len(b.created), s.BackendSessionClosed())
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputPanelFaultOrder(t *testing.T) {
	inputFault := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishGroupedOutputProjection, Index: 7}
	laterA := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishGroupedOutputProjection, Index: 3}
	earlierB := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishGroupedOutputProjection, Index: 0}
	meta := []v41GroupedPanelStatus{
		{stage: 0, group: 1, width: 4},
		{stage: 2, group: 1, width: 2},
		{stage: 3, group: -1, width: 3},
	}
	if got := v41GroupedPanelFirstFault([]error{inputFault, laterA, earlierB}, meta, 2, 2); got != earlierB {
		t.Fatalf("fault order chose %v, want earlier token B %v", got, earlierB)
	}
	inputEarlier := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishGroupedOutputProjection, Index: 0}
	if got := v41GroupedPanelFirstFault([]error{inputEarlier, nil, earlierB}, meta, 2, 2); got != inputEarlier {
		t.Fatalf("fault order chose %v, want input %v", got, inputEarlier)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputPanelSelectedFailureClosesAndCleans(t *testing.T) {
	m, s, b := v41GroupedPanelFixture(t)
	c := m.Cfg
	rows, width := 4, c.NumHeads*c.HeadDim
	b.deferredReadErr = errors.New("panel transport")
	got, outcome, err := s.v41GroupedOutputPanel(0, v41GroupedPanelInput(rows, width), rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
	var selected *BackendForwardOperationError
	if got != nil || outcome != v41ProjectionError || !errors.As(err, &selected) || selected.Path != "v41-grouped-output-panel" || !s.BackendSessionClosed() {
		t.Fatalf("selected failure got=%v outcome=%d err=%v closed=%t", got, outcome, err, s.BackendSessionClosed())
	}
	if len(b.deferredPending) != 0 {
		t.Fatalf("selected failure left %d status records", len(b.deferredPending))
	}
	live := 0
	for tensor, created := range b.created {
		live += created - b.freed[tensor]
	}
	if live != c.OGroups+1 {
		t.Fatalf("live tensors=%d, want %d immutable cached weights", live, c.OGroups+1)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41GroupedOutputPanelMalformedPublicationOwnership(t *testing.T) {
	for _, malformed := range []string{"publication-partial", "nil-output", "aliased-output", "record-output", "record-input", "record-weight", "record-prior"} {
		t.Run(malformed, func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			b.malformed, b.fault = malformed, errors.New("partial publication")
			c := m.Cfg
			rows, width := 4, c.NumHeads*c.HeadDim
			got, outcome, err := s.v41GroupedOutputPanel(0, v41GroupedPanelInput(rows, width), rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			var selected *BackendForwardOperationError
			if got != nil || outcome != v41ProjectionError || !errors.As(err, &selected) || !s.BackendSessionClosed() {
				t.Fatalf("malformed=%s got=%v outcome=%d err=%v closed=%t", malformed, got, outcome, err, s.BackendSessionClosed())
			}
			if len(b.deferredPending) != 0 {
				t.Fatalf("malformed=%s left %d status records", malformed, len(b.deferredPending))
			}
			for key, weight := range s.halW {
				if b.freed[weight.Buf()] != 0 {
					t.Fatalf("malformed=%s freed cached weight %s", malformed, key)
				}
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedOutputPanelForwardReachability(t *testing.T) {
	m := v41GroupedBF16Fixture(t)
	b := newV41GroupedResidentRecorder()
	s := v41DenseTestSession(t, m, b)
	ids := []int{1, 2, 3, 4}
	before := v41GroupedPhase(t, m, "prefill")
	got := s.Prefill(ids)
	want := lastLogits(m.Forward(ids))
	if len(got) != len(want) || b.gathers == 0 {
		t.Fatalf("panel forward was not reached: logits=%d/%d gathers=%d err=%v", len(got), len(want), b.gathers, s.halFailure)
	}
	v41GroupedParity(t, got, want, 1e-4)
	delta := v41DenseTestDelta(v41GroupedPhase(t, m, "prefill"), before)
	if delta["grouped_output_device_calls"] != 1 || delta["grouped_output_device_rows"] != float64(len(ids)) || delta["grouped_output_matmul_calls"] != float64(m.Cfg.OGroups+1) || delta["grouped_output_activation_upload_bytes"] != float64(4*len(ids)*m.Cfg.NumHeads*m.Cfg.HeadDim) || delta["grouped_output_readback_bytes"] != float64(4*len(ids)*m.Cfg.HiddenSize) || delta["grouped_output_nanos"] <= 0 {
		t.Fatalf("panel ledger=%v", delta)
	}
}
