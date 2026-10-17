package model

import (
	"context"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"reflect"
	"sync"
	"testing"
)

// CPU-backed protocol recorder, not GPU qualification.
type v41GroupedDecodeRecorder struct {
	scalarOutputWeights    [][]float32
	scalarOutputCalls      int
	freeFault              error
	badWeight              string
	allowQ2, declineWeight bool
	owner                  sync.Mutex
	batchedRows            []int
	afterMatMul            func()

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

func newV41GroupedDecodeRecorder() *v41GroupedDecodeRecorder {
	b := &v41GroupedDecodeRecorder{Backend: compute.Default(), created: map[compute.Buffer]int{}, freed: map[compute.Buffer]int{}}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41GroupedDecodeRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41GroupedDecodeRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || (b.allowQ2 && dt == compute.Q2_K)
}
func (b *v41GroupedDecodeRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, shape, append([]float32(nil), values...))
	b.created[x.Buf()]++
	return x
}
func (b *v41GroupedDecodeRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.uploads++
	if dt == compute.Q2_K {
		raw := x.Buf().(compute.HostBuffer).I8()
		data := make([]byte, len(raw))
		for i, v := range raw {
			data[i] = byte(v)
		}
		out := compute.NewQ2K(b, x.Shape, data)
		b.created[out.Buf()]++
		return out
	}
	out := b.tensor(x.Shape, b.Backend.Read(x))
	if b.uploads == 1 {
		switch b.badWeight {
		case "shape":
			out.Shape = []int{1}
		case "dtype":
			out.Dtype = compute.F16
		case "backend":
			b.Free(out)
			out = compute.NewF32(b.Backend, x.Shape, append([]float32(nil), b.Backend.Read(x)...))
			b.created[out.Buf()]++
		}
	}
	return out
}
func (b *v41GroupedDecodeRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	for _, weight := range b.scalarOutputWeights {
		if reflect.DeepEqual(b.Backend.Read(w), weight) {
			b.scalarOutputCalls++
			break
		}
	}
	b.matmuls++
	b.lastWeight = w
	if b.matmuls == b.matmulFaultAt {
		panic(b.fault)
	}
	y := b.Backend.MatMul(w, x)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41GroupedDecodeRecorder) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(x)
}
func (b *v41GroupedDecodeRecorder) Free(x compute.Tensor) {
	b.freed[x.Buf()]++
	clear(b.Backend.Read(x))
	delete(b.deferredPending, x.Buf())
	b.Backend.Free(x)
	if b.freeFault != nil {
		panic(b.freeFault)
	}
}
func (b *v41GroupedDecodeRecorder) SupportsV41BF16Publish(n int) bool { return n > 0 }
func (b *v41GroupedDecodeRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
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
func (b *v41GroupedDecodeRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
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
func (b *v41GroupedDecodeRecorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.rowReads += len(rows)
	return b.v41DeferredLedger.V41ReadDeferred(st, rows)
}
func (b *v41GroupedDecodeRecorder) V41RowLogAvailable() bool { return !b.deny }
func (b *v41GroupedDecodeRecorder) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	return b.tensor([]int{rows, width}, make([]float32, rows*width)), nil
}
func (b *v41GroupedDecodeRecorder) V41RowLogCopy(compute.Tensor, int, compute.Tensor, int, int) error {
	panic("resident grouped output needs gather only")
}
func (b *v41GroupedDecodeRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
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

func (b *v41GroupedDecodeRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41GroupedDecodeRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned cohort panel dispatch")
	}
	b.batchedRows = append(b.batchedRows, rows)
	b.matmuls++
	b.lastWeight = w
	if b.matmuls == b.matmulFaultAt {
		panic(b.fault)
	}
	y := b.Backend.BatchedMatMul(w, x, rows)
	if b.afterMatMul != nil {
		b.afterMatMul()
	}
	return b.tensor(y.Shape, b.Backend.Read(y))
}

func v41GroupedDecodeFixture(t *testing.T, m *Model, count int) (*v41GroupedDecodeRecorder, []v41StepCohortLane, []*Session) {
	t.Helper()
	b := newV41GroupedDecodeRecorder()
	lanes := make([]v41StepCohortLane, count)
	refs := make([]*Session, count)
	for i := range lanes {
		prefix := []int{1, 2}
		if i%2 != 0 {
			prefix = append(prefix, 3)
		}
		seed := v41IncrementalState(t, m, prefix)
		snapshot := captureV41ForwardSnapshot(seed)
		s, r := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
		s.v41Forward, r.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
		s.v41State()
		r.v41State()
		lanes[i] = v41StepCohortLane{session: s, id: 4 + i}
		refs[i] = r
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		b.scalarOutputWeights = append(b.scalarOutputWeights, cpuOracleTensor(t, m, layerName(l, "attn.wo_b.weight")))
	}
	return b, lanes, refs
}

// fak-test:runtime medium est=10s lane=default
func TestV41GroupedDecodeIntegrationParity(t *testing.T) {
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, role := range []string{"plain", "source", "reader"} {
		for _, count := range []int{1, 2, 3, 4} {
			t.Run(fmt.Sprintf("role=%s/rows=%d", role, count), func(t *testing.T) {
				var m *Model
				if role == "reader" {
					m = v41ReaderWidthFixture(t)
				} else {
					m = v41GroupedFixture(t, "F32", true, false)
					if role == "plain" {
						m.Cfg.DeepSeekV41.CompressRatios = []int{0}
						m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
						m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
					}

				}
				defer m.CloseWeights()
				b, lanes, refs := v41GroupedDecodeFixture(t, m, count)
				if count == 1 {
					before := captureV41ForwardSnapshot(lanes[0].session.v41Forward)
					t.Setenv(v41GroupedDecodeEnv, "1")
					got, err := m.forwardV41StepCohort(&b.owner, lanes)
					if !errors.Is(err, errV41StepCohortDeclined) || len(got) != 0 || len(b.batchedRows) != 0 || b.scalarOutputCalls != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].session.v41Forward)) {
						t.Fatalf("single-lane admission must refuse without mutation: %v", err)
					}
					return
				}
				want := make([][]float32, count)
				for i, r := range refs {
					var err error
					want[i], _, err = m.forwardV41Step(lanes[i].id, r.v41State(), nil)
					if err != nil {
						t.Fatal(err)
					}
				}
				b.batchedRows = nil
				b.rowReads = 0
				b.scalarOutputCalls = 0
				t.Setenv(v41GroupedDecodeEnv, "1")
				got, err := m.forwardV41StepCohort(&b.owner, lanes)
				if err != nil {
					t.Fatal(err)
				}
				for i, result := range got {
					if result.err != nil || !result.stats.Committed || !reflect.DeepEqual(result.logits, want[i]) || !reflect.DeepEqual(captureV41ForwardSnapshot(lanes[i].session.v41Forward), captureV41ForwardSnapshot(refs[i].v41Forward)) {
						t.Fatalf("lane %d scalar parity/commit differs: %v", i, result.err)
					}
					if lanes[i].session.v41State().routedExpert != nil {
						t.Fatal("fixture unexpectedly provides routed experts")
					}
				}
				if count == 2 || count == 4 {
					if len(b.batchedRows) == 0 || b.rowReads == 0 {
						t.Fatal("real grouped operations were not reached")
					}
					for _, n := range b.batchedRows {
						if n != count {
							t.Fatalf("actual BatchedMatMul rows=%d want=%d", n, count)
						}
					}
				} else if len(b.batchedRows) != 0 {
					t.Fatalf("unsupported cohort width selected panel: %v", b.batchedRows)
				}
			})
		}
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedDecodeIntegrationGateAndDecline(t *testing.T) {
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, gate := range []string{"", "0", "true", "1"} {
		t.Run("gate="+gate, func(t *testing.T) {
			t.Setenv(v41GroupedDecodeEnv, gate)
			m := v41GroupedFixture(t, "F32", true, false)
			defer m.CloseWeights()
			b, lanes, _ := v41GroupedDecodeFixture(t, m, 2)
			b.deny = gate == "1"

			got, err := m.forwardV41StepCohort(&b.owner, lanes)
			if err != nil {
				t.Fatal(err)
			}
			if len(b.batchedRows) != 0 {
				t.Fatal("disabled/refused panel selected arithmetic")
			}
			for i, result := range got {
				if !result.stats.Committed || b.scalarOutputCalls != 2*m.Cfg.NumLayers {
					t.Fatalf("lane %d commit=%t scalar=%d", i, result.stats.Committed, b.scalarOutputCalls)
				}
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedDecodeIntegrationCancellation(t *testing.T) {
	t.Setenv(v41GroupedDecodeEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, count := range []int{2, 4} {
		for _, duringPanel := range []bool{false, true} {
			t.Run(fmt.Sprintf("rows=%d/panel=%t", count, duringPanel), func(t *testing.T) {
				m := v41GroupedFixture(t, "F32", true, false)
				defer m.CloseWeights()
				b, lanes, _ := v41GroupedDecodeFixture(t, m, count)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				lanes[0].ctx = ctx
				before := captureV41ForwardSnapshot(lanes[0].session.v41Forward)

				if duringPanel {
					b.afterMatMul = cancel
				} else {
					st := lanes[count-1].session.v41State()
					original := st.denseProjection()
					st.setDenseProjection(func(l int, leaf string, x []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
						y, o, e := original(l, leaf, x, out, in, rows)
						if leaf == "attn.wq_a.weight" {
							cancel()
						}
						return y, o, e
					})
				}
				got, err := m.forwardV41StepCohort(&b.owner, lanes)
				if err != nil {
					t.Fatal(err)
				}
				if !errors.Is(got[0].err, context.Canceled) || got[0].stats.Committed || !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].session.v41Forward)) {
					t.Fatal("canceled lane escaped rollback/commit boundary")
				}
				for i := 1; i < count; i++ {
					if !got[i].stats.Committed {
						t.Fatalf("survivor %d did not commit", i)
					}
				}
				if duringPanel {
					if len(b.batchedRows) == 0 {
						t.Fatal("cancellation did not occur in selected panel")
					}
				} else {
					if len(b.batchedRows) != 0 {
						t.Fatal("1/3 survivors selected panel")
					}
					if b.scalarOutputCalls != (count-1)*m.Cfg.NumLayers {
						t.Fatalf("survivor scalar outputs=%d want=%d", b.scalarOutputCalls, (count-1)*m.Cfg.NumLayers)
					}
				}
			})
		}
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedDecodeIntegrationSelectedFailure(t *testing.T) {
	t.Setenv(v41GroupedDecodeEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, count := range []int{2, 4} {
		m := v41GroupedFixture(t, "F32", true, false)
		b, lanes, _ := v41GroupedDecodeFixture(t, m, count)
		fault := errors.New("selected grouped decode fault")
		b.afterMatMul = func() { panic(fault) }

		var recovered any
		func() { defer func() { recovered = recover() }(); _, _ = m.forwardV41StepCohort(&b.owner, lanes) }()
		if recovered != fault {
			t.Fatalf("primary panic=%v", recovered)
		}
		if b.scalarOutputCalls != 0 || len(b.batchedRows) != 1 {
			t.Fatalf("selected failure replay: scalar=%d rows=%v", b.scalarOutputCalls, b.batchedRows)
		}
		for _, lane := range lanes {
			if !lane.session.halClosed || lane.session.halFailure == nil {
				t.Fatal("selected peer not terminal")
			}
		}
		m.CloseWeights()
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedDecodeIntegrationOverrideRefusal(t *testing.T) {
	t.Setenv(v41GroupedDecodeEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, kind := range []string{"grouped-output", "grouped-error"} {
		t.Run(kind, func(t *testing.T) {
			m := v41GroupedFixture(t, "F32", true, false)
			defer m.CloseWeights()
			b, lanes, _ := v41GroupedDecodeFixture(t, m, 2)
			calls := 0
			fault := errors.New("custom scalar projection semantics")
			for _, lane := range lanes {
				st := lane.session.v41State()
				st.groupedOutput = func(_ int, _ []float32, _ int, _ int, _ int, _ int, dim int) ([]float32, v41DenseProjectionOutcome, error) {
					calls++
					if kind == "grouped-error" {
						return nil, v41ProjectionError, fault
					}
					return make([]float32, dim), v41ProjectionHandled, nil
				}
			}
			var got []v41StepCohortResult
			var err error
			var caught any
			func() { defer func() { caught = recover() }(); got, err = m.forwardV41StepCohort(&b.owner, lanes) }()
			if len(b.batchedRows) != 0 || calls == 0 {
				t.Fatalf("override bypassed: calls=%d batch=%v", calls, b.batchedRows)
			}
			if kind == "grouped-output" {
				if caught != nil || err != nil || len(got) != 2 || !got[0].stats.Committed || !got[1].stats.Committed {
					t.Fatalf("custom output behavior lost: %v %v", caught, err)
				}
			} else {
				cause := err
				if e, ok := caught.(error); ok {
					cause = e
				}
				if !errors.Is(cause, fault) {
					t.Fatalf("custom failure identity lost: %v %v", caught, err)
				}
			}
		})
	}
}

// fak-test:runtime fast est=2s lane=default
func TestV41GroupedDecodeIntegrationProvenanceChange(t *testing.T) {
	t.Setenv(v41GroupedDecodeEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, publication := range []bool{false, true} {
		m := v41GroupedFixture(t, "F32", true, false)
		defer m.CloseWeights()
		b, lanes, _ := v41GroupedDecodeFixture(t, m, 2)
		b.afterMatMul = func() {
			st := lanes[1].session.v41Forward
			if publication {
				st.gatherOps.setBF16Publication(nil)
			} else {
				st.groupedOutput = nil
			}
		}
		got, err := m.forwardV41StepCohort(&b.owner, lanes)
		if err == nil || len(b.batchedRows) == 0 || b.scalarOutputCalls != 0 {
			t.Fatalf("changed provenance escaped: err=%v batch=%v scalar=%d", err, b.batchedRows, b.scalarOutputCalls)
		}
		for i, lane := range lanes {
			if !lane.session.halClosed || lane.session.halFailure == nil || got[i].stats.Committed || got[i].logits != nil {
				t.Fatal("changed provenance published/committed or left peer live")
			}
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41GroupedDecodeIntegrationPublicationOverride(t *testing.T) {
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "0")
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	m := v41GroupedFixture(t, "F32", true, false)
	defer m.CloseWeights()
	fault := errors.New("custom non-BF16 projection boundary")
	var baseline []v41StepCohortResult
	var baselineStates []*v41ForwardSnapshot
	baselineCalls := 0
	for _, gate := range []string{"0", "1"} {
		t.Setenv(v41GroupedDecodeEnv, gate)
		b, lanes, _ := v41GroupedDecodeFixture(t, m, 2)
		calls := 0
		for _, lane := range lanes {
			st := lane.session.v41State()
			original := st.gatherOps.bf16PublishFunc()
			st.gatherOps.setBF16Publication(func(l int, site string, x []float32) ([]float32, error) {
				if site == "attn.wo_a.weight input" {
					calls++
					return nil, fault
				}
				return original(l, site, x)
			})
		}
		got, err := m.forwardV41StepCohort(&b.owner, lanes)
		if err != nil || len(got) != 2 || len(b.batchedRows) != 0 {
			t.Fatalf("custom publication route: %v grouped=%v", err, b.batchedRows)
		}
		states := make([]*v41ForwardSnapshot, 2)
		for i, r := range got {
			if r.err != nil || !r.stats.Committed {
				t.Fatalf("lane %d: %v", i, r.err)
			}
			states[i] = captureV41ForwardSnapshot(lanes[i].session.v41Forward)
		}
		if gate == "0" {
			baseline, baselineStates, baselineCalls = got, states, calls
		} else {
			if calls != baselineCalls || !reflect.DeepEqual(states, baselineStates) {
				t.Fatal("publication override state/call parity differs from gate-off")
			}
			for i := range got {
				if !reflect.DeepEqual(got[i].logits, baseline[i].logits) {
					t.Fatal("publication override logits differ from gate-off")
				}
			}
		}
		// Already-widened attention input legitimately elides ordinary publication
		// in v41PublishBF16Input. A non-BF16 controlled input makes the same captured
		// projector boundary eligible and must preserve its selected custom error.
		scratch := &v41ProjScratch{}
		scratch.bindV41StepState(lanes[0].session.v41State())
		cfg := m.Cfg
		project := m.v41GroupedOutputProjector(0, cfg.NumHeads, cfg.HeadDim, cfg.OGroups, cfg.OLoraRank, cfg.HiddenSize, scratch)
		input := make([]float32, cfg.NumHeads*cfg.HeadDim)
		for i := range input {
			input[i] = 0.1
		}
		before := calls
		if _, err := project(input); !errors.Is(err, fault) || calls != before+1 {
			t.Fatalf("eligible publication callback lost: calls=%d error=%v", calls-before, err)
		}
	}
}
