package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU-backed protocol recorder, not GPU qualification.
type v41CohortPanelRecorder struct {
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

func newV41CohortPanelRecorder() *v41CohortPanelRecorder {
	b := &v41CohortPanelRecorder{Backend: compute.Default(), created: map[compute.Buffer]int{}, freed: map[compute.Buffer]int{}}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41CohortPanelRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41CohortPanelRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || (b.allowQ2 && dt == compute.Q2_K)
}
func (b *v41CohortPanelRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, shape, append([]float32(nil), values...))
	b.created[x.Buf()]++
	return x
}
func (b *v41CohortPanelRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
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
func (b *v41CohortPanelRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	b.lastWeight = w
	if b.matmuls == b.matmulFaultAt {
		panic(b.fault)
	}
	y := b.Backend.MatMul(w, x)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41CohortPanelRecorder) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(x)
}
func (b *v41CohortPanelRecorder) Free(x compute.Tensor) {
	b.freed[x.Buf()]++
	clear(b.Backend.Read(x))
	delete(b.deferredPending, x.Buf())
	b.Backend.Free(x)
	if b.freeFault != nil {
		panic(b.freeFault)
	}
}
func (b *v41CohortPanelRecorder) SupportsV41BF16Publish(n int) bool { return n > 0 }
func (b *v41CohortPanelRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
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
func (b *v41CohortPanelRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
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
func (b *v41CohortPanelRecorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.rowReads += len(rows)
	return b.v41DeferredLedger.V41ReadDeferred(st, rows)
}
func (b *v41CohortPanelRecorder) V41RowLogAvailable() bool { return !b.deny }
func (b *v41CohortPanelRecorder) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	return b.tensor([]int{rows, width}, make([]float32, rows*width)), nil
}
func (b *v41CohortPanelRecorder) V41RowLogCopy(compute.Tensor, int, compute.Tensor, int, int) error {
	panic("resident grouped output needs gather only")
}
func (b *v41CohortPanelRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
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

func (b *v41CohortPanelRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41CohortPanelRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
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

func v41CohortPanelFixture(t *testing.T, n int) (*Model, *v41CohortPanelRecorder, []v41GroupedOutputCohortRow) {
	t.Helper()
	m := v41GroupedBF16Fixture(t)
	b := newV41CohortPanelRecorder()
	rows := make([]v41GroupedOutputCohortRow, n)
	width := m.Cfg.NumHeads * m.Cfg.HeadDim
	input := v41GroupedPanelInput(n, width)
	for i := range rows {
		rows[i] = v41GroupedOutputCohortRow{Lane: i * 2, Session: v41DenseTestSession(t, m, b), Input: input[i*width : (i+1)*width]}
	}
	return m, b, rows
}

// fak-test:runtime fast est=1s lane=default
func TestV41GroupedOutputCohortParity(t *testing.T) {
	for _, n := range []int{2, 4} {
		m, b, rows := v41CohortPanelFixture(t, n)
		scope := beginBackendExecution(b)
		func() {
			defer scope.finish()
			for _, row := range rows {
				scope.bind(row.Session)
			}
			owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
			if err != nil || outcome != v41ProjectionHandled {
				t.Fatalf("prepare=%v %v", outcome, err)
			}
			cfg := m.Cfg
			a := cpuOracleTensor(t, m, layerName(0, "attn.wo_a.weight"))
			bb := cpuOracleTensor(t, m, layerName(0, "attn.wo_b.weight"))
			want := make([][]float32, n)
			for i, row := range rows {
				input := append([]float32(nil), row.Input...)
				for j := range input {
					input[j] = v41RoundBF16(input[j])
				}
				want[i], err = v41FullGroupedOutput(0, input, a, bb, cfg.NumHeads, cfg.HeadDim, cfg.OGroups, cfg.OLoraRank, cfg.HiddenSize)
				if err != nil {
					t.Fatal(err)
				}
			}
			got, outcome, err := owner.Run()
			if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
				t.Fatalf("rows=%d outcome=%v err=%v parity=%t", n, outcome, err, reflect.DeepEqual(got, want))
			}
			if len(b.batchedRows) != cfg.OGroups+1 || b.rowReads != 1 {
				t.Fatalf("actual dispatches=%v reads=%d", b.batchedRows, b.rowReads)
			}
			for _, count := range b.batchedRows {
				if count != n {
					t.Fatalf("actual BatchedMatMul rows=%d want=%d", count, n)
				}
			}
			for _, row := range rows {
				if row.Session.halClosed || row.Session.halFailure != nil {
					t.Fatal("successful peer retired")
				}
			}
			before := b.matmuls
			if _, _, err = owner.Run(); err == nil || b.matmuls != before {
				t.Fatal("cohort replay accepted")
			}
			// Tile Free poisons its borrowed buffers; successful rows remain owned.
			if !reflect.DeepEqual(got, want) {
				t.Fatal("output aliases freed tile")
			}
		}()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputCohortRefusals(t *testing.T) {
	for _, kind := range []string{"duplicate", "unordered", "shape", "stale", "restore", "abandon"} {
		_, b, rows := v41CohortPanelFixture(t, 2)
		scope := beginBackendExecution(b)
		func() {
			defer scope.finish()
			for _, row := range rows {
				scope.bind(row.Session)
			}
			switch kind {
			case "duplicate":
				rows[1].Session = rows[0].Session
			case "unordered":
				rows[1].Lane = rows[0].Lane
			case "shape":
				rows[1].Input = rows[1].Input[:1]
			}
			before := b.uploads
			owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
			if kind == "stale" || kind == "restore" || kind == "abandon" {
				if err != nil || outcome != v41ProjectionHandled {
					t.Fatal("valid prepare refused")
				}
				if kind == "stale" || kind == "restore" {
					if kind == "stale" {
						rows[1].Session.cacheGeometryEpoch++
					} else {
						rows[1].Session.v41Forward = &v41ForwardState{}
					}
					_, outcome, err = owner.Run()
					if err != nil || outcome != v41ProjectionDeclined {
						t.Fatal("stale admission dispatched")
					}
				} else {
					owner.close()
					if _, _, err = owner.Run(); err == nil {
						t.Fatal("abandoned owner ran")
					}
				}
			} else if err != nil || owner != nil || outcome != v41ProjectionDeclined {
				t.Fatal("invalid membership accepted")
			}
			if b.uploads != before || b.matmuls != 0 {
				t.Fatal("refusal touched backend")
			}
			for _, row := range rows {
				if row.Session.halClosed {
					t.Fatal("refusal retired peer")
				}
			}
		}()
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestV41GroupedOutputCohortSelectedFailures(t *testing.T) {
	for _, kind := range []string{"matmul", "cleanup", "status", "epoch", "timing"} {
		m, b, rows := v41CohortPanelFixture(t, 4)
		scope := beginBackendExecution(b)
		func() {
			defer scope.finish()
			for _, row := range rows {
				scope.bind(row.Session)
			}
			owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
			if err != nil || outcome != v41ProjectionHandled {
				t.Fatal("valid prepare refused")
			}
			primary := errors.New("selected backend fault")
			switch kind {
			case "matmul", "cleanup":
				if kind == "cleanup" {
					b.freeFault = errors.New("secondary cleanup")
				}
				b.matmulFaultAt = 1
				b.fault = primary
			case "status":
				b.malformed = "publication-partial"
				b.fault = primary
			case "timing":
				m.v41SetExpertTimeClock(func() int64 {
					if b.uploads > 0 {
						panic(primary)
					}
					return 1
				})
			case "epoch":
				b.afterMatMul = func() { rows[2].Session.cacheGeometryEpoch++ }
			}
			var got [][]float32
			var caught any
			func() { defer func() { caught = recover() }(); got, outcome, err = owner.Run() }()
			if kind == "matmul" || kind == "cleanup" || kind == "timing" {
				if caught != primary {
					t.Fatalf("primary panic=%v", caught)
				}
			} else if err == nil || caught != nil {
				t.Fatalf("selected error=%v panic=%v", err, caught)
			}
			if kind == "timing" && (!owner.selected || owner.arithmeticSelected || b.uploads != m.Cfg.OGroups+1) {
				t.Fatal("timing fault did not hit post-weight gap")
			}
			if got != nil {
				t.Fatal("selected failure published output")
			}
			for i, row := range rows {
				if !row.Session.halClosed || row.Session.halFailure == nil {
					t.Fatalf("lane%d not terminal", i)
				}
			}
			before := b.matmuls
			if _, _, err = owner.Run(); err == nil || b.matmuls != before {
				t.Fatal("failed cohort replayed")
			}
			if len(b.deferredPending) != 0 {
				t.Fatal("selected failure left deferred work")
			}
			for tensor, n := range b.freed {
				if n > 1 {
					t.Fatalf("tensor %v freed %d times", tensor, n)
				}
			}
		}()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputCohortOverflowRefusal(t *testing.T) {
	for _, kind := range []string{"panel-elements", "panel-bytes", "joined", "output-bytes"} {
		m, b, rows := v41CohortPanelFixture(t, 4)
		scope := beginBackendExecution(b)
		func() {
			defer scope.finish()
			for _, row := range rows {
				scope.bind(row.Session)
			}
			before := b.uploads
			maxInt := int(^uint(0) >> 1)
			switch kind {
			case "panel-elements":
				m.Cfg.NumHeads = 1
				m.Cfg.HeadDim = maxInt/2 + 1
			case "panel-bytes":
				m.Cfg.NumHeads = 1
				m.Cfg.HeadDim = maxInt/8 + 1
			case "joined":
				m.Cfg.OGroups = maxInt/2 + 1
				m.Cfg.NumHeads = m.Cfg.OGroups
				m.Cfg.HeadDim = 1
				m.Cfg.OLoraRank = 3
			case "output-bytes":
				m.Cfg.HiddenSize = maxInt/8 + 1
			}
			owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
			if err != nil || owner != nil || outcome != v41ProjectionDeclined || b.uploads != before || b.matmuls != 0 {
				t.Fatalf("%s overflow touched backend or admitted", kind)
			}
		}()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputCohortBorrowedWeightIdentity(t *testing.T) {
	for _, kind := range []string{"backend", "dtype", "shape"} {
		_, b, rows := v41CohortPanelFixture(t, 2)
		scope := beginBackendExecution(b)
		func() {
			defer scope.finish()
			for _, row := range rows {
				scope.bind(row.Session)
			}
			owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
			if err != nil || outcome != v41ProjectionHandled {
				t.Fatal("valid cohort refused")
			}
			b.badWeight = kind
			got, outcome, err := owner.Run()
			if got != nil || err == nil || outcome != v41ProjectionError || b.matmuls != 0 {
				t.Fatalf("%s mismatched handle reached arithmetic", kind)
			}
			for _, row := range rows {
				if !row.Session.halClosed || row.Session.halFailure == nil {
					t.Fatalf("%s mismatch did not retire all peers", kind)
				}
			}
		}()
	}
}

func (b *v41CohortPanelRecorder) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	return b.allowQ2 && !b.declineWeight
}
func (b *v41CohortPanelRecorder) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	return b.BatchedMatMul(w, x, rows), nil
}

// fak-test:runtime fast est=200ms lane=default
func TestV41GroupedOutputCohortPostWeightDecline(t *testing.T) {
	m := v41GroupedFixture(t, "Q2_K", true, false)
	b := newV41CohortPanelRecorder()
	b.allowQ2, b.declineWeight = true, true
	width := m.Cfg.NumHeads * m.Cfg.HeadDim
	input := v41GroupedPanelInput(2, width)
	rows := []v41GroupedOutputCohortRow{{Lane: 0, Session: v41DenseTestSession(t, m, b), Input: input[:width]}, {Lane: 1, Session: v41DenseTestSession(t, m, b), Input: input[width:]}}
	scope := beginBackendExecution(b)
	defer scope.finish()
	for _, row := range rows {
		scope.bind(row.Session)
	}
	owner, outcome, err := prepareV41GroupedOutputCohort(scope, 0, rows)
	if err != nil || outcome != v41ProjectionHandled {
		t.Fatal("valid membership refused")
	}
	got, outcome, err := owner.Run()
	if got != nil || err != nil || outcome != v41ProjectionDeclined || !owner.selected || owner.arithmeticSelected || b.matmuls != 0 {
		t.Fatalf("capability decline changed into execution: %v %v", outcome, err)
	}
	for _, row := range rows {
		if row.Session.halClosed || row.Session.halFailure != nil {
			t.Fatal("legitimate capability refusal retired peer")
		}
	}
}
