package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ScoreTestCall struct {
	q, keys, weights []float32
	rows, heads, dim int
}

// CPU storage and an independent ordered binary32 oracle test only the model
// adapter. This recorder is not a hardware or production qualification.
type v41ScoreTestBackend struct {
	*v41DenseTestBackend
	identity          compute.Backend
	afterAdmission    func()
	onKeyUpload       func([]float32)
	afterQueryUpload  func()
	supported, device bool
	calls             []v41ScoreTestCall
	allocations       map[compute.Buffer]bool
	managed           map[compute.Buffer]bool
	output            compute.Buffer
	readScratch       []float32
	uploadCount       int
	dispatchCount     int
	readCount         int
	fault             string
	cause             error
	panicValue        any
	cleanupPanic      any
	freeAttempts      int
	forceNaN          bool
	healthErr         error
	healthCalls       int
	rowLog            bool
	keyUploadBytes    int64
	keyCopyBytes      int64
	keyLiveBytes      int64
	keyPeakBytes      int64
	keyOwned          map[compute.Buffer]int64
	copyPending       map[compute.Buffer]bool
}

func (b *v41ScoreTestBackend) scoreIdentity() compute.Backend {
	if b.identity != nil {
		return b.identity
	}
	return b
}

func newV41ScoreTestBackend() *v41ScoreTestBackend {
	return &v41ScoreTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), supported: true, device: true,
		allocations: map[compute.Buffer]bool{}, managed: map[compute.Buffer]bool{}, keyOwned: map[compute.Buffer]int64{}, copyPending: map[compute.Buffer]bool{}}
}

func (b *v41ScoreTestBackend) Caps() compute.Caps {
	c := b.v41DenseTestBackend.Caps()
	c.DeviceMemory = b.device
	return c
}

func (b *v41ScoreTestBackend) SupportsV41IndexerScore() bool { return b.supported }
func (b *v41ScoreTestBackend) V41IndexerScoreAdmission() (bool, error) {
	b.healthCalls++
	if b.afterAdmission != nil {
		b.afterAdmission()
	}
	if b.healthErr != nil {
		return false, b.healthErr
	}
	return b.supported, nil
}
func (b *v41ScoreTestBackend) V41IndexerScoreUnavailableReason() string {
	return "software recorder; no physical qualification"
}

func (b *v41ScoreTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 indexer score ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("unexpected score upload dtype/class")
	}
	if strings.TrimPrefix(site, "V4.1 indexer score ")+" upload" == b.fault {
		panic(b.cause)
	}
	source := b.Backend.Read(x)
	if strings.HasSuffix(site, " keys") && b.onKeyUpload != nil {
		b.onKeyUpload(source)
	}
	values := append([]float32(nil), source...)
	out := compute.NewF32(b.scoreIdentity(), append([]int(nil), x.Shape...), values)
	b.allocations[out.Buf()], b.managed[out.Buf()] = true, true
	b.uploadCount++
	if strings.HasSuffix(site, " keys") || strings.HasSuffix(site, " key append") {
		b.keyUploadBytes += int64(len(values)) * 4
	}
	if strings.HasSuffix(site, " key append") {
		b.ownKeys(out)
	}
	if strings.HasSuffix(site, " query") && b.afterQueryUpload != nil {
		b.afterQueryUpload()
	}
	return out
}

func (b *v41ScoreTestBackend) V41IndexerScore(q, keys, weights compute.Tensor, rows, heads, dim int) (compute.Tensor, error) {
	b.dispatchCount++
	for _, input := range []struct {
		x     compute.Tensor
		shape []int
	}{{q, []int{heads, dim}}, {keys, []int{rows, dim}}, {weights, []int{heads}}} {
		if input.x.Backend() != b.scoreIdentity() || !b.allocations[input.x.Buf()] || input.x.Dtype != compute.F32 ||
			input.x.Layout != compute.RowMajor || input.x.Quant != nil || !reflect.DeepEqual(input.x.Shape, input.shape) {
			return compute.Tensor{}, errors.New("invalid score tensor")
		}
	}
	if q.Buf() == keys.Buf() || q.Buf() == weights.Buf() || keys.Buf() == weights.Buf() {
		return compute.Tensor{}, errors.New("aliased score inputs")
	}
	c := v41ScoreTestCall{append([]float32(nil), b.Backend.Read(q)...), append([]float32(nil), b.Backend.Read(keys)...),
		append([]float32(nil), b.Backend.Read(weights)...), rows, heads, dim}
	b.calls = append(b.calls, c)
	if b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	if b.fault == "panic" {
		if b.panicValue != nil {
			panic(b.panicValue)
		}
		panic(b.cause)
	}
	if b.fault == "alias" {
		return q, nil
	}
	if b.fault == "cache alias" {
		return keys, nil
	}
	values := v41ScoreTestOracle(c)
	if b.forceNaN {
		values[0] = float32(math.NaN())
	}
	out := compute.NewF32(b.scoreIdentity(), []int{rows}, values)
	b.allocations[out.Buf()], b.managed[out.Buf()], b.output = true, true, out.Buf()
	if b.fault == "partial output" {
		return out, b.cause
	}
	if b.fault == "shape" {
		out.Shape = []int{rows, 1}
	}
	if b.fault == "dtype" {
		out.Dtype = compute.BF16
	}
	if b.fault == "layout" {
		out.Layout = compute.ColMajor
	}
	if b.fault == "quant" {
		out.Quant = &compute.QuantSpec{}
	}
	return out, nil
}

func (b *v41ScoreTestBackend) Read(x compute.Tensor) []float32 {
	if x.Buf() != b.output {
		return b.v41DenseTestBackend.Read(x)
	}
	b.readCount++
	if b.fault == "read" {
		panic(b.cause)
	}
	clear(b.copyPending) // Checked score read fences the copies feeding it.
	b.readScratch = append(b.readScratch[:0], b.Backend.Read(x)...)
	if b.fault == "short read" {
		return b.readScratch[:len(b.readScratch)-1]
	}
	return b.readScratch
}

func (b *v41ScoreTestBackend) Free(x compute.Tensor) {
	if b.copyPending[x.Buf()] {
		panic("index key buffer freed before score read or batch flush")
	}
	if !b.managed[x.Buf()] {
		b.v41DenseTestBackend.Free(x)
		return
	}
	if !b.allocations[x.Buf()] {
		panic("score double free")
	}
	isOutput := x.Buf() == b.output
	if isOutput {
		for i := range b.readScratch {
			b.readScratch[i] = float32(math.NaN())
		}
		b.output = nil
	}
	for i := range b.Backend.Read(x) {
		b.Backend.Read(x)[i] = float32(math.NaN())
	}
	delete(b.allocations, x.Buf())
	b.keyLiveBytes -= b.keyOwned[x.Buf()]
	delete(b.keyOwned, x.Buf())
	b.Backend.Free(x)
	b.freeAttempts++
	if b.cleanupPanic != nil {
		panic(b.cleanupPanic)
	}
	if isOutput && b.fault == "free" {
		panic(b.cause)
	}
}

func (b *v41ScoreTestBackend) BeginBatch()              {}
func (b *v41ScoreTestBackend) FlushBatch()              { clear(b.copyPending) }
func (b *v41ScoreTestBackend) V41RowLogAvailable() bool { return b.rowLog }
func (b *v41ScoreTestBackend) ownKeys(t compute.Tensor) {
	bytes := int64(t.Numel()) * 4
	b.keyOwned[t.Buf()] = bytes
	b.keyLiveBytes += bytes
	b.keyPeakBytes = max(b.keyPeakBytes, b.keyLiveBytes)
}
func (b *v41ScoreTestBackend) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	if err := compute.ValidateV41RowLogAlloc(rows, width); err != nil {
		return compute.Tensor{}, err
	}
	if b.fault == "row allocation" {
		return compute.Tensor{}, b.cause
	}
	out := compute.NewF32(b.scoreIdentity(), []int{rows, width}, make([]float32, rows*width))
	b.allocations[out.Buf()], b.managed[out.Buf()] = true, true
	b.ownKeys(out)
	return out, nil
}
func (b *v41ScoreTestBackend) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	if b.fault == "row copy" {
		return b.cause
	}
	stride, err := compute.ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows)
	if err != nil || dst.Backend() != b.scoreIdentity() || src.Backend() != b.scoreIdentity() || !b.allocations[dst.Buf()] || !b.allocations[src.Buf()] {
		return errors.New("invalid live row-log copy")
	}
	width := stride / 4
	copy(b.Backend.Read(dst)[dstRow*width:(dstRow+rows)*width], b.Backend.Read(src)[srcRow*width:(srcRow+rows)*width])
	b.keyCopyBytes += int64(rows * stride)
	b.copyPending[dst.Buf()], b.copyPending[src.Buf()] = true, true
	return nil
}
func (b *v41ScoreTestBackend) V41RowLogGather([]compute.V41RowSpan, int) (compute.Tensor, error) {
	return compute.Tensor{}, errors.New("exact key residency must not gather full history per score")
}

// fak-test:runtime medium est=3s lane=default
func TestV41IndexerKeyResidency(t *testing.T) {
	const allowance int64 = 32 << 10
	fixture := func(t *testing.T) *Model {
		m := v41IndexerTestFixture(t)
		// Layer 1 scores keys owned by the registry's layer-0 source.
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		return m
	}
	newSession := func(t *testing.T, m *Model, budget int64) (*Session, *v41ScoreTestBackend) {
		b := newV41ScoreTestBackend()
		b.rowLog = true
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		s.v41State().gatherOps.indexerKeyBudgetBytes = budget
		return s, b
	}
	t.Run("actual producer reader traffic and rebind budget", func(t *testing.T) {
		m := fixture(t)
		s, b := newSession(t, m, allowance)
		control, plain := newSession(t, m, 0)
		bound, callbacks := s.v41State().indexerScore(), 0
		s.v41State().setIndexerScore(func(layer int, q, keys, weights []float32, heads, width, rows int) ([]float32, error) {
			callbacks++
			return bound(layer, q, keys, weights, heads, width, rows)
		})
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), control.Prefill([]int{1, 2, 3}))
		width := int64(m.Cfg.IndexHeadDim * 4)
		for _, token := range []int{4, 5, 6} {
			before, copyBefore := b.keyUploadBytes, b.keyCopyBytes
			beforeCallbacks := callbacks
			v41ScoreTestBits(t, s.Step(token), control.Step(token))
			want := int64(2) * width
			if token == 4 {
				want = 4 * width // Both independent source domains seed two rows.
			} else if token == 5 {
				want = 0
				if b.keyCopyBytes != copyBefore {
					t.Fatal("unchanged key history was copied on device")
				}
			} else if b.keyCopyBytes-copyBefore != 6*width {
				t.Fatal("append D2D accounting omitted old-prefix or staging copies")
			}
			if delta := b.keyUploadBytes - before; delta != want {
				t.Fatalf("token=%d key H2D=%d want=%d", token, delta, want)
			}
			if b.keyPeakBytes > allowance || plain.keyLiveBytes != 0 || len(b.copyPending) != 0 || callbacks != beforeCallbacks+2 || s.v41State().gatherOps.indexerKeyCall != nil ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(control.v41Forward)) {
				t.Fatal("budget, read fence, disabled control or retained-state parity changed")
			}
		}
		// Restore/rebinding creates a fresh gather owner but cannot erase its
		// session's still-live tensors from the accounting ledger.
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer snap.Close()
		if err := snap.Restore(s); err != nil {
			t.Fatal(err)
		}
		v41ScoreOnly(s)
		s.v41State().gatherOps.indexerKeyBudgetBytes = b.keyLiveBytes
		before, retained := b.keyUploadBytes, b.keyLiveBytes
		v41ScoreTestBits(t, s.Step(7), control.Step(7))
		if b.keyUploadBytes-before != 6*width || b.keyLiveBytes != retained {
			t.Fatal("rebound cache discarded live-byte charges or mutated before refusal")
		}
		s.v41State().gatherOps.indexerKeyBudgetBytes = allowance
		v41ScoreTestBits(t, s.Step(0), control.Step(0))
		if b.keyPeakBytes > allowance {
			t.Fatal("reseed exceeded all-owner peak allowance")
		}
		s.Close()
		if b.keyLiveBytes != 0 || len(b.allocations) != 0 {
			t.Fatal("session close retained index-key allocations")
		}
	})
	t.Run("original captured admission and panic hook", func(t *testing.T) {
		for _, panicCallback := range []bool{false, true} {
			m := fixture(t)
			s, b := newSession(t, m, allowance)
			s.Prefill([]int{1, 2, 3})
			s.Step(4)
			st := s.v41State()
			fault := &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassDeviceLost, Err: ErrV41ForwardStage}
			before := b.uploadCount
			if panicCallback {
				marker := &struct{ value int }{73}
				st.setIndexerScore(func(int, []float32, []float32, []float32, int, int, int) ([]float32, error) { panic(marker) })
				if got := v41IndexerTestRecover(func() { s.Step(5) }); got != marker {
					t.Fatalf("callback panic identity changed: %v", got)
				}
			} else {
				b.healthErr = fault
				st.setIndexerScore(s.v41IndexerScoreFunc())
				b.healthErr = nil // Even this artificial recovery cannot erase the captured error.
				if err := recoverError(func() { s.Step(5) }); !errors.Is(err, fault) || !s.BackendSessionClosed() {
					t.Fatalf("original admission capture was bypassed: %v", err)
				}
			}
			if st.gatherOps.indexerKeyCall != nil || b.uploadCount != before {
				t.Fatal("invocation hook survived unwind or staged before original refusal")
			}
		}
	})
	t.Run("rollback reseeds immutable anchor", func(t *testing.T) {
		m := fixture(t)
		s, b := newSession(t, m, allowance)
		control, _ := newSession(t, m, 0)
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), control.Prefill([]int{1, 2, 3}))
		v41ScoreTestBits(t, s.Step(4), control.Step(4))
		v41ScoreTestBits(t, s.Step(5), control.Step(5))
		st := s.v41State()
		before := captureV41ForwardSnapshot(st)
		fault := errors.New("after index scoring")
		st.finalNorm = func([]float32) ([]float32, error) { return nil, fault }
		calls := len(b.calls)
		err := recoverError(func() { _, _, _ = m.forwardV41Step(6, st, nil) })
		var selected *V41ProjectionOperationError
		if !errors.As(err, &selected) || !errors.Is(err, fault) || len(b.calls) != calls+2 || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatalf("late selected panic did not roll back publication after both scores: %v", err)
		}
		st.finalNorm = nil
		uploads := b.keyUploadBytes
		v41ScoreTestBits(t, s.Step(7), control.Step(7))
		if b.keyUploadBytes-uploads != int64(2*3*m.Cfg.IndexHeadDim*4) {
			t.Fatal("retry reused the failed step's removed publication")
		}
	})
	t.Run("same state reset and forged caller rows", func(t *testing.T) {
		m := fixture(t)
		s, b := newSession(t, m, allowance)
		s.Prefill([]int{1, 2, 3})
		s.Step(4)
		st := s.v41State()
		owner := st.layerState(0)
		owner.Reset()
		for value := 1; value <= 2; value++ {
			row := make([]float32, m.Cfg.IndexHeadDim)
			row[0] = float32(value)
			owner.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0, IsIndexSource: true}, IndexKey: row}})
		}
		keys, _ := owner.borrowIndexKeys(0)
		view := &v41IndexerKeyView{owner: owner, source: 0}
		forged := cloneV41Rows(keys)
		if st.gatherOps.residentIndexerScore(m, 0, 4, forged, view, st.indexerScore()) != nil {
			t.Fatal("caller-owned equal rows gained publication identity")
		}
		score := st.gatherOps.residentIndexerScore(m, 0, 4, keys, view, st.indexerScore())
		if score == nil {
			t.Fatal("same-owner reset did not admit a bounded reseed")
		}
		q := make([]float32, m.Cfg.IndexHeadDim)
		q[0] = 1
		before := b.keyUploadBytes
		got, err := score(0, q, nil, []float32{1}, 1, len(q), 2)
		if err != nil {
			t.Fatal(err)
		}
		v41ScoreTestBits(t, got, []float32{1, 2})
		if b.keyUploadBytes-before != int64(2*len(q)*4) {
			t.Fatal("same-pointer reset reused the old terminal anchor")
		}
	})
	for _, fault := range []string{"row allocation", "row copy", "key append upload", "cache alias", "read"} {
		t.Run("selected failure "+fault, func(t *testing.T) {
			m := fixture(t)
			s, b := newSession(t, m, allowance)
			s.Prefill([]int{1, 2, 3})
			s.Step(4)
			s.Step(5)
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault, b.cause = fault, &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			var closed *BackendForwardOperationError
			err := recoverError(func() { s.Step(6) })
			if !errors.As(err, &closed) || closed.Path != "v41-indexer-score" || !s.BackendSessionClosed() || b.keyLiveBytes != 0 || len(b.allocations) != 0 ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatalf("selected key failure did not retire/roll back once: %v", err)
			}
		})
	}
}

func v41ScoreTestOracle(c v41ScoreTestCall) []float32 {
	add := func(a, b float32) float32 { return float32(float64(a) + float64(b)) }
	mul := func(a, b float32) float32 { return float32(float64(a) * float64(b)) }
	out := make([]float32, c.rows)
	for row := range out {
		var total float32
		for head := 0; head < c.heads; head++ {
			var dot float32
			for d := 0; d < c.dim; d++ {
				dot = add(dot, mul(c.q[head*c.dim+d], c.keys[row*c.dim+d]))
			}
			if dot < 0 {
				dot = 0
			}
			total = add(total, mul(dot, c.weights[head]))
		}
		out[row] = total
	}
	return out
}

func v41ScoreTestSession(t *testing.T) (*Session, *v41ScoreTestBackend) {
	t.Helper()
	b := newV41ScoreTestBackend()
	s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
	t.Cleanup(s.Close)
	return s, b
}

func v41ScoreTestBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("score widths %d/%d", len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(want[i])) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("score %d = %g, want NaN", i, got[i])
			}
		} else if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("score %d bits=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexerScoreDeviceContract(t *testing.T) {
	for _, mode := range []string{"absent", "false", "host storage", "no F32"} {
		t.Run(mode, func(t *testing.T) {
			s, b := v41ScoreTestSession(t)
			switch mode {
			case "absent":
				s.Backend = b.v41DenseTestBackend
			case "false":
				b.supported = false
			case "host storage":
				b.device = false
			case "no F32":
				b.deny = true
			}
			if s.v41IndexerScoreFunc() != nil || b.uploadCount != 0 || len(b.calls) != 0 {
				t.Fatal("unqualified backend selected or uploaded")
			}
		})
	}
	for _, tc := range []struct {
		name             string
		q, keys, weights []float32
		heads, dim       int
		want             []float32
	}{
		{"rounding", []float32{0x1p24, 1, -0x1p24}, []float32{1, 1, 1, 0, 0.5, 0}, []float32{1}, 1, 3, []float32{0, 0.5}},
		{"signed heads", []float32{1, -1}, []float32{2, -2}, []float32{1, -2}, 2, 1, []float32{2, -4}},
		{"ties and newest block", []float32{1}, []float32{5, 5, 4, 4, 1}, []float32{1}, 1, 1, []float32{5, 5, 4, 4, 1}},
		{"subnormal", []float32{math.SmallestNonzeroFloat32}, []float32{1}, []float32{1}, 1, 1, []float32{math.SmallestNonzeroFloat32}},
		{"infinity", []float32{math.MaxFloat32}, []float32{2}, []float32{1}, 1, 1, []float32{float32(math.Inf(1))}},
		{"negative infinity", []float32{math.MaxFloat32}, []float32{2}, []float32{-1}, 1, 1, []float32{float32(math.Inf(-1))}},
		{"NaN", []float32{math.MaxFloat32}, []float32{2}, []float32{0}, 1, 1, []float32{float32(math.NaN())}},
		{"empty", []float32{1}, nil, []float32{1}, 1, 1, []float32{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, b := v41ScoreTestSession(t)
			q, keys, weights := append([]float32(nil), tc.q...), append([]float32(nil), tc.keys...), append([]float32(nil), tc.weights...)
			got, err := s.v41IndexerScoreFunc()(2, tc.q, tc.keys, tc.weights, tc.heads, tc.dim, len(tc.want))
			if err != nil || got == nil {
				t.Fatalf("score failed: %v", err)
			}
			v41ScoreTestBits(t, got, tc.want) // release poisoned the borrowed readback
			v41ScoreTestBits(t, tc.q, q)
			v41ScoreTestBits(t, tc.keys, keys)
			v41ScoreTestBits(t, tc.weights, weights)
			wantCalls := 1
			if len(tc.want) == 0 {
				wantCalls = 0
			}
			if len(b.calls) != wantCalls || b.uploadCount != 3*wantCalls || b.readCount != wantCalls || len(b.allocations) != 0 || s.BackendSessionClosed() {
				t.Fatal("score dispatch, transfer, ownership or empty-row contract changed")
			}
			wantPub, wantErr := NewV41IndexerPublication(2, tc.q, tc.keys, tc.weights, tc.heads, tc.dim, len(tc.want), 1, 2, 2, 0)
			rows, gotErr := v41IndexRowsWithScore(2, tc.q, tc.keys, tc.weights, tc.heads, tc.dim, len(tc.want), 1, 2, 2, s.v41IndexerScoreFunc(), s.v41IndexerScoreHealthFunc())
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("selection acceptance differs: %v/%v", gotErr, wantErr)
			}
			if wantErr == nil && !reflect.DeepEqual(rows, wantPub.Rows()) {
				t.Fatalf("selection=%v want=%v", rows, wantPub.Rows())
			}
			if gotErr != nil {
				var selected *V41IndexerScoreOperationError
				if !errors.As(gotErr, &selected) {
					t.Fatal("NaN rejection lost selected marker")
				}
			}
		})
	}
	for _, tc := range []struct {
		name             string
		q, keys, weights []float32
		heads, dim, rows int
	}{
		{"query empty rows", []float32{float32(math.NaN())}, nil, []float32{1}, 1, 1, 0},
		{"weights empty rows", []float32{1}, nil, []float32{float32(math.Inf(-1))}, 1, 1, 0},
		{"last key", []float32{1}, []float32{1, float32(math.Inf(1))}, []float32{1}, 1, 1, 2},
		{"shape", []float32{1}, []float32{1}, []float32{1}, 2, 1, 1},
		{"overflow", nil, nil, nil, math.MaxInt32, 2, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, b := v41ScoreTestSession(t)
			got, err := s.v41IndexerScoreFunc()(3, tc.q, tc.keys, tc.weights, tc.heads, tc.dim, tc.rows)
			var closed *BackendForwardOperationError
			if got != nil || !errors.As(err, &closed) || closed.Stage != "payload" || !s.BackendSessionClosed() || b.uploadCount != 0 || len(b.calls) != 0 {
				t.Fatalf("invalid host payload uploaded or failed open: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexerScoreDeviceFailures(t *testing.T) {
	for _, fault := range []string{"query upload", "keys upload", "weights upload", "dispatch", "partial output", "panic", "alias", "shape", "dtype", "layout", "quant", "read", "short read", "free"} {
		t.Run(fault, func(t *testing.T) {
			s, b := v41ScoreTestSession(t)
			b.fault = fault
			b.cause = &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			score := s.v41IndexerScoreFunc()
			out, err := score(4, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1)
			var closed *BackendForwardOperationError
			if out != nil || !errors.As(err, &closed) || closed.Path != "v41-indexer-score" || closed.Layer != 4 || !s.BackendSessionClosed() || len(b.allocations) != 0 {
				t.Fatalf("selected failure did not close/release: %v", err)
			}
			calls, uploads, reads := len(b.calls), b.uploadCount, b.readCount
			if r := v41IndexerTestRecover(func() { _, _ = score(4, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1) }); r != closed || len(b.calls) != calls || b.uploadCount != uploads || b.readCount != reads {
				t.Fatal("closed callback changed latch or retried")
			}
		})
	}
	t.Run("unknown panic identity", func(t *testing.T) {
		s, b := v41ScoreTestSession(t)
		b.fault, b.panicValue = "panic", &struct{ marker int }{17}
		if r := v41IndexerTestRecover(func() { _, _ = s.v41IndexerScoreFunc()(0, []float32{1}, []float32{1}, []float32{1}, 1, 1, 1) }); r != b.panicValue || !s.BackendSessionClosed() || len(b.allocations) != 0 {
			t.Fatal("unknown panic identity, cleanup or close changed")
		}
	})
}

// fak-test:runtime fast est=2ms lane=default
func TestV41IndexerScoreDeviceHealth(t *testing.T) {
	for _, boundFirst := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(boundFirst)), func(t *testing.T) {
			s, b := v41ScoreTestSession(t)
			var score v41IndexerScoreFunc
			if boundFirst {
				score = s.v41IndexerScoreFunc()
			}
			fault := &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassDeviceLost, Err: ErrV41ForwardStage}
			b.healthErr, b.supported = fault, false
			if !boundFirst {
				score = s.v41IndexerScoreFunc()
			}
			if score == nil {
				t.Fatal("sticky false capability became host eligibility")
			}
			out, err := score(2, []float32{1}, nil, []float32{1}, 1, 1, 0)
			var closed *BackendForwardOperationError
			if out != nil || !errors.Is(err, fault) || !errors.As(err, &closed) || closed.Stage != "admission" || !s.BackendSessionClosed() || b.uploadCount != 0 || len(b.calls) != 0 || b.readCount != 0 {
				t.Fatalf("empty score lost sticky health: %v", err)
			}
		})
	}
	// #13791: healthy absence (the Vulkan shim's current production gate) binds
	// no health seam, so the host scorer pays no per-call admission probe.
	t.Run("unsupported-binds-no-health", func(t *testing.T) {
		s, b := v41ScoreTestSession(t)
		b.supported = false
		score, health := s.v41IndexerScoreFunc(), s.v41IndexerScoreHealthFunc()
		if score != nil || health != nil {
			t.Fatal("healthy absence bound a device score or health seam")
		}
		bound := b.healthCalls
		for i := 0; i < 3; i++ {
			rows, err := v41IndexRowsWithScore(0, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1, 0, 0, 1, score, health)
			if err != nil || !reflect.DeepEqual(rows, []int32{0}) {
				t.Fatalf("healthy unsupported host changed: %v %v", rows, err)
			}
		}
		if b.healthCalls != bound || b.uploadCount != 0 || len(b.calls) != 0 || b.readCount != 0 || s.BackendSessionClosed() {
			t.Fatalf("host path probed the device: admission calls %d after bind %d", b.healthCalls, bound)
		}
	})
	t.Run("bind-time-fault-keeps-health", func(t *testing.T) {
		s, b := v41ScoreTestSession(t)
		fault := &compute.BackendError{Backend: "score-recorder", Class: compute.VulkanClassSubmissionFailed, Err: ErrV41ForwardStage}
		b.supported, b.healthErr = false, fault
		health := s.v41IndexerScoreHealthFunc()
		if health == nil {
			t.Fatal("bind-time admission fault dropped the health seam")
		}
		rows, err := v41IndexRowsWithScore(0, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1, 0, 0, 1, nil, health)
		var selected *V41IndexerScoreOperationError
		if rows != nil || !errors.Is(err, fault) || !errors.As(err, &selected) || !s.BackendSessionClosed() || b.uploadCount != 0 || len(b.calls) != 0 || b.readCount != 0 {
			t.Fatalf("bind-time fault hidden: %v", err)
		}
	})
	t.Run("supported-keeps-health", func(t *testing.T) {
		s, _ := v41ScoreTestSession(t)
		if s.v41IndexerScoreHealthFunc() == nil {
			t.Fatal("supported backend lost its health seam")
		}
	})
}

// fak-test:runtime fast est=5ms lane=default
// Estimate is unmeasured. CPU recorder checks the callback contract only; no
// current native-backend double fault or physical qualification is claimed.
func TestV41IndexerScorePrimaryFailureSurvivesCleanup(t *testing.T) {
	for _, cleanup := range []any{&compute.BackendError{Backend: "cleanup", Class: compute.VulkanClassExecutionFailed, Err: errors.New("release failure")}, "unclassified release panic"} {
		for _, tc := range []struct {
			name, fault string
			panicValue  any
			owned       int
			cleanupOnly bool
		}{
			{name: "returned dispatch", fault: "dispatch", owned: 3},
			{name: "returned partial", fault: "partial output", owned: 4},
			{name: "typed dispatch panic", fault: "panic", owned: 3},
			{name: "typed read panic", fault: "read", owned: 4},
			{name: "unknown pointer panic", fault: "panic", panicValue: &struct{ marker int }{17}, owned: 3},
			{name: "unknown error panic", fault: "panic", panicValue: errors.New("original unknown panic"), owned: 3},
			{name: "unknown string panic", fault: "panic", panicValue: "original panic", owned: 3},
			{name: "cleanup only", owned: 4, cleanupOnly: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				s, b := v41ScoreTestSession(t)
				primary := &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected operation failure")}
				b.fault, b.cause, b.panicValue, b.cleanupPanic = tc.fault, primary, tc.panicValue, cleanup
				var out []float32
				var err error
				recovered := v41IndexerTestRecover(func() { out, err = s.v41IndexerScoreFunc()(4, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1) })
				if tc.panicValue != nil {
					if recovered != tc.panicValue {
						t.Fatalf("primary panic identity lost: got %v want %v", recovered, tc.panicValue)
					}
				} else if tc.cleanupOnly {
					if cleanupErr, ok := cleanup.(error); ok {
						if recovered != nil || !errors.Is(err, cleanupErr) {
							t.Fatalf("cleanup-only typed failure changed: panic=%v err=%v", recovered, err)
						}
					} else if recovered != cleanup {
						t.Fatalf("cleanup-only panic changed: %v", recovered)
					}
				} else if recovered != nil || !errors.Is(err, primary) {
					t.Fatalf("primary selected cause lost: panic=%v err=%v", recovered, err)
				}
				var closed *BackendForwardOperationError
				if out != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Layer != 4 || len(b.allocations) != 0 || b.freeAttempts != tc.owned || len(b.calls) != 1 {
					t.Fatalf("failure did not close and attempt every owned release: out=%v closed=%v live=%d free=%d calls=%d", out, s.halFailure, len(b.allocations), b.freeAttempts, len(b.calls))
				}
				if !tc.cleanupOnly && tc.panicValue == nil && !errors.Is(s.halFailure, primary) {
					t.Fatal("latched primary cause changed")
				}
				releases := b.freeAttempts
				retry := v41IndexerTestRecover(func() { _, _ = s.v41IndexerScoreFunc()(4, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1) })
				if retry != s.halFailure || len(b.calls) != 1 || b.freeAttempts != releases {
					t.Fatal("closed callback retried work or cleanup")
				}
			})
		}
	}
}
