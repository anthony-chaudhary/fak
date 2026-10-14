package model

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// An ordinary ReaderAt deliberately offers no mapped view or retained payload.
type v41GroupedPanelCacheReader struct {
	data         []byte
	calls, bytes int
	err          error
	panicValue   any
}

func (r *v41GroupedPanelCacheReader) ReadAt(p []byte, offset int64) (int, error) {
	r.calls++
	r.bytes += len(p)
	if r.panicValue != nil {
		panic(r.panicValue)
	}
	if r.err != nil {
		return 0, r.err
	}
	return bytes.NewReader(r.data).ReadAt(p, offset)
}

func v41GroupedPanelCacheLazy(m *Model) []*v41GroupedPanelCacheReader {
	var readers []*v41GroupedPanelCacheReader
	for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
		q := m.kqw[layerName(0, leaf)]
		reader := &v41GroupedPanelCacheReader{data: append([]byte(nil), q.raw...)}
		q.lazy, q.raw = &LazyQ4KRange{Reader: reader, Bytes: len(reader.data)}, nil
		readers = append(readers, reader)
	}
	return readers
}

func v41GroupedPanelCacheOracle(t *testing.T, m *Model, input []float32, rows int) [][]float32 {
	t.Helper()
	c := m.Cfg
	width := c.NumHeads * c.HeadDim
	a, aOK := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
	b, bOK := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
	if !aOK || !bOK {
		t.Fatal("oracle weights absent")
	}
	want := make([][]float32, rows)
	for row := range want {
		x := append([]float32(nil), input[row*width:(row+1)*width]...)
		for i := range x {
			x[i] = v41RoundBF16(x[i])
		}
		var err error
		want[row], err = v41FullGroupedOutput(0, x, a, b, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
	}
	return want
}

func v41GroupedPanelCacheCall(s *Session, input []float32, rows int) ([][]float32, v41DenseProjectionOutcome, error) {
	c := s.M.Cfg
	return s.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
}

// fak-test:runtime medium est=15s lane=default
func TestV41GroupedOutputPanelCachedWeights(t *testing.T) {
	for _, packed := range []bool{false, true} {
		name := "F32"
		if packed {
			name = "Q2_K_Q3_K"
		}
		t.Run(name, func(t *testing.T) {
			var m *Model
			var s *Session
			var b *v41GroupedPanelQuantRecorder
			if packed {
				m, s, b = v41GroupedPanelQuantFixture(t)
			} else {
				m, b = v41GroupedBF16Fixture(t), newV41GroupedPanelQuantRecorder()
				s = v41DenseTestSession(t, m, b)
			}
			c := m.Cfg
			const rows = 2
			input := v41GroupedPanelInput(rows, c.NumHeads*c.HeadDim)
			want := v41GroupedPanelCacheOracle(t, m, input, rows)
			var readers []*v41GroupedPanelCacheReader
			if packed {
				readers = v41GroupedPanelCacheLazy(m)
			}
			call := func(session *Session) {
				t.Helper()
				got, outcome, err := v41GroupedPanelCacheCall(session, input, rows)
				if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
					t.Fatalf("panel outcome=%d err=%v parity=%t", outcome, err, reflect.DeepEqual(got, want))
				}
			}
			call(s)
			weights := make(map[string]compute.Tensor, len(s.halW))
			for key, weight := range s.halW {
				weights[key] = weight
			}
			if len(weights) != c.OGroups+1 {
				t.Fatalf("staged weights=%d, want %d", len(weights), c.OGroups+1)
			}
			borrower := v41DenseTestSession(t, m, b)
			// Poisoning only the already-copied host view witnesses skipped
			// rescanning; this does not make mutation of model weights supported.
			var restoreHost func()
			if !packed {
				a := m.tensor(layerName(0, "attn.wo_a.weight"))
				bb := m.tensor(layerName(0, "attn.wo_b.weight"))
				av, bv := a[0], bb[0]
				a[0], bb[0] = float32(math.NaN()), float32(math.NaN())
				restoreHost = func() { a[0], bb[0] = av, bv }
				defer restoreHost()
			}
			before := b.uploads
			call(s)                // Session-local hit.
			s.cacheGeometryEpoch++ // Cache rebuild leaves immutable weights intact.
			call(s)
			// A discarded local memo must resolve the live model-owned handle.
			s.halW, s.borrowedHALW = map[string]compute.Tensor{}, nil
			call(s)
			call(borrower) // Empty session memo, existing model/backend residency.
			if b.uploads-before != 4*c.OGroups {
				t.Fatalf("warm uploads=%d, want only %d activation uploads", b.uploads-before, 4*c.OGroups)
			}
			for key, weight := range weights {
				if s.halW[key].Buf() != weight.Buf() || borrower.halW[key].Buf() != weight.Buf() || b.freed[weight.Buf()] != 0 {
					t.Fatalf("cached handle changed or was freed: %s", key)
				}
				if _, borrowed := borrower.borrowedHALW[key]; !borrowed {
					t.Fatalf("model-owned handle lacks borrowed marker: %s", key)
				}
			}
			for i, reader := range readers {
				if reader.calls != 1 || reader.bytes != len(reader.data) {
					t.Fatalf("weight %d reread lazy payload: calls=%d bytes=%d want 1/%d", i, reader.calls, reader.bytes, len(reader.data))
				}
			}
			if restoreHost != nil {
				restoreHost()
			}
			// The same name, dtype and geometry cannot reuse another backend's handle.
			otherBackend := newV41GroupedPanelQuantRecorder()
			otherSession := v41DenseTestSession(t, m, otherBackend)
			call(otherSession)
			if otherBackend.uploads != 2*c.OGroups+1 {
				t.Fatalf("new backend skipped weight staging: uploads=%d", otherBackend.uploads)
			}
			for key, weight := range weights {
				if otherSession.halW[key].Buf() == weight.Buf() || otherSession.halW[key].Backend() != otherBackend {
					t.Fatalf("weight escaped backend identity: %s", key)
				}
			}
			for i, reader := range readers {
				if reader.calls != 2 || reader.bytes != 2*len(reader.data) {
					t.Fatalf("new backend weight %d materialization=%d/%d", i, reader.calls, reader.bytes)
				}
			}
			var otherModel *Model
			if packed {
				otherModel = v41GroupedFixture(t, "Q2_K", true, false)
				otherModel.kqw[layerName(0, "attn.wo_b.weight")] = q3kFixtureTensor(c.HiddenSize, c.OGroups*c.OLoraRank)
			} else {
				otherModel = v41GroupedBF16Fixture(t)
			}
			otherModelSession := v41DenseTestSession(t, otherModel, b)
			before = b.uploads
			call(otherModelSession)
			if b.uploads-before != 2*c.OGroups+1 {
				t.Fatalf("new model skipped weight staging: uploads=%d", b.uploads-before)
			}
			for key, weight := range weights {
				if otherModelSession.halW[key].Buf() == weight.Buf() {
					t.Fatalf("weight escaped model identity: %s", key)
				}
			}
			otherModelSession.Close()
			if err := otherModel.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			s.Close()
			before = b.uploads
			if err := panicAsError(func() { v41GroupedPanelCacheCall(s, input, rows) }); err == nil || b.uploads != before {
				t.Fatal("closed session reused or restaged a stale handle")
			}
			call(borrower)
			borrower.Close()
			otherSession.Close()
			for _, recorder := range []*v41GroupedPanelQuantRecorder{b, otherBackend} {
				live := 0
				for buffer, created := range recorder.created {
					live += created - recorder.freed[buffer]
				}
				if live != c.OGroups+1 {
					t.Fatalf("live handles after session close=%d, want %d immutable weights", live, c.OGroups+1)
				}
			}
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			for _, recorder := range []*v41GroupedPanelQuantRecorder{b, otherBackend} {
				for buffer, created := range recorder.created {
					if recorder.freed[buffer] != created {
						t.Fatalf("handle lifetime created=%d freed=%d", created, recorder.freed[buffer])
					}
				}
			}
		})
	}
}

// fak-test:runtime medium est=20s lane=default
func TestV41GroupedOutputPanelWarmRefusal(t *testing.T) {
	for _, reason := range []string{"shape", "descriptor", "row-log", "packed-panel", "nonfinite", "deferred-disabled"} {
		t.Run(reason, func(t *testing.T) {
			m, s, b := v41GroupedPanelQuantFixture(t)
			readers := v41GroupedPanelCacheLazy(m)
			input := v41GroupedPanelInput(2, m.Cfg.NumHeads*m.Cfg.HeadDim)
			if _, outcome, err := v41GroupedPanelCacheCall(s, input, 2); err != nil || outcome != v41ProjectionHandled {
				t.Fatalf("warmup failed: outcome=%d err=%v", outcome, err)
			}
			switch reason {
			case "shape":
				m.kqw[layerName(0, "attn.wo_b.weight")].out++
			case "descriptor":
				m.kqw[layerName(0, "attn.wo_b.weight")].nblk++
			case "row-log":
				b.deny = true
			case "packed-panel":
				b.denyQ3 = true
			case "nonfinite":
				input[0] = float32(math.NaN())
			case "deferred-disabled":
				t.Setenv(v41DeferredBF16PublishEnv, "0")
			}
			uploads, matmuls, created, publishes := b.uploads, b.matmuls, len(b.created), b.deferredPublishes
			got, outcome, err := v41GroupedPanelCacheCall(s, input, 2)
			if got != nil || outcome != v41ProjectionDeclined || err != nil || s.BackendSessionClosed() {
				t.Fatalf("refusal got=%v outcome=%d err=%v closed=%t", got, outcome, err, s.BackendSessionClosed())
			}
			if b.uploads != uploads || b.matmuls != matmuls || len(b.created) != created || b.deferredPublishes != publishes {
				t.Fatal("warm refusal performed activation or weight work")
			}
			for _, reader := range readers {
				if reader.calls != 1 {
					t.Fatalf("warm refusal reread payload %d times", reader.calls)
				}
			}
		})
	}
}

// fak-test:runtime medium est=7s lane=default
func TestV41GroupedOutputPanelPayloadFailure(t *testing.T) {
	for failedWeight, leaf := range []string{"wo_a", "wo_b"} {
		t.Run(leaf, func(t *testing.T) {
			m, s, b := v41GroupedPanelQuantFixture(t)
			readers := v41GroupedPanelCacheLazy(m)
			cause := errors.New("grouped checkpoint read")
			readers[failedWeight].err = cause
			resident := m.immutableDeviceWeights(b)
			input := v41GroupedPanelInput(2, m.Cfg.NumHeads*m.Cfg.HeadDim)
			got, outcome, err := v41GroupedPanelCacheCall(s, input, 2)
			var selected *BackendForwardOperationError
			if got != nil || outcome != v41ProjectionError || !errors.Is(err, cause) || !errors.As(err, &selected) ||
				selected.Path != "v41-grouped-output-panel-weights" || selected.Stage != "weight upload" || !s.BackendSessionClosed() {
				t.Fatalf("payload failure got=%v outcome=%d err=%v closed=%t", got, outcome, err, s.BackendSessionClosed())
			}
			staged := failedWeight * m.Cfg.OGroups
			if b.uploads != staged || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 || len(s.halW) != 0 {
				t.Fatalf("failed payload effects uploads=%d matmuls=%d gathers=%d publishes=%d memo=%d", b.uploads, b.matmuls, b.gathers, b.deferredPublishes, len(s.halW))
			}
			if !resident.mu.TryLock() {
				t.Fatal("selected error retained cache lock")
			}
			cached := len(resident.weights)
			resident.mu.Unlock()
			if cached != staged || readers[0].calls != 1 || readers[1].calls != failedWeight {
				t.Fatalf("failed payload published/retried: cached=%d reads=%d/%d", cached, readers[0].calls, readers[1].calls)
			}
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			for buffer, created := range b.created {
				if b.freed[buffer] != created {
					t.Fatalf("failed payload leaked staged handle: created=%d freed=%d", created, b.freed[buffer])
				}
			}
		})
	}
}

// fak-test:runtime medium est=15s lane=default
func TestV41GroupedPanelWeightFailureDoesNotPublish(t *testing.T) {
	for _, failure := range []string{"returned-error", "plain-panic", "backend-panic", "noncomparable-panic"} {
		t.Run(failure, func(t *testing.T) {
			m, s, b := v41GroupedPanelQuantFixture(t)
			readers := v41GroupedPanelCacheLazy(m)
			resident := m.immutableDeviceWeights(b)
			input := v41GroupedPanelInput(2, m.Cfg.NumHeads*m.Cfg.HeadDim)
			cause := errors.New("panel payload")
			var thrown any = cause
			if failure == "backend-panic" {
				thrown = &compute.BackendError{Backend: "panel-test", Class: compute.VulkanClassExecutionFailed, Err: cause}
			} else if failure == "noncomparable-panic" {
				thrown = []int{1, 2}
			}
			if failure == "returned-error" {
				readers[0].err = cause
			} else {
				readers[0].panicValue = thrown
			}
			var got [][]float32
			var outcome v41DenseProjectionOutcome
			var err error
			var recovered any
			func() {
				defer func() { recovered = recover() }()
				got, outcome, err = v41GroupedPanelCacheCall(s, input, 2)
			}()
			if failure == "returned-error" || failure == "backend-panic" {
				if !errors.Is(err, cause) || recovered != nil || got != nil || outcome != v41ProjectionError {
					t.Fatalf("returned error changed: err=%v panic=%v tensor=%v", err, recovered, got)
				}
			} else if !reflect.DeepEqual(recovered, thrown) {
				t.Fatalf("panic changed: got=%v want=%v", recovered, thrown)
			}
			if !resident.mu.TryLock() {
				t.Fatal("payload failure retained model cache lock")
			}
			cached := len(resident.weights)
			resident.mu.Unlock()
			if cached != 0 || len(s.halW) != 0 || b.uploads != 0 {
				t.Fatalf("failed payload published: model=%d session=%d uploads=%d", cached, len(s.halW), b.uploads)
			}
			if !s.BackendSessionClosed() {
				t.Fatal("selected failure left the session open")
			}
			readers[0].err, readers[0].panicValue = nil, nil
			fresh := v41DenseTestSession(t, m, b)
			got, outcome, err = v41GroupedPanelCacheCall(fresh, input, 2)
			if err != nil || got == nil || outcome != v41ProjectionHandled || b.uploads != 2*m.Cfg.OGroups+1 {
				t.Fatalf("clean staging after failure: outcome=%d err=%v uploads=%d", outcome, err, b.uploads)
			}
			fresh.Close()
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
