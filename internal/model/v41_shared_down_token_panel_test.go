package model

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder runs CPU reference arithmetic with the real outer chunk's
// owner, publication and cleanup protocol. It does not execute a native tile.
type v41SharedDownPanelRecorder struct {
	*v41ColdSharedRecorder
	owner                sync.Mutex
	reportedOwner        *sync.Mutex
	tiles, recording     bool
	tileRows             []int
	gatherRows           [][]int
	failGather, failTile int
	failure              error
	aliasTile            bool
	afterSupport         func()
	afterRecording       func()
	probes               []string
}

func newV41SharedDownPanelRecorder() *v41SharedDownPanelRecorder {
	b := &v41SharedDownPanelRecorder{v41ColdSharedRecorder: newV41ColdSharedRecorder(), tiles: true, recording: true}
	b.panel = true
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41SharedDownPanelRecorder) ForwardExecutionOwner() *sync.Mutex {
	if b.reportedOwner != nil {
		return b.reportedOwner
	}
	return &b.owner
}
func (b *v41SharedDownPanelRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q3_K || b.v41ColdSharedRecorder.SupportsDeviceWeightDtype(dt)
}
func (b *v41SharedDownPanelRecorder) tensor(shape []int, values []float32) compute.Tensor {
	t := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	b.created[t.Buf()]++
	return t
}
func (b *v41SharedDownPanelRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, _ string) compute.Tensor {
	if class == compute.MemoryActivation {
		b.activationUploads++
		b.activationBytes += x.Numel() * 4
	} else {
		b.weightUploads++
	}
	if dt == compute.F32 {
		return b.tensor(x.Shape, compute.Default().Read(x))
	}
	raw := x.Buf().(compute.HostBuffer).I8()
	bytes := make([]byte, len(raw))
	for i, value := range raw {
		bytes[i] = byte(value)
	}
	y := compute.NewQ3K(b, x.Shape, bytes)
	b.created[y.Buf()]++
	return y
}
func (b *v41SharedDownPanelRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(x, dt, compute.MemoryWeights, "")
}
func (b *v41SharedDownPanelRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	y := compute.Default().MatMul(w, x)
	defer compute.Default().Free(y)
	return b.tensor(y.Shape, compute.Default().Read(y))
}
func (b *v41SharedDownPanelRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.batchedRows = append(b.batchedRows, rows)
	b.batchedDtypes = append(b.batchedDtypes, w.Dtype)
	y := compute.Default().BatchedMatMul(w, x, rows)
	defer compute.Default().Free(y)
	return b.tensor(y.Shape, compute.Default().Read(y))
}
func (b *v41SharedDownPanelRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	y := b.v41ColdSharedRecorder.SwiGLUWithLimit(g, u, limit)
	defer b.Free(y)
	return b.tensor(y.Shape, compute.Default().Read(y))
}
func (b *v41SharedDownPanelRecorder) V41RowLogAvailable() bool {
	b.probes = append(b.probes, "copies")
	return true
}
func (*v41SharedDownPanelRecorder) V41RowLogAlloc(int, int) (compute.Tensor, error) {
	panic("shared down uses only gather")
}
func (*v41SharedDownPanelRecorder) V41RowLogCopy(compute.Tensor, int, compute.Tensor, int, int) error {
	panic("shared down uses only gather")
}
func (b *v41SharedDownPanelRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("copy without execution owner")
	}
	rows, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	var values []float32
	var rowIDs []int
	for _, span := range spans {
		data := compute.Default().Read(span.Src)
		values = append(values, data[span.Row*width:(span.Row+span.Rows)*width]...)
		rowIDs = append(rowIDs, span.Row, span.Rows)
	}
	b.gatherRows = append(b.gatherRows, rowIDs)
	y := b.tensor([]int{rows, width}, values)
	if len(b.gatherRows) == b.failGather {
		return y, b.failure // Partial outputs must join the outer owner.
	}
	return y, nil
}
func (b *v41SharedDownPanelRecorder) SupportsV41Q3KTokenPanel(w compute.Tensor, rows int) bool {
	if b.deferredPublishes%4 != 3 {
		panic("tile admission moved before existing publications")
	}
	b.probes = append(b.probes, "tile")
	if b.afterSupport != nil {
		b.afterSupport()
	}
	return b.tiles && w.Dtype == compute.Q3_K && rows == 8
}
func (b *v41SharedDownPanelRecorder) V41StepBatchRecording() bool {
	b.probes = append(b.probes, "recording")
	if b.afterRecording != nil {
		b.afterRecording()
	}
	return b.recording
}
func (b *v41SharedDownPanelRecorder) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.tileRows = append(b.tileRows, rows)
	if b.aliasTile {
		return x, nil
	}
	y := compute.Default().BatchedMatMul(w, x, rows)
	defer compute.Default().Free(y)
	result := b.tensor([]int{rows * w.Shape[0]}, compute.Default().Read(y))
	if len(b.tileRows) == b.failTile {
		return result, b.failure
	}
	return result, nil
}

func v41SharedDownPanelFixture() *Model {
	const hidden, inter = 4, 256
	m := &Model{Cfg: Config{NumLayers: 1, HeadDim: 512, HiddenSize: hidden, MoEIntermediateSize: inter, SwigluLimit: .125}, manifest: map[string]tensorMeta{}, kqw: map[string]*kQuantTensor{}}
	for n, leaf := range []string{"w1", "w3"} {
		start := len(m.raw)
		for i := 0; i < hidden*inter; i++ {
			m.raw = binary.LittleEndian.AppendUint32(m.raw, math.Float32bits(float32((i+3*n)%11-4)*.137+.023))
		}
		m.manifest[layerName(0, "ffn.shared_experts."+leaf+".weight")] = tensorMeta{Dtype: "F32", Shape: []int{inter, hidden}, Offset: start, Nbytes: len(m.raw) - start}
	}
	m.kqw[layerName(0, "ffn.shared_experts.w2.weight")] = q3kFixtureTensor(hidden, inter)
	return m
}

// One bounded orchestration regression covers default decline, selection,
// partial-result ownership and the unchanged deferred fault ordering.
// fak-test:runtime fast est=300ms lane=default
func TestV41SharedDown64TokenPanel(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, mode := range []string{"selected", "chunk-tail", "two-chunks", "disabled", "inactive", "tail", "unowned", "owner-mismatch", "epoch-drift", "scope-drift", "copy-partial", "tile-partial", "join-partial", "tile-alias", "numeric-order", "earlier-numeric"} {
		t.Run(mode, func(t *testing.T) {
			m := v41SharedDownPanelFixture()
			b := newV41SharedDownPanelRecorder()
			if !m.holdModelWeights() {
				t.Fatal("model pin refused")
			}
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}, modelWeightsHeld: true}
			t.Cleanup(func() { s.Close(); _ = m.CloseWeights() })
			rows := 64
			failure := errors.New("selected tile failure")
			b.failure = failure
			var want error
			selectedTiles := mode != "disabled" && mode != "inactive" && mode != "tail" && mode != "unowned"
			switch mode {
			case "chunk-tail":
				rows = 65
			case "two-chunks":
				rows = 128
			case "disabled":
				b.tiles = false
			case "inactive":
				b.recording = false
			case "tail":
				rows = 63
			case "owner-mismatch":
				want = errV41ProjectionResult
			case "epoch-drift":
				b.afterSupport = func() { s.cacheGeometryEpoch++ }
				want = errV41ProjectionResult
			case "scope-drift":
				b.afterRecording = func() { s.backendExecutionScope = nil }
				want = errV41ProjectionResult
			case "copy-partial":
				b.failGather, want = 2, failure
			case "tile-partial":
				b.failTile, want = 2, failure
			case "join-partial":
				b.failGather, want = 9, failure
			case "tile-alias":
				b.aliasTile, want = true, errV41ProjectionResult
			case "numeric-order":
				gate := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishSharedExpertProjection, Index: 9 * m.Cfg.MoEIntermediateSize}
				down := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishSharedExpertProjection, Index: 8 * m.Cfg.HiddenSize}
				b.panelVerdicts, want = []error{gate, nil, nil, down}, down
			case "earlier-numeric":
				b.failTile = 2
				b.deferredFaultAt = 2
				b.deferredFault = &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishSharedExpertProjection, Index: 0}
				want = b.deferredFault
			}
			input := v41ColdSharedPanelInput(rows, m.Cfg.HiddenSize)
			var got []float32
			var chosen bool
			var err error
			func() {
				var scope *backendExecutionScope
				if mode != "unowned" {
					scope = s.beginBackendExecution()
					defer scope.finish()
				}
				if mode == "owner-mismatch" {
					b.reportedOwner = &sync.Mutex{}
				}
				got, chosen, err = s.v41ColdSharedExpertPanel(0)(input, rows)
				// Restore the injected binding only so the ordinary teardown can
				// exercise its real owner path after the helper refused drift.
				if mode == "scope-drift" {
					s.backendExecutionScope = scope
				}
				b.reportedOwner = nil
				var numeric *v41SharedPanelDeferredFault
				if errors.As(err, &numeric) {
					if mode != "numeric-order" || numeric.Row != 8 || len(got) != 8*m.Cfg.HiddenSize {
						t.Fatalf("wrong earliest token: %v, values=%d", numeric, len(got))
					}
					err = numeric.finalize()
				}
			}()
			if !chosen || !errors.Is(err, want) || want == nil && err != nil {
				t.Fatalf("selected=%t error=%v want=%v", chosen, err, want)
			}
			chunks := (rows + 63) / 64
			if b.activationUploads != chunks || b.deferredFences != chunks || len(b.deferredPending) != 0 {
				t.Fatalf("uploads=%d fences=%d pending=%d", b.activationUploads, b.deferredFences, len(b.deferredPending))
			}
			wantLegacy := 2
			if !selectedTiles {
				wantLegacy = 3
			} else if mode == "chunk-tail" {
				wantLegacy = 5
			} else if mode == "two-chunks" {
				wantLegacy = 4
			}
			if len(b.batchedRows) != wantLegacy {
				t.Fatalf("legacy projections=%v, want %d; selected work replayed", b.batchedRows, wantLegacy)
			}
			if want == nil {
				// The unchanged scalar shared expert is an independent composer.
				ref := newV41SharedDownPanelRecorder()
				r := &Session{M: m, Backend: ref, halW: map[string]compute.Tensor{}}
				var expected []float32
				for row := 0; row < rows; row++ {
					values, e := m.v41FullSharedExpertSwiGLU(0, input[row*m.Cfg.HiddenSize:(row+1)*m.Cfg.HiddenSize], m.Cfg, r.v41DenseProjectionFunc(), r.v41SharedActivationFunc(), r.v41BF16PublishFunc())
					if e != nil {
						t.Fatal(e)
					}
					expected = append(expected, values...)
				}
				r.Close()
				if !v41ColdSharedEqualWords(got, expected) || b.dataReads != chunks || b.deferredPublishes != 4*chunks {
					t.Fatal("output bits or outer publication/readback count changed")
				}
			}
			if mode == "selected" || mode == "chunk-tail" || mode == "two-chunks" {
				fullChunks := rows / 64
				wantRows, wantProbes := make([]int, 8*fullChunks), make([]string, 0, 3*fullChunks)
				for i := range wantRows {
					wantRows[i] = 8
				}
				for i := 0; i < fullChunks; i++ {
					wantProbes = append(wantProbes, "copies", "tile", "recording")
				}
				if !reflect.DeepEqual(b.tileRows, wantRows) || len(b.gatherRows) != 9*fullChunks || !reflect.DeepEqual(b.probes, wantProbes) {
					t.Fatalf("tiles=%v gathers=%v probes=%v", b.tileRows, b.gatherRows, b.probes)
				}
				for chunk := 0; chunk < fullChunks; chunk++ {
					for tile := 0; tile < 8; tile++ {
						if !reflect.DeepEqual(b.gatherRows[chunk*9+tile], []int{tile * 8, 8}) {
							t.Fatalf("chunk %d tile %d changed source row order", chunk, tile)
						}
					}
				}
			}
			if !selectedTiles && (len(b.gatherRows) != 0 || len(b.tileRows) != 0) {
				t.Fatal("declined path issued tile work")
			}
			// Closing the real pin releases model residents; all owned and
			// partial tensors must then have exactly one matching release.
			s.Close()
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			for buffer, creates := range b.created {
				if b.freed[buffer] != creates {
					t.Fatalf("buffer freed %d times, created %d", b.freed[buffer], creates)
				}
			}
		})
	}
}
