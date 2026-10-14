package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder witnesses exact operands, transfer calls and ownership on cpu-ref.
// It does not establish physical traffic, submission counts or latency.
type v41InverseResidencyRecorder struct {
	*v41DenseTestBackend
	tables, outputs            map[compute.Buffer]bool
	frees                      map[compute.Buffer]int
	tableUploads, calls, reads int
	lastTable                  []float32
	faultSite                  string
	fault                      error
}

func newV41InverseResidencyRecorder() *v41InverseResidencyRecorder {
	return &v41InverseResidencyRecorder{
		v41DenseTestBackend: newV41DenseTestBackend(),
		tables:              map[compute.Buffer]bool{}, outputs: map[compute.Buffer]bool{}, frees: map[compute.Buffer]int{},
	}
}

func (b *v41InverseResidencyRecorder) V41TailRoPERowsAvailable() bool { return true }

func (b *v41InverseResidencyRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 inverse-RoPE table" && site != "V4.1 inverse-RoPE output" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		panic("inverse-RoPE operand is not a classed F32 activation")
	}
	if site == "V4.1 inverse-RoPE table" && b.faultSite == "table upload" {
		panic(b.fault)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.live[out.Buf()] = true
	if site == "V4.1 inverse-RoPE table" {
		b.tables[out.Buf()] = true
		b.tableUploads++
	}
	return out
}

func (b *v41InverseResidencyRecorder) V41TailRoPERows(x, table compute.Tensor, rows, headDim, rotaryDim int) (compute.Tensor, error) {
	if x.Backend() != b || table.Backend() != b || !b.live[x.Buf()] || !b.live[table.Buf()] ||
		!reflect.DeepEqual(x.Shape, []int{rows, headDim}) || !reflect.DeepEqual(table.Shape, []int{rotaryDim / 2, 2}) {
		return compute.Tensor{}, errors.New("inverse-RoPE consumed invalid or freed operands")
	}
	b.calls++
	b.lastTable = append([]float32(nil), b.Backend.Read(table)...)
	if b.faultSite == "inverse-rope" {
		return compute.Tensor{}, b.fault
	}
	out := compute.NewF32(b, []int{rows, headDim}, v41TailRoPEExactOracle(b.Backend.Read(x), b.lastTable, headDim, rotaryDim))
	b.live[out.Buf()], b.outputs[out.Buf()] = true, true
	return out, nil
}

func (b *v41InverseResidencyRecorder) Read(x compute.Tensor) []float32 {
	if b.outputs[x.Buf()] {
		b.reads++
		if b.faultSite == "output readback" {
			panic(b.fault)
		}
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41InverseResidencyRecorder) Free(x compute.Tensor) {
	b.frees[x.Buf()]++
	delete(b.tables, x.Buf())
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// fak-test:runtime fast est=100ms lane=default
func TestV41InverseRoPETableResidency(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("inverse table residency witness requires cpu-ref")
	}
	newSession := func(t *testing.T, model *Model) (*Session, *v41InverseResidencyRecorder) {
		t.Helper()
		b := newV41InverseResidencyRecorder()
		s := &Session{M: model, Backend: b}
		t.Cleanup(s.Close)
		return s, b
	}
	check := func(t *testing.T, b *v41InverseResidencyRecorder, rotate v41InverseOutputRoPEFunc, layer int, cos, sin []float32, uploads int) {
		t.Helper()
		input := []float32{1, -2, 3, -4, 5, -6, 7, -8}
		before := append([]float32(nil), input...)
		table := make([]float32, len(cos)*2)
		for i := range cos {
			table[2*i], table[2*i+1] = sin[i], cos[i]
		}
		got, err := rotate(layer, input, cos, sin, 1, len(input), len(table))
		if err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "current table", b.lastTable, table)
		v41TailRoPECheckBits(t, "inverse result", got, v41TailRoPEExactOracle(input, table, len(input), len(table)))
		v41TailRoPECheckBits(t, "unchanged input", input, before)
		if b.tableUploads != uploads || len(b.tables) > 2 || len(b.live) != len(b.tables) || b.reads != b.calls {
			t.Fatalf("uploads=%d want=%d live=%d tables=%d reads=%d calls=%d", b.tableUploads, uploads, len(b.live), len(b.tables), b.reads, b.calls)
		}
	}

	t.Run("exact-keys-bounded-ownership-and-rebinding", func(t *testing.T) {
		s, b := newSession(t, &Model{})
		rotate := s.v41InverseOutputRoPEFunc()
		cos, sin := []float32{1, 0.5}, []float32{0, -0.25}
		check(t, b, rotate, 0, cos, sin, 1)
		check(t, b, rotate, 39, append([]float32(nil), cos...), append([]float32(nil), sin...), 1)
		// Equal floating-point values are not equal tables: preserve signed zero.
		sin[0] = math.Float32frombits(0x80000000)
		check(t, b, rotate, 1, cos, sin, 2)
		sin[0] = 0
		check(t, b, rotate, 2, cos, sin, 2)
		// A changed value at the same host address must evict, never reuse stale data.
		cos[0] = 0.75
		check(t, b, rotate, 2, cos, sin, 3)
		check(t, b, rotate, 2, cos[:1], sin[:1], 4)
		s.cacheGeometryEpoch++
		check(t, b, rotate, 2, cos[:1], sin[:1], 5)
		s.M = &Model{}
		check(t, b, rotate, 2, cos[:1], sin[:1], 6)
		// The older callback's entry must not survive replacement by another binding.
		rebound := s.v41InverseOutputRoPEFunc()
		check(t, b, rebound, 2, cos[:1], sin[:1], 7)
		check(t, b, rebound, 2, cos, sin, 8)
		check(t, b, rotate, 2, cos[:1], sin[:1], 9)
		s.Close()
		if len(b.live) != 0 || len(b.tables) != 0 {
			t.Fatal("session close retained inverse table storage")
		}
		for _, n := range b.frees {
			if n != 1 {
				t.Fatalf("inverse allocation freed %d times", n)
			}
		}
	})

	t.Run("pinned-decode-two-theta-schedule", func(t *testing.T) {
		_, cfg := readDeepSeekV41Config(t)
		s, b := newSession(t, &Model{Cfg: cfg})
		rotate := s.v41InverseOutputRoPEFunc()
		input := make([]float32, cfg.HeadDim)
		for i := range input {
			input[i] = float32(i%13-6) * 0.0625
		}
		for _, pos := range []int{7, 8} {
			before := b.tableUploads
			for layer := 0; layer < cfg.NumLayers; layer++ {
				cos, sin := v41InverseOutputTable(cfg, layer, pos)
				got, err := rotate(layer, input, cos, sin, 1, cfg.HeadDim, cfg.QKRopeHeadDim)
				if err != nil {
					t.Fatal(err)
				}
				want := append([]float32(nil), input...)
				applyRopeTailInterleaved(want, cos, sin, cfg.QKRopeHeadDim)
				v41TailRoPECheckBits(t, "pinned inverse", got, want)
			}
			if n := b.tableUploads - before; n != 2 || len(b.tables) != 2 {
				t.Fatalf("position %d table uploads=%d live=%d want two", pos, n, len(b.tables))
			}
		}
		if b.calls != 2*cfg.NumLayers || b.reads != b.calls || len(b.live) != 2 {
			t.Fatal("table reuse changed dispatch/readback count or transient ownership")
		}
	})

	t.Run("session-and-backend-isolation", func(t *testing.T) {
		m := &Model{}
		a, b := newSession(t, m)
		peer := &Session{M: m, Backend: b}
		t.Cleanup(peer.Close)
		other, c := newSession(t, m)
		cos, sin := []float32{1}, []float32{0}
		check(t, b, a.v41InverseOutputRoPEFunc(), 0, cos, sin, 1)
		check(t, b, peer.v41InverseOutputRoPEFunc(), 0, cos, sin, 2)
		check(t, c, other.v41InverseOutputRoPEFunc(), 0, cos, sin, 1)
		a.Close()
		if len(b.live) != 1 || len(c.live) != 1 {
			t.Fatal("closing one session freed another owner's table")
		}
		check(t, b, peer.v41InverseOutputRoPEFunc(), 0, cos, sin, 3)
	})

	for _, site := range []string{"table upload", "inverse-rope", "output readback"} {
		t.Run("selected-failure-"+site, func(t *testing.T) {
			s, b := newSession(t, &Model{})
			rotate := s.v41InverseOutputRoPEFunc()
			cos, sin := []float32{1}, []float32{0}
			check(t, b, rotate, 0, cos, sin, 1)
			cos[0] = 0.5
			check(t, b, rotate, 0, cos, sin, 2)
			if site == "table upload" {
				cos[0] = 0.75 // force replacement of one of two live entries
			}
			b.faultSite, b.fault = site, &compute.BackendError{Backend: "inverse-residency", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			_, err := rotate(0, make([]float32, 8), cos, sin, 1, 8, 2)
			var closed *BackendForwardOperationError
			if !errors.Is(err, b.fault) || !errors.As(err, &closed) || closed.Path != "v41-inverse-output-rope" || closed.Stage != site || !s.BackendSessionClosed() {
				t.Fatalf("selected failure lost identity: %v", err)
			}
			if len(b.live) != 0 || len(b.tables) != 0 {
				t.Fatal("selected failure retained cached or transient storage")
			}
			calls := b.calls
			if retry := recoverError(func() { _, _ = rotate(0, make([]float32, 8), cos, sin, 1, 8, 2) }); retry != s.halFailure || b.calls != calls {
				t.Fatalf("closed callback retried selected work: %v", retry)
			}
			for _, n := range b.frees {
				if n != 1 {
					t.Fatalf("failed inverse allocation freed %d times", n)
				}
			}
		})
	}
}
