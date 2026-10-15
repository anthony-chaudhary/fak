package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41EngramCacheFixture struct {
	m        *Model
	s        *Session
	be       *v41MHCTestBackend
	layer    int
	geometry v41EngramGeometry
	gathered *v41EngramGatheredRows
	row      []float32
	reader   *v41GroupedPanelCacheReader
	key      string
}

func newV41EngramCacheFixture(t *testing.T) *v41EngramCacheFixture {
	t.Helper()
	m := v41EngProjFixture(t, "Q2_K", false)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	be := v41EngramPoolBackend(true, false)
	s := v41DenseTestSession(t, m, be)
	layer := m.Cfg.DeepSeekV41.EngramLayerIDs[0]
	g, ids, err := m.v41EngramLayerRows(layer, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	gathered, err := m.v41EngramGatherRows(layer, 1, ids, float32(m.Cfg.RMSNormEps), g)
	if err != nil {
		t.Fatal(err)
	}
	row, err := gathered.decodeHost()
	if err != nil {
		t.Fatal(err)
	}
	name := layerName(layer, "engram_kv.weight")
	q := m.kqw[name]
	reader := &v41GroupedPanelCacheReader{data: append([]byte(nil), q.raw...)}
	q.lazy, q.raw = &LazyQ4KRange{Reader: reader, Bytes: len(reader.data)}, nil
	key := fmt.Sprintf("v41-engram:%s:%v:%d:%d", name, compute.Q2_K, (g.hc+1)*g.H, g.cols*g.dim)
	return &v41EngramCacheFixture{m, s, be, layer, g, gathered, row, reader, key}
}

func (f *v41EngramCacheFixture) call(s *Session, mode string) ([]float32, bool, error) {
	g := f.geometry
	if mode == "projection" {
		out, outcome, err := s.v41EngramProjectionFunc()(f.layer, f.row, (g.hc+1)*g.H, g.cols*g.dim)
		return out, outcome != v41ProjectionDeclined, err
	}
	panel := compute.NewF32(f.be, []int{1, 4, g.H}, make([]float32, 4*g.H))
	streams, err := newV41DeviceStreams(f.be, panel)
	if err != nil {
		f.be.Free(panel)
		return nil, false, err
	}
	defer streams.Free()
	handled, err := s.v41EngramGateFuncWith(nil)(f.layer, streams, f.gathered, 1, g, float32(f.m.Cfg.RMSNormEps))
	if err != nil || !handled {
		return nil, handled, err
	}
	// The recorder intentionally lends readback storage; own it before Free.
	return append([]float32(nil), f.be.Read(streams.Tensor())...), true, nil
}

// fak-test:runtime medium est=5s lane=default
func TestV41EngramCachedWeights(t *testing.T) {
	for _, mode := range []string{"projection", "gate"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EngramCacheFixture(t)
			other := "gate"
			if mode == "gate" {
				other = "projection"
			}
			if _, handled, err := f.call(f.s, other); err != nil || !handled {
				t.Fatalf("opposite-arm warmup: handled=%t err=%v", handled, err)
			}
			weight, ok := f.s.halW[f.key]
			if !ok {
				t.Fatal("opposite arm used a different projection cache key")
			}
			f.be.paramFrees = map[compute.Buffer]int{weight.Buf(): 0}
			want, handled, err := f.call(f.s, mode)
			if err != nil || !handled {
				t.Fatalf("same-session control: handled=%t err=%v", handled, err)
			}
			borrower := v41DenseTestSession(t, f.m, f.be)
			borrower.cacheGeometryEpoch++ // Immutable residency is model/backend scoped.
			got, handled, err := f.call(borrower, mode)
			if err != nil || !handled || !reflect.DeepEqual(got, want) {
				t.Fatalf("fresh-session hit: handled=%t err=%v parity=%t", handled, err, reflect.DeepEqual(got, want))
			}
			if f.reader.calls != 1 || f.reader.bytes != len(f.reader.data) {
				t.Fatalf("lazy weight reads=%d bytes=%d, want 1/%d", f.reader.calls, f.reader.bytes, len(f.reader.data))
			}
			if borrower.halW[f.key].Buf() != weight.Buf() {
				t.Fatal("fresh session changed the model-owned projection handle")
			}
			if _, ok := borrower.borrowedHALW[f.key]; !ok {
				t.Fatal("fresh session lacks the borrowed projection marker")
			}
			borrower.Close()
			f.s.Close()
			if f.be.paramFrees[weight.Buf()] != 0 {
				t.Fatal("session close freed a model-owned projection")
			}
			if err := f.m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			if f.be.paramFrees[weight.Buf()] != 1 || len(f.be.outputs) != 0 {
				t.Fatalf("model frees=%d live gate outputs=%d", f.be.paramFrees[weight.Buf()], len(f.be.outputs))
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41EngramCachedWeightRefusal(t *testing.T) {
	for _, mode := range []string{"projection", "gate"} {
		for _, reason := range []string{"cold-nonfinite", "warm-descriptor", "warm-shape", "warm-input"} {
			if mode == "gate" && reason == "warm-input" {
				continue // Gate input/norm admission has its own gathered-row contract.
			}
			t.Run(mode+"/"+reason, func(t *testing.T) {
				f := newV41EngramCacheFixture(t)
				var weight compute.Tensor
				if reason != "cold-nonfinite" {
					if _, handled, err := f.call(f.s, mode); err != nil || !handled {
						t.Fatalf("warmup: handled=%t err=%v", handled, err)
					}
					weight = f.s.halW[f.key]
					f.be.paramFrees = map[compute.Buffer]int{weight.Buf(): 0}
				}
				borrower := v41DenseTestSession(t, f.m, f.be)
				q := f.m.kqw[layerName(f.layer, "engram_kv.weight")]
				switch reason {
				case "cold-nonfinite":
					binary.LittleEndian.PutUint16(f.reader.data[q2kBlockBytes-4:], 0x7c00)
				case "warm-descriptor":
					q.nblk++
				case "warm-shape":
					q.out++
				case "warm-input":
					f.row[0] = float32(math.NaN())
				}
				uploads, gates := len(f.be.uploads), f.be.calls["engram"]
				got, handled, err := f.call(borrower, mode)
				if reason == "cold-nonfinite" {
					var closed *BackendForwardOperationError
					path := "v41-engram-projection"
					if mode == "gate" {
						path = "v41-engram-gate"
					}
					if !handled || !errors.Is(err, errV41ProjectionResult) || !errors.As(err, &closed) || closed.Path != path || closed.Stage != "payload" || !borrower.BackendSessionClosed() {
						t.Fatalf("cold invalid payload: handled=%t err=%v closed=%t", handled, err, borrower.BackendSessionClosed())
					}
				} else if handled || err != nil || borrower.BackendSessionClosed() {
					t.Fatalf("warm admission refusal: handled=%t err=%v closed=%t", handled, err, borrower.BackendSessionClosed())
				}
				if got != nil || len(f.be.uploads) != uploads || f.be.calls["engram"] != gates || len(borrower.halW) != 0 {
					t.Fatal("refusal adopted a handle or performed projection/gate work")
				}
				if f.reader.calls != 1 || f.reader.bytes != len(f.reader.data) {
					t.Fatalf("refusal lazy reads=%d bytes=%d", f.reader.calls, f.reader.bytes)
				}
				if weight.Buf() != nil && f.be.paramFrees[weight.Buf()] != 0 {
					t.Fatal("refusal freed a model-owned projection")
				}
			})
		}
	}
}
