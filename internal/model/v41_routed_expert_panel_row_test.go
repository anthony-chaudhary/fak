package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"reflect"
	"testing"
)

// CPU orchestration recorder. Its temporary host-backed rows are fixture
// implementation details, not device row gathers or hardware timing evidence.
type v41ExpertPanelRowWitness struct {
	*v41ExpertGateUp32Witness
	enabledRows        bool
	refuseRow          int
	probes, directRows []int
	scalarCalls        int
	partialAt          int
	partialError       error
	aliasAt            int
	frees              map[compute.Buffer]int
}

func (b *v41ExpertPanelRowWitness) SupportsV41RoutedSwiGLUPanelRow(g, u compute.Tensor, row int) bool {
	b.probes = append(b.probes, row)
	return b.enabledRows && row != b.refuseRow && len(g.Shape) == 2 && reflect.DeepEqual(g.Shape, u.Shape) && row >= 0 && row < g.Shape[0]
}
func (b *v41ExpertPanelRowWitness) V41RoutedSwiGLUDeferred(g, u compute.Tensor, l, w float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.scalarCalls++
	return b.v41ExpertDownTileWitness.V41RoutedSwiGLUDeferred(g, u, l, w)
}
func (b *v41ExpertPanelRowWitness) V41RoutedSwiGLUPanelRowDeferred(g, u compute.Tensor, row int, l, w float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.directRows = append(b.directRows, row)
	if !b.ownedTiles[g.Buf()] || !b.ownedTiles[u.Buf()] {
		return compute.Tensor{}, compute.V41DeferredStatus{}, errors.New("borrowed panel retired early")
	}
	width := g.Shape[1]
	gv, uv := b.Backend.Read(g), b.Backend.Read(u)
	gr := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, gv[row*width:(row+1)*width]), compute.F32)
	defer b.Backend.Free(gr)
	ur := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, uv[row*width:(row+1)*width]), compute.F32)
	defer b.Backend.Free(ur)
	out, st, err := b.v41ExpertDownTileWitness.V41RoutedSwiGLUDeferred(gr, ur, l, w)
	if len(b.directRows) == b.aliasAt {
		b.v41ExpertTileWitness.Free(out)
		out = g
	}
	if len(b.directRows) == b.partialAt {
		err = b.partialError
	}
	return out, st, err
}
func (b *v41ExpertPanelRowWitness) Free(x compute.Tensor) {
	if x.Buf() != nil {
		b.frees[x.Buf()]++
	}
	b.v41ExpertTileWitness.Free(x)
}

// fak-test:runtime medium est=10s lane=default
func TestV41ExpertPanelRowConsumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 32), make([]float32, 32)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[row] {
			inputs[row][i] = float32((row*3+i)%13-6) / 16
		}
		weights[row] = float32(row+1) / 64
	}
	makeSession := func(enabled bool, refuse int) (*Session, *v41ExpertPanelRowWitness) {
		b := &v41ExpertPanelRowWitness{v41ExpertGateUp32Witness: &v41ExpertGateUp32Witness{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, downTiles: true}, grid: true}, enabledRows: enabled, refuseRow: refuse, frees: map[compute.Buffer]int{}}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	checkOwners := func(b *v41ExpertPanelRowWitness) {
		t.Helper()
		if len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatal("temporary or status leak")
		}
		for _, n := range b.frees {
			if n != 1 {
				t.Fatalf("owner freed %d times", n)
			}
		}
	}
	for _, count := range []int{3, 9, 32} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			control, cb := makeSession(false, -1)
			want, o, e := control.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
			if o != v41RoutedExpertHandled || e != nil {
				t.Fatal(e)
			}
			for _, refuse := range []int{-1, 1} {
				s, b := makeSession(true, refuse)
				got, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
				if o != v41RoutedExpertHandled || e != nil || len(got) != count {
					t.Fatalf("%v/%v", o, e)
				}
				for r := range got {
					if !v41SameBits(got[r], want[r]) {
						t.Fatalf("row %d parity", r)
					}
				}
				wantDirect := count
				if refuse >= 0 {
					wantDirect--
				}
				if len(b.directRows) != wantDirect || b.scalarCalls != count-wantDirect || cb.gathers-b.gathers != 2*wantDirect || !reflect.DeepEqual(b.downRows, cb.downRows) || b.publishes != count || b.fences != 1 {
					t.Fatalf("calls direct=%v scalar=%d gathers=%d/%d down=%v", b.directRows, b.scalarCalls, b.gathers, cb.gathers, b.downRows)
				}
				for r, p := range b.probes {
					if p != r {
						t.Fatalf("exact row probe %d=%d", r, p)
					}
				}
				if len(b.probes) != count {
					t.Fatal("probe count")
				}
				checkOwners(b)
			}
			checkOwners(cb)
		})
	}
	for _, kind := range []string{"partial", "alias", "earlier-down"} {
		t.Run(kind, func(t *testing.T) {
			s, b := makeSession(true, -1)
			failure := &compute.BackendError{Backend: "test", Site: "panel row", Err: errors.New("selected failure")}
			b.partialAt, b.partialError = 12, failure
			if kind == "alias" {
				b.partialAt = 0
				b.aliasAt = 12
			}
			if kind == "earlier-down" {
				b.publishFaultAt = map[int]bool{1: true}
			}
			out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			if out != nil || o != v41RoutedExpertError || e == nil || !s.BackendSessionClosed() || len(b.directRows) != 12 || b.scalarCalls != 0 {
				t.Fatalf("selected failure replay/result %v/%v", o, e)
			}
			if kind == "partial" && !errors.Is(e, failure) {
				t.Fatalf("lost primary %v", e)
			}
			if kind == "earlier-down" {
				var numeric *V41ExpertOperationError
				if !errors.As(e, &numeric) {
					t.Fatalf("earlier numeric verdict lost: %v", e)
				}
			}
			checkOwners(b)
		})
	}
}
