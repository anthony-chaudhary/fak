package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"reflect"
	"testing"
)

type v41DownPanelRowWitness struct {
	*v41ExpertPanelRowWitness
	enabledDown                    bool
	refuseDown                     int
	downProbes, directDown         []int
	ordinaryDown                   int
	failAt, aliasAt, statusAliasAt int
	failure                        error
}

func (b *v41DownPanelRowWitness) SupportsV41RoutedDownPanelRow(p compute.Tensor, row int) bool {
	b.downProbes = append(b.downProbes, row)
	return b.enabledDown && row != b.refuseDown && len(p.Shape) == 2 && row >= 0 && row < p.Shape[0]
}
func (b *v41DownPanelRowWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.ordinaryDown++
	return b.v41RoutedLayerTestBackend.V41BF16PublishDeferred(x, site)
}
func (b *v41DownPanelRowWitness) V41RoutedDownPanelRowDeferred(p compute.Tensor, row int) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.directDown = append(b.directDown, row)
	if !b.ownedTiles[p.Buf()] {
		return compute.Tensor{}, compute.V41DeferredStatus{}, errors.New("down panel retired early")
	}
	width := p.Shape[1]
	values := b.Backend.Read(p)
	// Host-backed fixture transport is not a device gather or timing measurement.
	x := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, values[row*width:(row+1)*width]), compute.F32)
	defer b.Backend.Free(x)
	out, st, err := b.v41RoutedLayerTestBackend.V41BF16PublishDeferred(x, compute.V41BF16PublishRoutedDown)
	if len(b.directDown) == b.aliasAt {
		b.v41ExpertTileWitness.Free(out)
		out = p
	}
	if len(b.directDown) == b.statusAliasAt {
		b.v41ExpertTileWitness.Free(st.Record)
		st.Record = p
	}
	if len(b.directDown) == b.failAt {
		err = b.failure
	}
	return out, st, err
}

// fak-test:runtime medium est=10s lane=default
func TestV41DownPanelRowConsumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 32), make([]float32, 32)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r*3+i)%13-6) / 16
		}
		weights[r] = float32(r+1) / 64
	}
	makeSession := func(on bool, refuse int) (*Session, *v41DownPanelRowWitness) {
		base := &v41ExpertPanelRowWitness{v41ExpertGateUp32Witness: &v41ExpertGateUp32Witness{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, downTiles: true}, grid: true}, enabledRows: true, refuseRow: -1, frees: map[compute.Buffer]int{}}
		b := &v41DownPanelRowWitness{v41ExpertPanelRowWitness: base, enabledDown: on, refuseDown: refuse}
		s, e := m.NewBackendSessionChecked(b)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	checkOwners := func(b *v41DownPanelRowWitness) {
		t.Helper()
		if len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatal("temporary/status leak")
		}
		for _, n := range b.frees {
			if n != 1 {
				t.Fatalf("owner freed %d times", n)
			}
		}
	}
	for _, count := range []int{3, 9, 32} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			cs, cb := makeSession(false, -1)
			want, o, e := cs.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:count], weights[:count])
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
				var probes []int
				direct := 0
				for _, n := range b.downRows {
					for r := 0; r < n; r++ {
						probes = append(probes, r)
						if r != refuse {
							direct++
						}
					}
				}
				if !reflect.DeepEqual(b.downProbes, probes) || len(b.directDown) != direct || b.ordinaryDown != count-direct || cb.gathers-b.gathers != direct || !reflect.DeepEqual(b.downRows, cb.downRows) || b.publishes != count || b.fences != 1 {
					t.Fatalf("down calls=%v ordinary=%d gathers=%d/%d probes=%v", b.directDown, b.ordinaryDown, b.gathers, cb.gathers, b.downProbes)
				}
				checkOwners(b)
			}
			checkOwners(cb)
		})
	}
	for _, kind := range []string{"partial", "output-alias", "status-alias", "earlier-down"} {
		t.Run(kind, func(t *testing.T) {
			s, b := makeSession(true, -1)
			failure := &compute.BackendError{Backend: "test", Site: "down panel row", Err: errors.New("selected failure")}
			const faultCall = 12
			b.failAt, b.failure = faultCall, failure
			if kind == "output-alias" {
				b.failAt = 0
				b.aliasAt = faultCall
			}
			if kind == "status-alias" {
				b.failAt = 0
				b.statusAliasAt = faultCall
			}
			if kind == "earlier-down" {
				b.publishFaultAt = map[int]bool{1: true}
			}
			out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			if out != nil || o != v41RoutedExpertError || e == nil || !s.BackendSessionClosed() || len(b.directDown) != faultCall || b.ordinaryDown != 0 {
				t.Fatalf("selected failure replay/result %v/%v", o, e)
			}
			if kind == "partial" && !errors.Is(e, failure) {
				t.Fatalf("lost primary: %v", e)
			}
			if kind == "earlier-down" {
				var numeric *V41ExpertOperationError
				if !errors.As(e, &numeric) {
					t.Fatalf("earlier numeric lost: %v", e)
				}
			}
			checkOwners(b)
		})
	}
}
