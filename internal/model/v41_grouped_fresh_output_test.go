package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
)

var v41GroupedFreshAllocationSink [][]float32

// The control retains the public-to-package legacy helper, including its
// externally visible zero rows. Both arms call the actual shared contraction.
// Handled routed rows exercise planning, BF16 publication, and full replay
// without introducing backend allocations into the paired comparison.
// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedFreshAvoidsDeadFullRows(t *testing.T) {
	const tokens, hidden = 32, 64
	cfg := Config{HiddenSize: hidden, MoEIntermediateSize: 4, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	picks := make([][]routePick, tokens)
	inputs := make([][]float32, tokens)
	results := make([][]float32, tokens)
	for i := range results {
		results[i] = make([]float32, hidden)
		inputs[i] = results[i]
		picks[i] = []routePick{{expert: 0, weight: 1}}
	}
	calls, routedCalls := 0, 0
	st := &v41ForwardState{routedExpert: func(_ int, _ string, x []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
		routedCalls++
		return x, v41RoutedExpertHandled, nil
	}}
	scratch := &v41ProjScratch{routedSumBinding: newV41RoutedSumBinding(func(_ int, _ []routePick, _ [][]float32, width int) ([]float32, error) {
		if width != hidden {
			panic("unexpected hidden width")
		}
		row := results[calls]
		calls++
		return row, nil
	})}
	run := func(fresh bool) {
		calls, routedCalls = 0, 0
		var err error
		if fresh {
			v41GroupedFreshAllocationSink, err = m.v41ContractRoutedGroupedFresh(0, inputs, picks, scratch, cfg, st)
		} else {
			out := make([][]float32, tokens)
			err = m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, out, st)
			v41GroupedFreshAllocationSink = out
		}
		if err != nil || calls != tokens || routedCalls != tokens {
			panic("allocation fixture did not complete full replay")
		}
	}
	legacy := testing.AllocsPerRun(10, func() { run(false) })
	fresh := testing.AllocsPerRun(10, func() { run(true) })
	if legacy-fresh != tokens {
		t.Fatalf("removed full-row allocations=%g, want %d (legacy=%g fresh=%g)", legacy-fresh, tokens, legacy, fresh)
	}
	// Observe the actual legacy output buffers before its first sum overwrites
	// them. The fresh entry never exposes its private table at this point.
	out := make([][]float32, tokens)
	bytes := 0
	scratch.setRoutedSum(func(_ int, _ []routePick, _ [][]float32, _ int) ([]float32, error) {
		if bytes == 0 {
			for _, row := range out {
				bytes += 4 * len(row)
			}
		}
		return results[0], nil
	})
	if err := m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, out, st); err != nil {
		t.Fatal(err)
	}
	if bytes != tokens*hidden*4 {
		t.Fatalf("observed eager payload=%d, want %d", bytes, tokens*hidden*4)
	}
}

func v41GroupedFreshBitsEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				return false
			}
		}
	}
	return true
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedFreshFullAndReducedParity(t *testing.T) {
	for _, full := range []bool{false, true} {
		t.Run(fmt.Sprint(full), func(t *testing.T) {
			cfg := Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: 32}
			if full {
				cfg.HeadDim = v41KVLoraRank
			}
			m := &Model{Cfg: cfg}
			inputs := [][]float32{{1, .5, -.25, 0}, {-.5, 1, 0, .25}}
			picks := [][]routePick{{{expert: 2, weight: .5}, {expert: 0, weight: .25}}, {{expert: 2, weight: .75}, {expert: 2, weight: .125}}}
			run := func(fresh bool) ([][]float32, []string, error) {
				var trace []string
				st := &v41ForwardState{
					expertGateUp: func(_ int, stem string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
						trace = append(trace, "gate:"+stem)
						return []float32{1 + x[0], .5, -.25}, v41GateUpHandled, nil
					},
					expertDown: func(_ int, stem string, h []float32) ([]float32, v41ExpertDownOutcome, error) {
						trace = append(trace, "down:"+stem)
						return []float32{h[0], h[1], h[2], -0.5}, v41DownHandled, nil
					},
				}
				scratch := &v41ProjScratch{}
				if full {
					scratch.setRoutedSum(func(l int, p []routePick, rows [][]float32, h int) ([]float32, error) {
						trace = append(trace, "sum")
						return v41FullRoutedSum(l, p, rows, h)
					})
				}
				if fresh {
					out, err := m.v41ContractRoutedGroupedFresh(0, inputs, picks, scratch, cfg, st)
					return out, trace, err
				}
				out := make([][]float32, len(picks))
				err := m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, out, st)
				return out, trace, err
			}
			want, wt, we := run(false)
			got, gt, ge := run(true)
			if we != nil || ge != nil || !reflect.DeepEqual(wt, gt) || !v41GroupedFreshBitsEqual(got, want) {
				t.Fatalf("parity: old=%v next=%v traces=%v/%v", we, ge, wt, gt)
			}
			got[0][0] = 99
			if got[1][0] == 99 || want[0][0] == 99 {
				t.Fatal("independent output rows aliased")
			}
		})
	}
}

func v41GroupedFreshRecover(run func()) (p any) { defer func() { p = recover() }(); run(); return nil }

// The old output parameter is unchanged on error. The new private entry
// exposes no partial result, with identical callback order and primary fault.
// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedFreshFailureAndPanicParity(t *testing.T) {
	for _, mode := range []string{"routed-error", "sum-error", "routed-panic", "sum-panic"} {
		t.Run(mode, func(t *testing.T) {
			cfg := Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank}
			m := &Model{Cfg: cfg}
			inputs := [][]float32{{1, 0, 0, 0}, {2, 0, 0, 0}}
			picks := [][]routePick{{{expert: 0, weight: 1}}, {{expert: 0, weight: 1}}}
			fault := errors.New("selected grouped fault")
			run := func(fresh bool) (out [][]float32, trace []string, err error, p any) {
				calls, sums := 0, 0
				st := &v41ForwardState{routedExpert: func(_ int, _ string, x []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
					calls++
					trace = append(trace, fmt.Sprintf("routed%d", calls))
					if calls == 2 {
						if mode == "routed-error" {
							return nil, v41RoutedExpertError, fault
						}
						if mode == "routed-panic" {
							panic(fault)
						}
					}
					return []float32{x[0], 1, 2, 3}, v41RoutedExpertHandled, nil
				}}
				scratch := &v41ProjScratch{routedSumBinding: newV41RoutedSumBinding(func(l int, picks []routePick, rows [][]float32, h int) ([]float32, error) {
					sums++
					trace = append(trace, fmt.Sprintf("sum%d", sums))
					if sums == 2 {
						if mode == "sum-error" {
							return nil, fault
						}
						if mode == "sum-panic" {
							panic(fault)
						}
					}
					return v41FullRoutedSum(l, picks, rows, h)
				})}
				p = v41GroupedFreshRecover(func() {
					if fresh {
						out, err = m.v41ContractRoutedGroupedFresh(0, inputs, picks, scratch, cfg, st)
					} else {
						out = make([][]float32, len(picks))
						err = m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, out, st)
					}
				})
				return
			}
			old, ot, oe, op := run(false)
			next, nt, ne, np := run(true)
			if !reflect.DeepEqual(ot, nt) || op != np || errors.Is(oe, fault) != errors.Is(ne, fault) || reflect.TypeOf(oe) != reflect.TypeOf(ne) || fmt.Sprint(oe) != fmt.Sprint(ne) || next != nil {
				t.Fatalf("fault changed: old=%v/%v next=%v/%v traces=%v/%v", oe, op, ne, np, ot, nt)
			}
			if len(old) != 2 || len(old[0]) != 4 || len(old[1]) != 4 {
				t.Fatal("legacy eager partial-output contract changed")
			}
			if op == nil && (!errors.Is(oe, fault) || !errors.Is(ne, fault)) {
				t.Fatal("selected error lost")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41GroupedFreshBoundaryAndLegacyAdmission(t *testing.T) {
	cfg := Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	called := 0
	scratch := &v41ProjScratch{routedSumBinding: newV41RoutedSumBinding(func(_ int, _ []routePick, _ [][]float32, h int) ([]float32, error) {
		called++
		return make([]float32, h), nil
	})}
	if out, err := m.v41ContractRoutedGroupedFresh(0, nil, nil, scratch, cfg, nil); err != nil || len(out) != 0 || called != 0 {
		t.Fatalf("empty panel: %v %v", out, err)
	}
	picks := make([][]routePick, 2)
	old := make([][]float32, 1)
	p := v41GroupedFreshRecover(func() { _ = m.v41ContractRoutedGrouped(0, nil, picks, scratch, cfg, old, nil) })
	if p == nil || called != 0 || len(old[0]) != 4 {
		t.Fatal("legacy short output did not panic during eager initialization")
	}
	cfg.HiddenSize = -1
	oldP := v41GroupedFreshRecover(func() { _ = m.v41ContractRoutedGrouped(0, nil, picks, scratch, cfg, make([][]float32, 2), nil) })
	newP := v41GroupedFreshRecover(func() { _, _ = m.v41ContractRoutedGroupedFresh(0, nil, picks, scratch, cfg, nil) })
	if oldP == nil || newP == nil || fmt.Sprint(oldP) != fmt.Sprint(newP) || called != 0 {
		t.Fatal("nonpositive hidden-width allocation precedence changed")
	}
}

// Exercise the sole changed production callsite through public full forward.
// V4.1 emits positional expert events before contraction. Those events own
// their callback buffers.
// fak-test:runtime medium est=5s lane=default
func TestV41GroupedFreshPublicFullObserverParity(t *testing.T) {
	prior := v41ForceTokenMajor
	defer func() { v41ForceTokenMajor = prior }()
	m := v41FullStepPatchedNorms(t)
	defer m.SetExpertRouteEventObserver(nil)
	run := func(tokenMajor bool) (*Activations, []ExpertRouteEvent) {
		v41ForceTokenMajor = tokenMajor
		var events []ExpertRouteEvent
		m.SetExpertRouteEventObserver(func(ev ExpertRouteEvent) {
			saved := ev
			saved.Experts = append([]int(nil), ev.Experts...)
			events = append(events, saved)
			for i := range ev.Experts {
				ev.Experts[i] = -999
			}
		})
		return m.Forward([]int{1, 2}), events
	}
	got, gotEvents := run(false)
	want, wantEvents := run(true)
	if !v41GroupedFreshBitsEqual(got.Logits, want.Logits) || !reflect.DeepEqual(gotEvents, wantEvents) {
		t.Fatal("full grouped/token-major logits or route events differ")
	}
	if len(gotEvents) != 2*m.Cfg.NumLayers || len(gotEvents[0].Experts) != m.Cfg.NumExpertsPerTok {
		t.Fatal("public full fixture did not observe every routed token")
	}
}
