package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU contract recorder. A finished cohort releases its tile panels before
// the next cohort starts; keep the inherited single-cohort witness scoped to
// that lifetime. This records scheduling, not physical GPU performance.
type v41EPOwnerPanelBackend struct {
	*v41ExpertDownTileWitness
	declineAfterFirst bool
}

func (b *v41EPOwnerPanelBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	verdicts, values, err := b.v41ExpertDownTileWitness.V41ReadDeferred(statuses, rows)
	b.panels = nil
	if b.declineAfterFirst {
		b.enabled = false
	}
	return verdicts, values, err
}

// fak-test:runtime medium est=5s lane=default
func TestV41EPOwnerPanelRows(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	newOwner := func(t *testing.T, rank int, cohorts bool) (*Session, TPPlan, *v41ForwardState, *v41EPOwnerPanelBackend, *int) {
		t.Helper()
		s, owners, _ := v41EPOwnerPackedFixture(rank)
		t.Cleanup(func() { _ = s.M.CloseWeights() })
		b := &v41EPOwnerPanelBackend{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{
			v41ExpertTileWitness: &v41ExpertTileWitness{
				v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: cohorts},
				tiles:                  true, ownedTiles: map[compute.Buffer]bool{},
			}, downTiles: true,
		}}
		s.Backend = b
		t.Cleanup(s.Close)
		routed, err := s.v41EPOwnerDeviceBinding(owners, s.v41RoutedExpertFunc())
		if err != nil || routed == nil {
			t.Fatalf("owner binding: %v", err)
		}
		calls := new(int)
		st := &v41ForwardState{callbackOwner: s, routedExpert: func(layer int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
			*calls++
			return routed(layer, stem, x, weight)
		}}
		return s, owners, st, b, calls
	}
	inputs := make([][]float32, v41EPOwnerPanelTokens)
	picks := make([][]routePick, len(inputs))
	for token := range inputs {
		inputs[token] = make([]float32, 256)
		for i := range inputs[token] {
			inputs[token][i] = float32((token*3+i)%13-6) / 16
		}
		picks[token] = []routePick{{expert: 1, weight: float32(token+1) / 64}, {expert: 6, weight: .125}, {expert: 0, weight: .75}}
	}
	// Expert 1 has 33 rows, including two distinct slots from token 0.
	picks[0] = append(picks[0], routePick{expert: 1, weight: .875})
	t.Run("exact_owner_slots_and_bounded_tiles", func(t *testing.T) {
		s, owners, st, b, scalarCalls := newOwner(t, 0, true)
		control, _, cst, cb, controlCalls := newOwner(t, 0, false)
		got, err := s.M.v41EPOwnerPanelRows(0, owners, 0, picks, inputs, &v41ProjScratch{}, st)
		if err != nil {
			t.Fatal(err)
		}
		// The oracle is the original scalar owner loop, token-major then slot-
		// major. It bypasses the new grouping helper and never rank-pre-sums.
		want := make([][]float32, len(picks))
		scratch := &v41ProjScratch{}
		for token, tokenPicks := range picks {
			for _, pick := range tokenPicks {
				if pick.expert >= owners.Shards[0].Hi {
					continue
				}
				stem := "ffn.experts." + itoa(pick.expert)
				row, cause := control.M.v41FullRoutedExpertPublished(0, stem, inputs[token], pick.weight, control.M.Cfg, cst.routedExpert, nil, nil,
					func() ([]float32, []float32, []float32, error) {
						return control.M.v41ExpertTripleInto(0, stem, scratch, false)
					},
					func() ([]float32, error) { return control.M.hostExpertDown(0, stem, scratch) }, false, nil)
				if cause != nil {
					t.Fatal(cause)
				}
				want[token] = append(want[token], row...)
			}
			if !v41SameBits(got[token], want[token]) {
				t.Fatalf("token %d contribution bits or original slots changed", token)
			}
		}
		if !reflect.DeepEqual(b.cohortRows, []int{31, 2, 32}) || *scalarCalls != 0 || *controlCalls != 65 || b.fences != 3 || len(cb.tileRows) != 0 {
			t.Fatalf("cohorts=%v scalar=%d/%d fences=%d", b.cohortRows, *scalarCalls, *controlCalls, b.fences)
		}
		if len(b.tileRows) == 0 || len(b.downRows) == 0 {
			t.Fatal("owner grouping never reached the existing packed tiles")
		}
		for _, rows := range append(append([]int(nil), b.tileRows...), b.downRows...) {
			if rows != 2 && rows != 4 && rows != 8 {
				t.Fatalf("unbounded packed tile %d", rows)
			}
		}
		if v41SameBits(got[0][:256], got[0][512:768]) {
			t.Fatal("duplicate-slot coefficient witness is vacuous")
		}
		if len(b.pending) != 0 || len(b.ownedTiles) != 0 {
			t.Fatal("cohort retained temporary status or tile ownership")
		}
		before := append([]float32(nil), got[1]...)
		s.Close()
		inputs[0][0] = 17
		got[0][0] = 19
		if !v41SameBits(got[1], before) || !v41SameBits(got[1], want[1]) {
			t.Fatal("returned token rows alias inputs, another token or closed backend storage")
		}
		inputs[0][0] = -.375
	})
	t.Run("empty_owner_and_preselection_decline", func(t *testing.T) {
		s, owners, st, b, scalarCalls := newOwner(t, 1, true)
		out, err := s.M.v41EPOwnerPanelRows(0, owners, 1, picks, inputs, &v41ProjScratch{}, st)
		if err != nil || len(out) != len(inputs) {
			t.Fatalf("empty owner: %v", err)
		}
		for _, row := range out {
			if row == nil || len(row) != 0 {
				t.Fatal("empty owner lost its explicit token row")
			}
		}
		if *scalarCalls != 0 || b.fences != 0 || len(s.halW) != 0 {
			t.Fatal("empty owner admitted an expert")
		}
		s, owners, st, b, scalarCalls = newOwner(t, 0, false)
		out, err = s.M.v41EPOwnerPanelRows(0, owners, 0, picks[:2], inputs[:2], &v41ProjScratch{}, st)
		if err != nil || len(out) != 2 || *scalarCalls != 5 || len(b.cohortRows) != 0 || len(b.tileRows) != 0 {
			t.Fatalf("preselection scalar fallback: calls=%d, error=%v", *scalarCalls, err)
		}
	})
	for _, mode := range []string{"selected_tile_error", "later_cohort_decline"} {
		t.Run(mode, func(t *testing.T) {
			s, owners, st, b, scalarCalls := newOwner(t, 0, true)
			failure := &compute.BackendError{Backend: "test", Site: "owner panel tile", Err: errors.New("selected tile failed")}
			if mode == "selected_tile_error" {
				b.failTileAt, b.tileFailure = 3, failure
			} else {
				b.declineAfterFirst = true
			}
			out, err := s.M.v41EPOwnerPanelRows(0, owners, 0, picks, inputs, &v41ProjScratch{}, st)
			if out != nil || !errors.Is(err, ErrV41ExpertParallel) || *scalarCalls != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
				t.Fatalf("selected work replayed or partial rows leaked: %v", err)
			}
			if mode == "selected_tile_error" && (!errors.Is(err, failure) || !s.BackendSessionClosed()) {
				t.Fatalf("selected fault lost or backend remained reusable: %v", err)
			}
			if mode == "later_cohort_decline" && !reflect.DeepEqual(b.cohortRows, []int{31}) {
				t.Fatalf("later decline continued after cohort %v", b.cohortRows)
			}
		})
	}
	t.Run("preflight_before_any_contraction", func(t *testing.T) {
		for _, mode := range []string{"layer", "rank", "owner_dimension", "token_bound", "pick_bound", "byte_bound", "overflow", "input_count", "input_width", "nonfinite_input", "expert", "coefficient"} {
			t.Run(mode, func(t *testing.T) {
				m := &Model{Cfg: Config{HiddenSize: 4, MoEIntermediateSize: 3, NumLayers: 1, NumExperts: 7}}
				owners, _ := ExpertParallelPlan(7, 3)
				layer, rank, calls := 0, 0, 0
				p := [][]routePick{{{expert: 0, weight: 1}}}
				x := [][]float32{{1, 0, 0, 0}}
				switch mode {
				case "layer":
					layer = 1
				case "rank":
					rank = 3
				case "owner_dimension":
					owners, _ = ExpertParallelPlan(8, 3)
				case "token_bound":
					p, x = make([][]routePick, 33), make([][]float32, 33)
				case "pick_bound":
					p[0] = make([]routePick, v41EPOwnerPanelPicks+1)
				case "byte_bound":
					m.Cfg.HiddenSize = v41EPOwnerPanelMaxBytes/4 + 1
				case "overflow":
					m.Cfg.HiddenSize = int(^uint(0) >> 1)
				case "input_count":
					x = nil
				case "input_width":
					x[0] = x[0][:3]
				case "nonfinite_input":
					x = append(x, []float32{float32(math.NaN()), 0, 0, 0})
					p = append(p, []routePick{{expert: 1, weight: 1}})
				case "expert":
					p[0][0].expert = 7
				case "coefficient":
					p[0][0].weight = float32(math.Inf(1))
				}
				st := &v41ForwardState{routedExpert: func(_ int, _ string, x []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
					calls++
					return x, v41RoutedExpertHandled, nil
				}}
				out, err := m.v41EPOwnerPanelRows(layer, owners, rank, p, x, &v41ProjScratch{}, st)
				if out != nil || !errors.Is(err, ErrV41ExpertParallel) || calls != 0 {
					t.Fatalf("invalid window reached contraction: output=%v calls=%d error=%v", out, calls, err)
				}
			})
		}
	})
}
