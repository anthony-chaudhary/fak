package model

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"testing"
)

// CPU arithmetic recorder: the paired method delegates each arm to the same
// scalar oracle. Its call count witnesses orchestration, not native GPU parity.
type v41ExpertPairWitness struct {
	*v41ExpertGateUp32Witness
	pairEnabled bool
	pairCalls   int
	pairFailure string
}

func (b *v41ExpertPairWitness) SupportsV41Q2KTokenPair(g, u compute.Tensor, rows int) bool {
	return b.pairEnabled && g.Dtype == compute.Q2_K && u.Dtype == compute.Q2_K && rows >= 2 && rows <= 32
}
func (b *v41ExpertPairWitness) V41Q2KTokenPair(g, u, x compute.Tensor, rows int) (compute.Tensor, compute.Tensor, error) {
	b.pairCalls++
	if b.pairFailure == "empty" {
		return compute.Tensor{}, compute.Tensor{}, errors.New("paired transport")
	}
	go_, err := b.v41ExpertTileWitness.V41Q2KTokenPanel(g, x, rows)
	if err != nil {
		return go_, compute.Tensor{}, err
	}
	if b.pairFailure == "gate-only" {
		return go_, compute.Tensor{}, errors.New("paired transport")
	}
	uo, err := b.v41ExpertTileWitness.V41Q2KTokenPanel(u, x, rows)
	if b.pairFailure == "both" {
		return go_, uo, errors.New("paired transport")
	}
	if b.pairFailure == "alias" {
		b.Free(uo)
		return go_, go_, nil
	}
	return go_, uo, err
}

// fak-test:runtime medium est=10s lane=default
func TestV41ExpertPairedQ2Consumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	fresh := func(enabled bool, failure string) (*Session, *v41ExpertPairWitness) {
		b := &v41ExpertPairWitness{v41ExpertGateUp32Witness: &v41ExpertGateUp32Witness{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, downTiles: true}, grid: true}, pairEnabled: enabled, pairFailure: failure}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	for _, rows := range []int{2, 4, 8, 15, 32} {
		inputs := make([][]float32, rows)
		weights := make([]float32, rows)
		for r := range inputs {
			inputs[r] = make([]float32, m.Cfg.HiddenSize)
			for i := range inputs[r] {
				inputs[r][i] = float32((r+i)%13-6) / 16
			}
			weights[r] = float32(r+1) / 64
		}
		control, cb := fresh(false, "")
		want, _, err := control.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
		if err != nil {
			t.Fatal(err)
		}
		selected, b := fresh(true, "")
		got, outcome, err := selected.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
		if err != nil || outcome != v41RoutedExpertHandled || len(got) != rows {
			t.Fatalf("rows%d: %v %v", rows, outcome, err)
		}
		for r := range got {
			if !v41SameBits(got[r], want[r]) {
				t.Fatalf("rows%d row%d", rows, r)
			}
		}
		if b.pairCalls != 1 || cb.pairCalls != 0 || b.uploads != cb.uploads {
			t.Fatalf("rows%d pair=%d uploads=%d/%d", rows, b.pairCalls, b.uploads, cb.uploads)
		}
		for _, failure := range []string{"empty", "gate-only", "both", "alias"} {
			failed, fb := fresh(true, failure)
			_, _, err := failed.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			if err == nil || fb.pairCalls != 1 || len(fb.ownedTiles) != 0 {
				t.Fatalf("%s rows%d err=%v calls=%d owned=%d", failure, rows, err, fb.pairCalls, len(fb.ownedTiles))
			}
		}
	}
}
