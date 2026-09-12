package model

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestV41Budget pins the published, measured checkpoint byte counts and the
// per-node admission budget derived from them. It is I/O-free apart from the
// small pinned config fixture; it never reads the real checkpoint.
func TestV41Budget(t *testing.T) {
	t.Run("TestV41BudgetConstantsMatchCitedMeasurement", func(t *testing.T) {
		cases := []struct {
			name string
			got  int64
			want int64
		}{
			{"checkpoint 48 shards", DeepSeekV41CheckpointBytes, 510296708312},
			{"engram tables", DeepSeekV41EngramBytes, 202758032400},
		}
		for _, tc := range cases {
			if tc.got != tc.want {
				t.Fatalf("%s: got=%d want=%d", tc.name, tc.got, tc.want)
			}
		}
	})

	t.Run("TestV41BudgetBackboneIsCheckpointMinusEngram", func(t *testing.T) {
		b := DeepSeekV41BudgetForNodes(1, 1.0)
		wantBackbone := int64(510296708312 - 202758032400)
		if b.BackboneBytes != wantBackbone {
			t.Fatalf("backbone=%d want=%d", b.BackboneBytes, wantBackbone)
		}
		if b.BackboneBytes <= 0 {
			t.Fatalf("backbone must be positive, got %d", b.BackboneBytes)
		}
	})

	t.Run("TestV41BudgetAccountsEngramSeparately", func(t *testing.T) {
		for _, nodes := range []int{1, 3} {
			for _, frac := range []float64{1.0, DeepSeekV41DefaultResidentFraction} {
				b := DeepSeekV41BudgetForNodes(nodes, frac)
				if !b.Fits {
					t.Fatalf("nodes=%d fraction=%g not admitted: %s", nodes, frac, b.Describe())
				}
				if b.EngramResidentBytes+b.BackboneResidentBytes != b.ResidentBytes {
					t.Fatalf("nodes=%d fraction=%g shares %d+%d != resident %d",
						nodes, frac, b.EngramResidentBytes, b.BackboneResidentBytes, b.ResidentBytes)
				}
				if b.EngramResidentBytes <= 0 || b.EngramResidentBytes >= b.ResidentBytes {
					t.Fatalf("nodes=%d fraction=%g engram share %d not in (0,resident)",
						nodes, frac, b.EngramResidentBytes)
				}
				if b.ResidentBytes <= 0 {
					t.Fatalf("nodes=%d fraction=%g resident not positive", nodes, frac)
				}
			}
		}
	})

	t.Run("TestV41BudgetPerNodeDecreasesWithNodes", func(t *testing.T) {
		for _, frac := range []float64{1.0, DeepSeekV41DefaultResidentFraction} {
			one := DeepSeekV41BudgetForNodes(1, frac)
			three := DeepSeekV41BudgetForNodes(3, frac)
			if one.ResidentBytes < three.ResidentBytes {
				t.Fatalf("fraction=%g one-node %d < three-node %d", frac, one.ResidentBytes, three.ResidentBytes)
			}
			if three.ResidentBytes <= 0 {
				t.Fatalf("fraction=%g three-node resident not positive", frac)
			}
		}
	})

	t.Run("TestV41BudgetFullResidencySplitsExact", func(t *testing.T) {
		one := DeepSeekV41BudgetForNodes(1, 1.0)
		if one.ResidentBytes != DeepSeekV41CheckpointBytes {
			t.Fatalf("1 node full resident=%d want=%d", one.ResidentBytes, DeepSeekV41CheckpointBytes)
		}
		if one.EngramResidentBytes != DeepSeekV41EngramBytes {
			t.Fatalf("1 node engram=%d want=%d", one.EngramResidentBytes, DeepSeekV41EngramBytes)
		}
		if one.BackboneResidentBytes != one.BackboneBytes {
			t.Fatalf("1 node backbone share=%d want=%d", one.BackboneResidentBytes, one.BackboneBytes)
		}
		three := DeepSeekV41BudgetForNodes(3, 1.0)
		wantResident := int64(math.Ceil(float64(DeepSeekV41CheckpointBytes) / 3))
		if three.ResidentBytes != wantResident {
			t.Fatalf("3 node full resident=%d want=%d", three.ResidentBytes, wantResident)
		}
	})

	t.Run("TestV41BudgetStreamedFractionScalesDown", func(t *testing.T) {
		full := DeepSeekV41BudgetForNodes(1, 1.0)
		streamed := DeepSeekV41BudgetForNodes(1, DeepSeekV41DefaultResidentFraction)
		if streamed.ResidentBytes >= full.ResidentBytes {
			t.Fatalf("streamed resident %d not below full %d", streamed.ResidentBytes, full.ResidentBytes)
		}
		if streamed.ResidentBytes != full.ResidentBytes/4 {
			t.Fatalf("streamed resident=%d want fourth of %d", streamed.ResidentBytes, full.ResidentBytes)
		}
	})

	t.Run("TestV41BudgetAdmissionRefusesOverBudgetBeforeAllocation", func(t *testing.T) {
		budget := DeepSeekV41BudgetForNodes(3, DeepSeekV41DefaultResidentFraction)
		if !budget.Fits {
			t.Fatalf("budget not admitted: %s", budget.Describe())
		}
		if err := AdmitDeepSeekV41Checkpoint(budget.ResidentBytes, budget); err != nil {
			t.Fatalf("in-budget checkpoint refused: %v", err)
		}
		over := budget.ResidentBytes + 1
		err := AdmitDeepSeekV41Checkpoint(over, budget)
		if !errors.Is(err, ErrV41BudgetExceeded) {
			t.Fatalf("over-budget error=%v want ErrV41BudgetExceeded", err)
		}
		if err := AdmitDeepSeekV41Checkpoint(over, DeepSeekV41NodeBudget{}); !errors.Is(err, ErrV41BudgetExceeded) {
			t.Fatalf("zero budget error=%v want ErrV41BudgetExceeded", err)
		}
	})

	t.Run("TestV41BudgetInvalidInputsDoNotPanic", func(t *testing.T) {
		cases := []struct {
			name  string
			nodes int
			frac  float64
		}{
			{"zero nodes", 0, 1.0},
			{"negative nodes", -3, 1.0},
			{"zero fraction", 3, 0},
			{"negative fraction", 3, -0.5},
			{"fraction above one", 3, 1.5},
			{"nan fraction", 3, math.NaN()},
		}
		for _, tc := range cases {
			b := DeepSeekV41BudgetForNodes(tc.nodes, tc.frac)
			if b.Fits {
				t.Fatalf("%s: unexpectedly fits: %s", tc.name, b.Describe())
			}
			if b.ResidentBytes != 0 || b.EngramResidentBytes != 0 || b.BackboneResidentBytes != 0 {
				t.Fatalf("%s: nonzero bytes on invalid input: %s", tc.name, b.Describe())
			}
			if err := AdmitDeepSeekV41Checkpoint(1, b); !errors.Is(err, ErrV41BudgetExceeded) {
				t.Fatalf("%s: admission error=%v want ErrV41BudgetExceeded", tc.name, err)
			}
		}
	})

	t.Run("TestV41BudgetGeometryConsistentWithPinnedFixture", func(t *testing.T) {
		raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v41_flash_config.json"))
		if err != nil {
			t.Fatal(err)
		}
		var cfg Config
		if err := json.Unmarshal(raw, &cfg); err != nil {
			t.Fatalf("parse pinned V4.1 config: %v", err)
		}
		if !cfg.IsDeepSeekV41() || cfg.DeepSeekV41 == nil {
			t.Fatalf("pinned fixture is not V4.1: %#v", cfg.ModelType)
		}
		m := cfg.DeepSeekV41
		if len(m.EngramLayerIDs) != 2 {
			t.Fatalf("engram layer IDs=%v want 2 entries", m.EngramLayerIDs)
		}
		if len(m.EngramNumEmbeddings) != len(m.EngramLayerIDs) {
			t.Fatalf("engram rows=%v do not match layer IDs=%v", m.EngramNumEmbeddings, m.EngramLayerIDs)
		}
		if m.EngramHeadDim <= 0 || m.EngramVocabSize <= 0 {
			t.Fatalf("engram geometry not positive: head_dim=%d vocab=%d", m.EngramHeadDim, m.EngramVocabSize)
		}
	})
}
