package microfleeteconomics

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

type fixtureFile struct {
	Rates Rates         `json:"rates"`
	Cases []fixtureCase `json:"cases"`
}

type fixtureCase struct {
	Name     string    `json:"name"`
	Winner   string    `json:"winner"`
	Receipts []Receipt `json:"receipts"`
}

func TestReversalFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/fixtures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures fixtureFile
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures.Cases) != 2 {
		t.Fatalf("cases = %d, want 2", len(fixtures.Cases))
	}
	for _, tc := range fixtures.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if len(tc.Receipts) < 2 {
				t.Fatal("fixture needs at least two receipts")
			}
			seenWidth := map[uint64]bool{}
			winner := tc.Receipts[0]
			for _, receipt := range tc.Receipts {
				seenWidth[receipt.Width] = true
				if _, err := Evaluate(receipt, fixtures.Rates); err != nil {
					t.Fatalf("Evaluate(%s): %v", receipt.Name, err)
				}
				cmp, err := Compare(receipt, winner, fixtures.Rates)
				if err != nil {
					t.Fatal(err)
				}
				if cmp < 0 {
					winner = receipt
				}
			}
			if winner.Name != tc.Winner {
				t.Fatalf("winner = %q, want %q", winner.Name, tc.Winner)
			}
			if tc.Name == "shared-cache-wins" && (!seenWidth[1] || !seenWidth[1000] || !seenWidth[100000]) {
				t.Fatalf("shared-cache fixture widths = %v, want 1, 1000, and 100000", seenWidth)
			}
		})
	}
}

func TestAcceptedWorkIsTheDenominator(t *testing.T) {
	r := validReceipt()
	r.Attempted = 100
	r.Operations.Branches = 100
	r.Accepted = 4
	r.Costs.UsefulWork.DirectMicroUSD = 1000
	got, err := Evaluate(r, Rates{})
	if err != nil {
		t.Fatal(err)
	}
	if got.CostPerAcceptedNumerator != 1000 || got.CostPerAcceptedDenominator != 4 {
		t.Fatalf("cost ratio = %d/%d, want 1000/4", got.CostPerAcceptedNumerator, got.CostPerAcceptedDenominator)
	}
}

func TestAdversarialReceipts(t *testing.T) {
	t.Run("zero accepted", func(t *testing.T) {
		r := validReceipt()
		r.Accepted = 0
		_, err := Evaluate(r, Rates{})
		if !errors.Is(err, ErrNoAcceptedWork) {
			t.Fatalf("err = %v, want ErrNoAcceptedWork", err)
		}
	})
	t.Run("accepted exceeds attempted", func(t *testing.T) {
		r := validReceipt()
		r.Accepted = 2
		if _, err := Evaluate(r, Rates{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("missing branch accounting", func(t *testing.T) {
		r := validReceipt()
		r.Operations.Branches = 0
		if _, err := Evaluate(r, Rates{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("cost overflow", func(t *testing.T) {
		r := validReceipt()
		r.Costs.UsefulWork.ComputeJoules = math.MaxUint64
		_, err := Evaluate(r, Rates{ComputeMicroUSDPerJoule: 2})
		if !errors.Is(err, ErrOverflow) {
			t.Fatalf("err = %v, want ErrOverflow", err)
		}
	})
	t.Run("unmatched quality", func(t *testing.T) {
		a, b := validReceipt(), validReceipt()
		b.QualityMilli++
		if _, err := Compare(a, b, Rates{}); err == nil {
			t.Fatal("expected error")
		}
	})
}

func TestAllOverheadCategoriesAreIncluded(t *testing.T) {
	r := validReceipt()
	one := PhysicalCost{DirectMicroUSD: 1}
	r.Costs = Ledger{UsefulWork: one, Branching: one, CacheConstruction: one, CacheHits: one, QueueDelay: one, Verification: one, FanIn: one, Cancellation: one, Failures: one, Retries: one, Recovery: one, Stragglers: one}
	got, err := Evaluate(r, Rates{})
	if err != nil {
		t.Fatal(err)
	}
	if got.TotalMicroUSD != 12 {
		t.Fatalf("total = %d, want 12", got.TotalMicroUSD)
	}
}

func validReceipt() Receipt {
	return Receipt{Name: "valid", Task: "task", Width: 1, Attempted: 1, Accepted: 1, QualityMilli: 1, Operations: Operations{Branches: 1}}
}
