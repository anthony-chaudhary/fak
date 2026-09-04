package microfleeteconomics

import (
	"testing"
)

func BenchmarkMicroFleetEconomics(b *testing.B) {
	rates := Rates{
		ComputeMicroUSDPerJoule:       1,
		MachineMicroUSDPerMillisecond: 1,
		NetworkMicroUSDPerByte:        0,
		StorageMicroUSDPerByteSecond:  0,
	}
	r1 := Receipt{
		Name:         "baseline-small",
		Task:         "bounded-judgment-v1",
		Width:        1000,
		Attempted:    1000,
		Accepted:     10,
		QualityMilli: 900,
		Operations:   Operations{Branches: 1000, CacheHits: 999, Verifications: 1000},
		Costs: Ledger{
			UsefulWork:        PhysicalCost{ComputeJoules: 500000, MachineMillis: 100000},
			Branching:         PhysicalCost{ComputeJoules: 1000},
			CacheConstruction: PhysicalCost{ComputeJoules: 1000},
			CacheHits:         PhysicalCost{ComputeJoules: 999},
			QueueDelay:        PhysicalCost{MachineMillis: 10000},
			Verification:      PhysicalCost{ComputeJoules: 10000},
			FanIn:             PhysicalCost{ComputeJoules: 5000},
		},
	}
	r2 := Receipt{
		Name:         "candidate-100k",
		Task:         "bounded-judgment-v1",
		Width:        100000,
		Attempted:    100000,
		Accepted:     1000,
		QualityMilli: 900,
		Operations:   Operations{Branches: 100000, CacheHits: 99999, Verifications: 100000},
		Costs: Ledger{
			UsefulWork:        PhysicalCost{ComputeJoules: 5000000, MachineMillis: 1000000},
			Branching:         PhysicalCost{ComputeJoules: 100000},
			CacheConstruction: PhysicalCost{ComputeJoules: 1000},
			CacheHits:         PhysicalCost{ComputeJoules: 99999},
			QueueDelay:        PhysicalCost{MachineMillis: 100000},
			Verification:      PhysicalCost{ComputeJoules: 1000000},
			FanIn:             PhysicalCost{ComputeJoules: 500000},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Evaluate(r1, rates); err != nil {
			b.Fatal(err)
		}
		if _, err := Evaluate(r2, rates); err != nil {
			b.Fatal(err)
		}
		cmp, err := Compare(r1, r2, rates)
		if err != nil {
			b.Fatal(err)
		}
		if cmp == 0 {
			b.Fatal("unexpected tie")
		}
	}
}

func TestBenchmarkSmoke(t *testing.T) {
	rates := Rates{
		ComputeMicroUSDPerJoule:       1,
		MachineMicroUSDPerMillisecond: 1,
	}
	r := validReceipt()
	res, err := Evaluate(r, rates)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}
	if res.Accepted == 0 {
		t.Fatal("expected non-zero accepted work")
	}
}
