package ultracodecrossover

import "testing"

func BenchmarkUltraCodeCrossover(b *testing.B) {
	campaign := validComplexityCampaign()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := EvaluateComplexityCampaign(campaign)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestBenchmarkUltraCodeCrossover(t *testing.T) {
	campaign := validComplexityCampaign()
	if _, err := EvaluateComplexityCampaign(campaign); err != nil {
		t.Fatalf("crossover evaluation failed on valid campaign: %v", err)
	}
}
