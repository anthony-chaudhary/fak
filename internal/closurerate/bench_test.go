package closurerate

import "testing"

func BenchmarkClosureRate(b *testing.B) {
	records := []CloseRecord{
		{Issue: 800, Closed: true, HasWitness: true, Note: "commit abc123"},
		{Issue: 801, Closed: true, HasWitness: true, Note: "diff #801"},
		{Issue: 802, Closed: true, HasWitness: true, Note: "test TestFoo"},
		{Issue: 803, Closed: true, HasWitness: false, Note: "closed without witness"},
		{Issue: 804, Closed: true, HasWitness: true, Note: "commit def456"},
		{Issue: 805, Closed: true, HasWitness: true, Note: "diff #805"},
		{Issue: 806, Closed: true, HasWitness: true, Note: "test TestBar"},
		{Issue: 807, Closed: true, HasWitness: false, Note: "claimed without witness"},
		{Issue: 900, Closed: false, HasWitness: false, Note: "still open"},
		{Issue: 901, Closed: false, HasWitness: true, Note: "open with witness ignored"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Fold(records, 4.0)
	}
}
