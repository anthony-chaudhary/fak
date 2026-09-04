package deadlineadmit

import "testing"

func BenchmarkDeadlineAdmit(b *testing.B) {
	items := []Item{
		{ID: 1, Deadline: 100, PredictedCost: 20, Degradable: true},
		{ID: 2, Deadline: 50, PredictedCost: 10, Degradable: false},
		{ID: 3, Deadline: 70, PredictedCost: 40, Degradable: true},
		{ID: 4, Deadline: 30, PredictedCost: 50, Degradable: true},
		{ID: 5, Deadline: 120, PredictedCost: 15, Degradable: false},
		{ID: 6, Deadline: 80, PredictedCost: 90, Degradable: true},
	}
	now := 20
	dropThreshold := 5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Admit(items, now, dropThreshold)
	}
}

func BenchmarkDeadlineOrder(b *testing.B) {
	items := []Item{
		{ID: 1, Deadline: 100, PredictedCost: 20, Degradable: true},
		{ID: 2, Deadline: 50, PredictedCost: 10, Degradable: false},
		{ID: 3, Deadline: 70, PredictedCost: 40, Degradable: true},
		{ID: 4, Deadline: 30, PredictedCost: 50, Degradable: true},
		{ID: 5, Deadline: 120, PredictedCost: 15, Degradable: false},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Order(items)
	}
}

func BenchmarkDeadlineShed(b *testing.B) {
	items := []Item{
		{ID: 1, Deadline: 100, PredictedCost: 20, Degradable: true},
		{ID: 2, Deadline: 50, PredictedCost: 10, Degradable: false},
		{ID: 3, Deadline: 70, PredictedCost: 40, Degradable: true},
		{ID: 4, Deadline: 30, PredictedCost: 50, Degradable: true},
		{ID: 5, Deadline: 120, PredictedCost: 15, Degradable: false},
	}
	now := 20
	dropThreshold := 5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Shed(items, now, dropThreshold)
	}
}
