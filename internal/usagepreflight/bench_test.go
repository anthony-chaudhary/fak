package usagepreflight

import (
	"context"
	"testing"
)

func BenchmarkUsagePreflight(b *testing.B) {
	reader := &fakeReader{
		reading: Reading{Remaining: 15, Limit: 100, Known: true},
	}
	selector := &fakeSelector{
		seat: "seat-b",
		ok:   true,
	}
	hook := Hook{
		Config: Config{
			Enabled:        true,
			ReservePercent: 20,
			Policy:         PolicyAuto,
		},
		Reader:   reader,
		Selector: selector,
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, err := hook.Decide(ctx, "seat-a")
		if err != nil {
			b.Fatal(err)
		}
	}
}
