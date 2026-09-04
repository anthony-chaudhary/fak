package popularizationtickets

import (
	"testing"
)

var (
	benchSinkString  string
	benchSinkBytes   []byte
	benchSinkTickets []Ticket
)

func BenchmarkPopularizationTickets(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = List(tickets)
		benchSinkString = LanesTSV(tickets)
		for _, t := range tickets {
			benchSinkString = RenderBody(t, "epic #1")
		}
	}
}

func BenchmarkRenderBody(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := tickets[i%len(tickets)]
		benchSinkString = RenderBody(t, "epic #1")
	}
}

func BenchmarkLoad(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tickets, err := Load()
		if err != nil {
			b.Fatalf("Load failed: %v", err)
		}
		benchSinkTickets = tickets
	}
}

func BenchmarkList(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = List(tickets)
	}
}

func BenchmarkLanesTSV(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = LanesTSV(tickets)
	}
}

func BenchmarkJSON(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := JSON(tickets)
		if err != nil {
			b.Fatalf("JSON failed: %v", err)
		}
		benchSinkBytes = data
	}
}

func BenchmarkEmitFiles(b *testing.B) {
	tickets, err := Load()
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dir := b.TempDir()
		if err := EmitFiles(dir, "epic #1", tickets); err != nil {
			b.Fatalf("EmitFiles failed: %v", err)
		}
	}
}
