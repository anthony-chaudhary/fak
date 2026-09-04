package escalation

import (
	"testing"
	"time"
)

// BenchmarkEscalation exercises escalation validation and report folding in a loop.
func BenchmarkEscalation(b *testing.B) {
	p := fixturePacket()
	a := fixtureAck()
	rows := []Row{
		{Packet: &p},
		{Ack: &a},
	}
	asOf := time.Unix(0, a.AckedAtUnixNano).UTC()

	b.Run("ValidatePacket", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := p.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ValidateAck", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := a.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Fold", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			rep := Fold(rows, asOf)
			if len(rep.Acked) != 1 {
				b.Fatal("unexpected folded report")
			}
		}
	})
}
