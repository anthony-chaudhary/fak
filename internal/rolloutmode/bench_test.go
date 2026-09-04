package rolloutmode

import "testing"

var (
	benchSinkMode  Mode
	benchSinkBool  bool
	benchSinkSlice []Mode
)

func BenchmarkParse(b *testing.B) {
	b.Run("Canary", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = Parse("canary", Off)
		}
	})
	b.Run("Shadow", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = Parse("shadow", Off)
		}
	})
	b.Run("Unset", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = Parse("", Off)
		}
	})
	b.Run("Unknown", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = Parse("bogus", Off)
		}
	})
}

func BenchmarkParseIn(b *testing.B) {
	subset := []Mode{Off, Shadow, On}
	b.Run("SubsetHit", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = ParseIn("shadow", subset, Off)
		}
	})
	b.Run("SubsetExcluded", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = ParseIn("canary", subset, Off)
		}
	})
	b.Run("SubsetUnset", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			benchSinkMode, benchSinkBool = ParseIn("", subset, Off)
		}
	})
}

func BenchmarkModes(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkSlice = Modes()
	}
}

func BenchmarkValid(b *testing.B) {
	b.Run("Valid", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = Canary.Valid()
		}
	})
	b.Run("Invalid", func(b *testing.B) {
		m := Mode("bogus")
		for i := 0; i < b.N; i++ {
			benchSinkBool = m.Valid()
		}
	})
}

func BenchmarkComputes(b *testing.B) {
	b.Run("Off", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = Off.Computes()
		}
	})
	b.Run("Shadow", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = Shadow.Computes()
		}
	})
	b.Run("On", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = On.Computes()
		}
	})
}

func BenchmarkApplies(b *testing.B) {
	b.Run("Shadow", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = Shadow.Applies()
		}
	})
	b.Run("Canary", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = Canary.Applies()
		}
	})
	b.Run("On", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = On.Applies()
		}
	})
}

func BenchmarkIn(b *testing.B) {
	allowed := []Mode{Off, Shadow, On}
	b.Run("Hit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = In(Shadow, allowed)
		}
	})
	b.Run("Miss", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkBool = In(Canary, allowed)
		}
	})
}

func BenchmarkParseParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m, ok := Parse("canary", Off)
			if !ok || m != Canary {
				b.Fatal("unexpected parse result")
			}
		}
	})
}

func BenchmarkGatingParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if !Canary.Computes() || !Canary.Applies() {
				b.Fatal("unexpected canary gate behavior")
			}
			if Off.Computes() || Off.Applies() {
				b.Fatal("unexpected off gate behavior")
			}
		}
	})
}
