package newleaf

import (
	"testing"
)

func BenchmarkDocGo(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DocGo("myfeature", "mechanism", 3, "a high-performance feature")
	}
}

func BenchmarkImplGo(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ImplGo("myfeature", true)
	}
}

func BenchmarkTestGo(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TestGo("myfeature")
	}
}

func BenchmarkAddLeafLane(b *testing.B) {
	fixture := `[lanes]
concurrent = [
  "foo",
  # new-leaf:lane
]
autopick = [
  "foo",
  # new-leaf:lane
]
[lanes.trees]
foo = ["internal/foo/**"]
# new-leaf:tree
cmd = ["cmd/**"]
`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := AddLeafLane(fixture, "myleaf")
		if err != nil || len(out) == 0 {
			b.Fatal(err)
		}
	}
}
