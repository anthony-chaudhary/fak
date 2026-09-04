package wiplease

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// BenchmarkWIPLease measures throughput and allocation overhead of the
// pure WIP lease projection over a representative mixed attribution set.
func BenchmarkWIPLease(b *testing.B) {
	attrs := []wipattr.Attribution{
		{File: "internal/a/a.go", State: wipattr.AttrOwned, Owner: "sess-1"},
		{File: "internal/a/b.go", State: wipattr.AttrOwned, Owner: "sess-1"},
		{File: "internal/b/c.go", State: wipattr.AttrOwned, Owner: "sess-2"},
		{File: "internal/b/d.go", State: wipattr.AttrShared, Owners: []string{"sess-1", "sess-2"}},
		{File: "internal/c/e.go", State: wipattr.AttrOrphan},
		{File: "internal/d/f.go", State: wipattr.AttrOwned, Owner: "sess-dead"},
	}
	live := map[string]bool{
		"sess-1": true,
		"sess-2": true,
	}
	opts := Options{
		Declared: map[string]bool{"sess-1": true},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Project(attrs, live, opts)
	}
}
