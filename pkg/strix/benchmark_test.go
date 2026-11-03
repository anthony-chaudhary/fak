package strix_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/strix"
)

var (
	benchProfileSink strix.DeviceProfile
	benchNameSink    string
	benchSupportSink bool
	benchAddrSink    uintptr
	benchUnitsSink   int
)

// BenchmarkDefaultStrixHaloProfile measures canonical profile construction.
// Catalog: BENCHMARK-AUTHORITY.md (strix-halo row). SLO threshold: informational micro-bench, no SLO. Regression gate: go test ./pkg/strix/ -bench=. -run='^$'
func BenchmarkDefaultStrixHaloProfile(b *testing.B) {
	var sink strix.DeviceProfile
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = strix.DefaultStrixHaloProfile()
	}
	benchProfileSink = sink
}

// BenchmarkEngineDispatch measures AccelerationEngine interface dispatch on the Strix Halo profile.
// Catalog: BENCHMARK-AUTHORITY.md (strix-halo row). SLO threshold: informational micro-bench, no SLO. Regression gate: go test ./pkg/strix/ -bench=. -run='^$'
func BenchmarkEngineDispatch(b *testing.B) {
	var eng strix.AccelerationEngine = &mockAccelerationEngine{name: "strix-halo", supported: true}
	var name string
	var supported bool
	var addr uintptr
	var units int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		name = eng.Name()
		supported = eng.IsSupported()
		units = eng.WavefrontSize() + eng.ComputeUnits()
		a, err := eng.AllocUSWCGTT(1024)
		if err != nil {
			b.Fatal("unexpected allocation error:", err)
		}
		addr = a
	}
	benchNameSink = name
	benchSupportSink = supported
	benchAddrSink = addr
	benchUnitsSink = units
}
