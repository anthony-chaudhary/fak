package deliverystages

import (
	"testing"
)

// BenchmarkDeliveryStages measures the complete delivery-stages operational path:
// default registry instantiation, validation of stages and bottlenecks, stage lookup,
// downstream invalidation calculation, and crosswalk resolution.
func BenchmarkDeliveryStages(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Default()
		if err := r.Validate(); err != nil {
			b.Fatalf("validate: %v", err)
		}
		if _, ok := r.Stage(StageID("recording")); !ok {
			b.Fatal("missing recording stage")
		}
		if _, err := r.InvalidatedAfter(StageID("build")); err != nil {
			b.Fatalf("invalidated: %v", err)
		}
		if _, ok := r.ResolveLocal("go build"); !ok {
			b.Fatal("missing crosswalk")
		}
	}
}

// BenchmarkDefaultRegistry measures instantiation of the default delivery registry.
func BenchmarkDefaultRegistry(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Default()
	}
}

// BenchmarkRegistryValidate measures validation of the delivery-stages dependency graph.
func BenchmarkRegistryValidate(b *testing.B) {
	r := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := r.Validate(); err != nil {
			b.Fatalf("validate: %v", err)
		}
	}
}

// BenchmarkRegistryStageLookup measures linear lookup performance across registered stages.
func BenchmarkRegistryStageLookup(b *testing.B) {
	r := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := r.Stage(StageID("closure")); !ok {
			b.Fatal("missing closure stage")
		}
	}
}

// BenchmarkRegistryInvalidatedAfter measures calculation of downstream invalidated stages.
func BenchmarkRegistryInvalidatedAfter(b *testing.B) {
	r := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.InvalidatedAfter(StageID("authoring")); err != nil {
			b.Fatalf("invalidated: %v", err)
		}
	}
}

// BenchmarkRegistryResolveLocal measures crosswalk lookup from local tool names.
func BenchmarkRegistryResolveLocal(b *testing.B) {
	r := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := r.ResolveLocal("dos arbitrate"); !ok {
			b.Fatal("missing dos arbitrate crosswalk")
		}
	}
}
