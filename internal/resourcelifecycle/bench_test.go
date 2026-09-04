package resourcelifecycle

import "testing"

// BenchmarkResourceLifecycle exercises resource claim resolution, observations, and teardown in a loop.
func BenchmarkResourceLifecycle(b *testing.B) {
	m := New()
	c := Claim{
		Kind:          "attention_kv",
		Owner:         "session-bench",
		Isolation:     "session-bench",
		Lifetime:      "step",
		Compatibility: "kv/v1",
		Mutable:       true,
		Shareable:     false,
		Bytes:         4096,
		Geometry:      Geometry{Shape: []int{16, 64}, Alignment: 64},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		alloc, err := m.Resolve(c, "host", "device")
		if err != nil {
			b.Fatal(err)
		}
		if err := m.Observe(Observation{
			Ref:           alloc.Ref,
			Action:        "transfer",
			TransferBytes: c.Bytes,
			To:            "device",
			Reason:        "benchmark_cycle",
		}); err != nil {
			b.Fatal(err)
		}
		if _, ok := m.Get(alloc.Ref); !ok {
			b.Fatalf("missing ref %d", alloc.Ref)
		}
		if err := m.Observe(Observation{
			Ref:    alloc.Ref,
			Action: "release",
			Reason: "benchmark_cycle",
		}); err != nil {
			b.Fatal(err)
		}
	}
}

func TestBenchmarkResourceLifecycleSmoke(t *testing.T) {
	m := New()
	c := Claim{
		Kind:          "model_weights",
		Owner:         "session-smoke",
		Isolation:     "session-smoke",
		Lifetime:      "session",
		Compatibility: "weights/v1",
		Mutable:       false,
		Shareable:     true,
		Bytes:         2048,
	}
	alloc, err := m.Resolve(c, "host", "device")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.Get(alloc.Ref); !ok {
		t.Fatalf("expected ref %d", alloc.Ref)
	}
	m.Teardown("session-smoke")
}
