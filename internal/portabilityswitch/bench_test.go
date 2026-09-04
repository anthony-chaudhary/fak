package portabilityswitch

import (
	"strconv"
	"testing"
)

func BenchmarkPortabilitySwitch(b *testing.B) {
	c, _, _, _ := fixture(HotSwitch)
	req := Request{
		Transaction: "bench-tx",
		Root:        "parent",
		Context:     "context-bench",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		req.Transaction = "bench-tx-" + strconv.Itoa(i)
		if _, err := c.Switch(req); err != nil {
			b.Fatalf("Switch failed at iteration %d: %v", i, err)
		}
	}
}
