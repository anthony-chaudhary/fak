package quantlicense

import (
	"encoding/json"
	"os"
	"testing"
)

func BenchmarkQuantLicense(b *testing.B) {
	raw, err := os.ReadFile("testdata/compatible.json")
	if err != nil {
		b.Fatalf("read fixture: %v", err)
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		b.Fatalf("unmarshal fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res := Evaluate(m)
		if res.Outcome != OutcomeAllow {
			b.Fatalf("unexpected benchmark outcome: %s", res.Outcome)
		}
	}
}

func TestBenchmarkQuantLicenseSmoke(t *testing.T) {
	m := readFixture(t, "compatible.json")
	res := Evaluate(m)
	if res.Outcome != OutcomeAllow {
		t.Fatalf("unexpected outcome: %s", res.Outcome)
	}
}
