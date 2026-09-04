package ultracodenegcontrol

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkUltraCodeNegControl(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("testdata", "campaign.json"))
	if err != nil {
		b.Fatal(err)
	}
	var campaign Campaign
	if err := json.Unmarshal(data, &campaign); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		report, err := Evaluate(campaign)
		if err != nil {
			b.Fatal(err)
		}
		if len(report.Results) == 0 {
			b.Fatal("unexpected empty results")
		}
	}
}
