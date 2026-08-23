package bench

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codelint"
)

func TestGoSyntaxValidationComparisonWitness(t *testing.T) {
	result := codelint.CompareGoSyntaxLocal()
	if len(result.Arms) != 8 { //boundarylint:ignore CHANGE_DETECTOR_TEST eight is the fixed comparison-arm contract documented by this witness.
		t.Fatalf("arms=%d, want 8", len(result.Arms))
	}
	for i, arm := range result.Arms {
		if i < 2 {
			if !arm.Available || !arm.Correct || arm.CorrectFiles != 4 || arm.FalseSyntaxErrors != 0 || arm.MissedSyntaxErrors != 0 || arm.LocationErrors != 0 {
				t.Fatalf("local arm %q: %+v", arm.Name, arm)
			}
			t.Logf("observed local arm: name=%q latency=%s files=%d errors=%d bytes=%d", arm.Name, arm.Latency, arm.Files, arm.ReportedErrors, arm.InputBytes)
			continue
		}
		if arm.Available || arm.Latency != 0 || arm.ReportedErrors != 0 {
			t.Fatalf("external arm must remain an unavailable zero row until independently run: %+v", arm)
		}
	}
	if result.Arms[0].ReportedErrors <= result.Arms[1].ReportedErrors {
		t.Fatalf("native all-errors detail=%d, first-error baseline=%d", result.Arms[0].ReportedErrors, result.Arms[1].ReportedErrors)
	}
}
