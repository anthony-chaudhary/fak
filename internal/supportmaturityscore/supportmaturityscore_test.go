package supportmaturityscore

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
)

func TestBuildFoldsCoverageMatrix(t *testing.T) {
	payload := Build()
	if payload.Schema != Schema {
		t.Fatalf("schema = %q, want %q", payload.Schema, Schema)
	}
	if _, ok := payload.Corpus[DebtKey]; !ok {
		t.Fatalf("corpus missing %q", DebtKey)
	}
	if payload.Corpus["cells"] != len(covmatrix.Grid()) {
		t.Fatalf("cells corpus = %v, want %d", payload.Corpus["cells"], len(covmatrix.Grid()))
	}
	if payload.Corpus["support_coverage_pct"] != payload.Corpus["score"] {
		t.Fatalf("coverage pct = %v, score = %v; support coverage should drive the grade",
			payload.Corpus["support_coverage_pct"], payload.Corpus["score"])
	}
	if _, err := json.Marshal(payload); err != nil {
		t.Fatalf("payload is not JSON serializable: %v", err)
	}
}

func TestSupportMaturityDebtCountsCellsBelowSupported(t *testing.T) {
	payload := Build()
	cells := covmatrix.Grid()
	wantDebt := 0
	wantSupported := 0
	for _, c := range cells {
		if c.Support == covmatrix.Supported {
			wantSupported++
		} else {
			wantDebt++
		}
	}
	if payload.Corpus[DebtKey] != wantDebt {
		t.Fatalf("%s = %v, want %d", DebtKey, payload.Corpus[DebtKey], wantDebt)
	}
	if payload.Corpus["supported"] != wantSupported {
		t.Fatalf("supported = %v, want %d", payload.Corpus["supported"], wantSupported)
	}
	if len(payload.KPIs) != 1 {
		t.Fatalf("len(kpis) = %d, want 1", len(payload.KPIs))
	}
	if len(payload.KPIs[0].Defects) != wantDebt {
		t.Fatalf("defects = %d, want %d", len(payload.KPIs[0].Defects), wantDebt)
	}
}
