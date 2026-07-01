package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
)

func TestToolEnginesSelfDeclareClassical(t *testing.T) {
	cases := []struct {
		id  string
		eng abi.EngineDriver
	}{
		{"classical-localtools", localEngine{}},
		{"classical-fakread", readEngine{}},
	}
	for _, tc := range cases {
		abi.RegisterEngine(tc.id, tc.eng)
		if got := fusedturn.ClassifyResolved(&abi.ToolCall{Engine: tc.id}); got != fusedturn.ClassClassical {
			t.Fatalf("%s ClassifyResolved = %v, want ClassClassical", tc.id, got)
		}
	}
}
