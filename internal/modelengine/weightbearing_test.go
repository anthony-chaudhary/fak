package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelEnginesSelfDeclareWeightBearing(t *testing.T) {
	abi.ResetForTest()
	cases := []struct {
		id  string
		eng abi.EngineDriver
	}{
		{"weight-inkernel", New()},
		{"weight-native-sched", NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))},
		{"weight-pipeline", NewPipelineEngine(model.PipelineStage{}, nil)},
	}
	for _, tc := range cases {
		abi.RegisterEngine(tc.id, tc.eng)
		if got := fusedturn.ClassifyResolved(&abi.ToolCall{Engine: tc.id}); got != fusedturn.ClassWeight {
			t.Fatalf("%s ClassifyResolved = %v, want ClassWeight", tc.id, got)
		}
	}
}
