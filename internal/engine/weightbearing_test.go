package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
)

func TestServingEnginesSelfDeclareWeightBearing(t *testing.T) {
	abi.ResetForTest()
	cases := []struct {
		id  string
		eng abi.EngineDriver
	}{
		{"weight-vllm", NewVLLMEngine(VLLMConfig{})},
		{"weight-sglang", NewSGLangEngine(SGLangConfig{})},
		{"weight-llmd", NewLLMDEngine(LLMDConfig{})},
		{"weight-dynamo", NewDynamoEngine(DynamoConfig{})},
		{"weight-ondevice", NewOnDeviceEngine("on-device:test", nil)},
		{"weight-adapter", NewAdapterEngine("adapter:test", nil)},
	}
	for _, tc := range cases {
		abi.RegisterEngine(tc.id, tc.eng)
		if got := fusedturn.ClassifyResolved(&abi.ToolCall{Engine: tc.id}); got != fusedturn.ClassWeight {
			t.Fatalf("%s ClassifyResolved = %v, want ClassWeight", tc.id, got)
		}
	}
}
