package agent

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestNativeEncodePromptPopulatesOrdinaryMemo(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	for _, modelType := range []string{"qwen2", "qwen3_5"} {
		t.Run(modelType, func(t *testing.T) {
			cfg := model.Config{ModelType: modelType, MaxPositionEmbeddings: 32768}
			if modelType == "qwen3_5" {
				cfg.LayerTypes = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
				cfg.Architectures = []string{"Qwen3_5ForCausalLM"}
			}
			if inKernelUsesOrnithQwen35Template(cfg) != (modelType == "qwen3_5") {
				t.Fatal("fixture selected the wrong renderer profile")
			}
			p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, loadProbeTok(t), "memo-fixture", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
			messages := []Message{{Role: RoleSystem, Content: strings.Repeat("Stable instructions. ", 32)}, {Role: RoleUser, Content: "First question"}}
			opts := []SampleOpt{WithMaxTokens(32), WithReasoningEffort(EffortTierNone)}
			for round := 0; round < 2; round++ {
				wantIDs, wantDigest := promptEncodingOracle(t, p, messages, nil, opts...)
				got, err := p.EncodePrompt(context.Background(), messages, nil, opts...)
				if err != nil || !reflect.DeepEqual(got.TokenIDs, wantIDs) || got.RenderedSHA256 != wantDigest {
					t.Fatalf("round %d: EncodePrompt parity failed: error=%v", round, err)
				}
				// Probe only after the public request path. A bypass of the memo in
				// preparePrompt must leave this cold and fail the work assertion.
				rendered := inKernelRenderPrompt(messages, nil, cfg, applySampleOpts(opts...))
				warm, work, err := p.promptEncodeMemo.Encode(p.tok, rendered)
				if err != nil || !reflect.DeepEqual(warm, got.TokenIDs) || work.OrdinaryBytesEncoded != 0 || work.OrdinaryBytesReused < len(messages[0].Content) {
					t.Fatalf("round %d: public path did not populate exact memo: error=%v work=%+v", round, err, work)
				}
				for i := range got.TokenIDs {
					got.TokenIDs[i] = -1
				}
				messages = append(messages, Message{Role: RoleAssistant, Content: "First answer"}, Message{Role: RoleUser, Content: "Next question"})
			}
		})
	}
}
