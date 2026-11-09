package agent

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestSharedPrefixMemoKeepsFullRequestAndExactBoundary(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	for _, modelType := range []string{"", "qwen3_5"} {
		t.Run("profile_"+modelType, func(t *testing.T) {
			cfg := model.Config{
				ModelType: modelType, MaxPositionEmbeddings: 32768,
				LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
			}
			p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, loadProbeTok(t), "shared-prefix-memo", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
			if p.tree == nil || !inKernelHostSnapshotReuse(p) {
				t.Fatal("fixture must select the real host hybrid shared-boundary path")
			}
			stable := strings.Repeat("Stable instructions. ", 32)
			messages := []Message{{Role: RoleSystem, Content: stable}, {Role: RoleUser, Content: "Unique registry text"}}
			prepared, err := p.preparePrompt(context.Background(), messages, nil, applySampleOpts(WithReasoningEffort(EffortTierNone)))
			if err != nil {
				t.Fatal(err)
			}
			shared := inKernelSharedPrefixText(prepared.rendered)
			want, err := p.tok.Encode(shared)
			if err != nil || len(want) == 0 || prepared.sharedBoundary != len(want) {
				t.Fatalf("prepared boundary=%d want=%d err=%v", prepared.sharedBoundary, len(want), err)
			}
			boundary, work := p.sharedPrefixBoundaryTokensWithWork(prepared.rendered, prepared.ids)
			if boundary != len(want) || work.OrdinaryBytesEncoded != 0 || work.OrdinaryBytesReused < len(stable) {
				t.Fatalf("boundary retokenized shared content: boundary=%d work=%+v", boundary, work)
			}
			full, fullWork, err := p.promptEncodeMemo.Encode(p.tok, prepared.rendered)
			if err != nil || !reflect.DeepEqual(full, prepared.ids) || fullWork.OrdinaryBytesEncoded != 0 {
				t.Fatalf("boundary measurement replaced full request: err=%v work=%+v", err, fullWork)
			}
			bad := append([]int(nil), prepared.ids...)
			bad[0] = -1
			if got := p.sharedPrefixBoundaryTokens(prepared.rendered, bad); got != 0 {
				t.Fatalf("token-prefix guard accepted mismatching IDs at %d", got)
			}
			p.tree = nil
			if got, work := p.sharedPrefixBoundaryTokensWithWork(prepared.rendered, prepared.ids); got != 0 || work.OrdinaryBytesEncoded != 0 || work.OrdinaryBytesReused != 0 {
				t.Fatalf("ineligible route did work: boundary=%d work=%+v", got, work)
			}
		})
	}
}
