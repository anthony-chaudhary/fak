package agent

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelRequestPlanAgreesWithServeBootSizer is the #1049 acceptance witness: the
// serve boot path and the per-request planner now feed the SAME single auto-sizer
// (compute.AutoSizeContextPlan), so for one (model, host) input they build a
// byte-identical context plan. Before unification the boot path sized KV from
// MaxPositionEmbeddings and the per-request path from prompt+new, so they could disagree
// (boot refuse at full ctx where a request would have fit). Here the per-request planner's
// real plan must equal the plan the boot path delegates to for the SAME token count.
func TestInKernelRequestPlanAgreesWithServeBootSizer(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.MaxPositionEmbeddings = 4096
	// backend nil → no resident-weights prepend, so the comparison is the shared
	// context portion both call sites own.
	p := &InKernelPlanner{m: model.NewSynthetic(cfg)}

	const promptTokens, maxNew = 100, 28
	tokens := promptTokens + maxNew

	// Per-request: the real plan the in-kernel planner builds for this request.
	got := p.requestMemoryPlan(promptTokens, maxNew)

	// Serve boot: size the SAME (model, tokens) through the SAME single auto-sizer the
	// boot path delegates to (cmd/fak/serve.go appendServeGGUFDevicePlan), with the
	// per-request token count supplied as the explicit context override.
	_, want := compute.AutoSizeContextPlan(cfg.ContextSizeConfig(), nil, compute.FreeUnknown, tokens)

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("per-request and serve-boot plans disagree for the same (model, tokens=%d):\n per-request=%#v\n serve-boot =%#v", tokens, got, want)
	}
}
