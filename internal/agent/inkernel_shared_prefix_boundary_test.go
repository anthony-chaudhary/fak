package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s
func TestInKernelPrefillCheckpointsOrdersAndDedupes(t *testing.T) {
	for _, tc := range []struct {
		name                                  string
		matched, cacheable, boundary, prompts int
		want                                  []int
	}{
		{"cold no boundary falls back to grid", 0, 0, 0, 150, []int{128}},
		{"boundary before grid", 0, 0, 40, 150, []int{40, 128}},
		{"boundary after divergence", 0, 30, 40, 150, []int{30, 40}},
		{"boundary equals divergence", 0, 40, 40, 150, []int{40}},
		{"boundary already restored", 40, 40, 40, 150, []int{128}},
		{"boundary at prompt end left to full admission", 0, 0, 150, 150, []int{128}},
		{"short prompt boundary only", 0, 0, 20, 50, []int{20}},
	} {
		if got := inKernelPrefillCheckpoints(tc.matched, tc.cacheable, tc.boundary, tc.prompts); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: checkpoints(%d,%d,%d,%d)=%v, want %v", tc.name, tc.matched, tc.cacheable, tc.boundary, tc.prompts, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1s
func TestInKernelSharedPrefixTextEndsAtSystemBlock(t *testing.T) {
	msgs := []Message{{Role: RoleSystem, Content: "be terse"}, {Role: RoleUser, Content: "hi"}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file"}}}
	for name, rendered := range map[string]string{
		"chatml": renderChatMLTools(msgs, tools),
		"ornith": renderOrnithQwen35ChatMLTools(msgs, tools, false),
	} {
		shared := inKernelSharedPrefixText(rendered)
		if !strings.HasPrefix(shared, inKernelSharedBlockOpen) || !strings.HasSuffix(shared, inKernelSharedBlockEnd) ||
			!strings.Contains(shared, "be terse") || !strings.Contains(shared, "read a file") ||
			!strings.HasPrefix(rendered[len(shared):], "<|im_start|>user\n") {
			t.Errorf("%s: shared block %q does not end exactly at the system/tools block", name, shared)
		}
	}
	if got := inKernelSharedPrefixText(renderChatMLTools([]Message{{Role: RoleUser, Content: "hi"}}, nil)); got != "" {
		t.Errorf("no system block: shared=%q, want empty", got)
	}
}

// Subagent fan-out witness: a parent (system S + user A) and its FIRST child
// (system S + user B) on a recurrent hybrid. Recurrent state cannot be truncated,
// so before the shared-boundary snapshot the first child restored 0 tokens; it must
// now restore at least the rendered system/tools block, and its output must equal a
// cold reference.
// fak-test:runtime fast est=4s lane=default
func TestInKernelFirstFanOutChildRestoresSharedSystemBoundary(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	system := "You are a subagent of a coding harness. Follow the plan, cite files, and keep answers short. " +
		"Never invent tool output; report blockers with the exact command that failed."
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file from the workspace"}},
		{Type: "function", Function: ToolDefFunction{Name: "Grep", Description: "search file contents"}},
	}
	parent := []Message{{Role: RoleSystem, Content: system}, {Role: RoleUser, Content: "Summarize the parent task in one line."}}
	child := []Message{{Role: RoleSystem, Content: system}, {Role: RoleUser, Content: "Child B: list the files you would read first."}}
	opts := []SampleOpt{WithMaxTokens(3)}

	for _, tc := range []struct {
		name      string
		modelType string
		device    bool
	}{
		{"chatml/device", "", true},
		{"ornith-qwen35/device", "qwen3_5", true},
		{"chatml/host", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tinyHybridCfg()
			cfg.VocabSize = 320 // covers the probe tokenizer's byte vocab + ChatML specials
			cfg.ModelType = tc.modelType
			newPlanner := func(reuse bool) *InKernelPlanner {
				var backend compute.Backend
				if tc.device {
					backend = &countingBackend{Backend: compute.Default(), deviceMemory: true}
				}
				p := NewInKernelPlanner(model.NewSynthetic(cfg), loadProbeTok(t), "shared-boundary-hybrid", false, backend, false)
				p.quant = false
				if !reuse {
					p.tree, p.scopedTree = nil, nil
				}
				return p
			}
			p := newPlanner(true)
			if p.tree == nil {
				t.Fatal("precondition: prefix reuse tree disabled")
			}
			preparedTools, err := p.preparePrompt(context.Background(), child, tools, applySampleOpts(opts...), opts...)
			if err != nil {
				t.Fatal(err)
			}
			boundary := preparedTools.sharedBoundary
			if boundary <= 0 || boundary >= len(preparedTools.ids) {
				t.Fatalf("shared boundary=%d of %d prompt tokens, want inside the prompt", boundary, len(preparedTools.ids))
			}

			parentComp, err := p.Complete(context.Background(), parent, tools, opts...)
			if err != nil {
				t.Fatalf("parent: %v", err)
			}
			if got := parentComp.Usage.PromptTokensDetails.CachedTokens; got != 0 {
				t.Fatalf("cold parent restored %d tokens, want 0", got)
			}
			childComp, err := p.Complete(context.Background(), child, tools, opts...)
			if err != nil {
				t.Fatalf("first child: %v", err)
			}
			restored := childComp.Usage.PromptTokensDetails.CachedTokens
			if restored < boundary {
				t.Fatalf("first child restored %d tokens, want >= shared system/tools boundary %d", restored, boundary)
			}
			coldComp, err := newPlanner(false).Complete(context.Background(), child, tools, opts...)
			if err != nil {
				t.Fatalf("cold child reference: %v", err)
			}
			if childComp.Message.Content != coldComp.Message.Content {
				t.Fatalf("warm child content %q != cold reference %q", childComp.Message.Content, coldComp.Message.Content)
			}
			t.Logf("SW-VERIFIED first child restored %d/%d prompt tokens (boundary %d); no hardware-performance claim",
				restored, childComp.Usage.PromptTokens, boundary)
		})
	}
}

// The speculative/MTP generation path admits the same shared-boundary checkpoint, so
// the first fan-out child restores the parent's shared prefix there too. The prompts
// are shorter than one 64-token grid block, so the historical grid checkpoint alone
// left the first child at 0 restored tokens.
// fak-test:runtime fast est=2s lane=default
func TestInKernelSpeculativeFirstFanOutChildRestoresSharedBoundary(t *testing.T) {
	cfg := speculativeHybridConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	backend := &prefixReuseQwenBackend{Backend: compute.Default()}
	shared := synthIDs(cfg.VocabSize, 40, 77001)
	parent := append(append([]int(nil), shared...), synthIDs(cfg.VocabSize, 20, 77002)...)
	child := append(append([]int(nil), shared...), synthIDs(cfg.VocabSize, 20, 77003)...)
	ctx := withInKernelSharedPrefixBoundary(context.Background(), len(shared))
	run := func(p *InKernelPlanner, ids []int) ([]int, inKernelGenerateResult) {
		t.Helper()
		var out []int
		res, err := p.generateReusedRecovering(ctx, ids, 4, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			out = append(out, token)
			return false
		})
		if err != nil {
			t.Fatalf("speculative decode len=%d: %v", len(ids), err)
		}
		return out, res
	}

	cold := NewInKernelPlanner(m, nil, "spec-boundary-cold", false, backend, false)
	cold.EnableSpeculativeDecoding(&speculativeRequestProbe{err: errors.New("draft unavailable")}, 4)
	cold.tree, cold.scopedTree = nil, nil
	want, _ := run(cold, child)

	probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	p := NewInKernelPlanner(m, nil, "spec-boundary-reuse", false, backend, false)
	p.EnableSpeculativeDecoding(probe, 4)
	if p.tree == nil {
		t.Fatal("precondition: eligible hybrid backend did not initialize prefix reuse")
	}
	if _, res := run(p, parent); res.matched != 0 {
		t.Fatalf("cold parent matched=%d, want 0", res.matched)
	}
	got, res := run(p, child)
	if probe.calls == 0 {
		t.Fatal("precondition: speculative path was not exercised")
	}
	if res.matched < len(shared) {
		t.Fatalf("first speculative child matched=%d, want >= shared boundary %d", res.matched, len(shared))
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("warm child output=%v, want cold output=%v", got, want)
	}
}
