package main

import (
	"flag"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

func TestRunNativeControlsUseExplicitFlagsOverAmbientValues(t *testing.T) {
	t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "4096")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "0")
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	t.Setenv("FAK_PREFIX_PROFILE", "ambient.jsonl")
	t.Setenv("FAK_VULKAN_Q4K_PROFILE", "0")
	t.Setenv("FAK_VULKAN_STAGE_Q4K", "0")

	fs := flag.NewFlagSet("run-native", flag.ContinueOnError)
	flags := registerRunNativeControlFlags(fs)
	if err := fs.Parse([]string{
		"--native-qwen-q4k-prefill-chunk-tokens=8192",
		"--native-qwen35-metal-gdn-sequence",
		"--native-q4k-gateup-slab",
		"--native-prefix-profile=explicit.jsonl",
		"--vulkan-q4k-profile",
		"--vulkan-stage-q4k",
	}); err != nil {
		t.Fatal(err)
	}
	if err := validateNativeQwenQ4KPrefillChunk(*flags.prefillChunk); err != nil {
		t.Fatalf("run rejected the established explicit 8192 contract: %v", err)
	}
	want := nativeControlConfig{
		Planner: agent.InKernelPlannerConfig{
			QwenQ4KPrefillChunkTokens: 8192,
			Qwen35MetalGDNSequence:    true,
			Q4KGateUpOutputSlab:       true,
		},
		PrefixProfile:    "explicit.jsonl",
		VulkanQ4KProfile: true,
		VulkanStageQ4K:   true,
	}
	if got := flags.config(); !reflect.DeepEqual(got, want) {
		t.Fatalf("run native config = %+v, want %+v", got, want)
	}
}

// TestRunDispatchRule pins the argv[0] split that keeps `fak run --trace` (trace mode)
// and `fak run <model>` (chat mode) on separate parsers: a leading '-' is trace mode,
// anything else is chat mode. This is the contract cmdRun depends on.
func TestRunDispatchRule(t *testing.T) {
	cases := []struct {
		argv     []string
		wantChat bool
	}{
		{[]string{"--trace", "x.json"}, false}, // flag-first → trace
		{[]string{"-trace", "x.json"}, false},  // single-dash flag → trace
		{[]string{}, false},                    // bare → trace (errors on --trace required)
		{[]string{"smollm2", "hi"}, true},      // alias → chat
		{[]string{"./model.gguf"}, true},       // path → chat
		{[]string{"hf://o/r/m.gguf"}, true},    // hf uri → chat
	}
	for _, c := range cases {
		isChat := len(c.argv) > 0 && !strings.HasPrefix(c.argv[0], "-")
		if isChat != c.wantChat {
			t.Errorf("argv=%v: dispatch isChat=%v, want %v", c.argv, isChat, c.wantChat)
		}
	}
}

// TestRunSampleOpts checks that only flags the user actually set become SampleOpts:
// max-tokens is always present; sampling and penalty flags are no-ops at their zero
// defaults so an unset value does not silently override the planner.
func TestRunSampleOpts(t *testing.T) {
	// Defaults: only max-tokens.
	if got := len(runSampleOpts(512, 0, 0, 0, 0, 0)); got != 1 {
		t.Errorf("default opts = %d; want 1 (max-tokens only)", got)
	}
	// All set: six opts.
	if got := len(runSampleOpts(256, 0.7, 0.95, 40, 0.2, 1.5)); got != 6 {
		t.Errorf("all-set opts = %d; want 6", got)
	}
	// The opts must actually apply to a SampleParams without panicking.
	var sp agent.SampleParams
	for _, o := range runSampleOpts(128, 0.5, 0, 0, 0.2, 1.5) {
		o(&sp)
	}
	if sp.MaxTokens == nil || *sp.MaxTokens != 128 {
		t.Errorf("MaxTokens not applied: %v", sp.MaxTokens)
	}
	if sp.Temperature == nil || *sp.Temperature != 0.5 {
		t.Errorf("Temperature not applied: %v", sp.Temperature)
	}
	if sp.TopP != nil {
		t.Errorf("TopP should be unset (top-p=0), got %v", *sp.TopP)
	}
	if sp.FrequencyPenalty == nil || *sp.FrequencyPenalty != 0.2 {
		t.Errorf("FrequencyPenalty not applied: %v", sp.FrequencyPenalty)
	}
	if sp.PresencePenalty == nil || *sp.PresencePenalty != 1.5 {
		t.Errorf("PresencePenalty not applied: %v", sp.PresencePenalty)
	}
}

// TestCacheValueLine pins the WITNESSED per-turn cache-value summary `fak run` prints by
// default (#333). It is the DELTA of the cacheobs tap across one turn, so the line reports
// this turn's reuse, not the cumulative process total — and an idle turn (no prompt delta)
// prints nothing rather than a phantom 0/0 line.
func TestCacheValueLine(t *testing.T) {
	// A frozen turn: 1000 prompt tokens, 950 served from the cached KV prefix.
	before := cacheobs.Stats{PromptTokens: 200, ReusedTokens: 100}
	after := cacheobs.Stats{PromptTokens: 1200, ReusedTokens: 1050}
	got := cacheValueLine(before, after)
	for _, want := range []string{"reused 950/1000", "95% frozen", "by=vdso", "computed 50"} {
		if !strings.Contains(got, want) {
			t.Errorf("cacheValueLine = %q; missing %q", got, want)
		}
	}
	// A cold first turn (no reuse) lands in the cold regime.
	if got := cacheValueLine(cacheobs.Stats{}, cacheobs.Stats{PromptTokens: 500}); !strings.Contains(got, "0% cold") {
		t.Errorf("cold turn line = %q; want 0%% cold", got)
	}
	// A partial turn (between ColdCeil and FrozenFloor) is labeled partial.
	if got := cacheValueLine(cacheobs.Stats{}, cacheobs.Stats{PromptTokens: 100, ReusedTokens: 50}); !strings.Contains(got, "partial") {
		t.Errorf("partial turn line = %q; want partial", got)
	}
	// An idle turn with no prompt delta prints nothing (no phantom 0/0 line).
	if got := cacheValueLine(cacheobs.Stats{PromptTokens: 7}, cacheobs.Stats{PromptTokens: 7}); got != "" {
		t.Errorf("idle turn line = %q; want empty", got)
	}
}

// TestCacheTurnLine pins the showCache GATE runChatTurn applies — the wire that makes
// the #333 cache-value line actually fire (it was a dead `showCache` parameter before:
// threaded through runChatTurn/runChatREPL but never consumed). show=false (--quiet)
// must suppress entirely; show=true must render the same line cacheValueLine produces.
func TestCacheTurnLine(t *testing.T) {
	before := cacheobs.Stats{PromptTokens: 200, ReusedTokens: 100}
	after := cacheobs.Stats{PromptTokens: 1200, ReusedTokens: 1050}

	// --quiet (show=false): suppressed regardless of how much was reused.
	if got := cacheTurnLine(before, after, false); got != "" {
		t.Errorf("show=false must suppress the line, got %q", got)
	}
	// show=true: renders exactly what cacheValueLine produces for the same delta.
	got := cacheTurnLine(before, after, true)
	if want := cacheValueLine(before, after); got != want {
		t.Errorf("show=true line = %q; want %q", got, want)
	}
	if got == "" {
		t.Error("show=true with real reuse must render a non-empty line")
	}
	// show=true but an idle turn (no prompt delta) still prints nothing.
	if got := cacheTurnLine(cacheobs.Stats{PromptTokens: 7}, cacheobs.Stats{PromptTokens: 7}, true); got != "" {
		t.Errorf("idle turn must print nothing even when show=true, got %q", got)
	}
}
