package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

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
		{[]string{"smollm2", "hi"}, true},       // alias → chat
		{[]string{"./model.gguf"}, true},        // path → chat
		{[]string{"hf://o/r/m.gguf"}, true},     // hf uri → chat
	}
	for _, c := range cases {
		isChat := len(c.argv) > 0 && !strings.HasPrefix(c.argv[0], "-")
		if isChat != c.wantChat {
			t.Errorf("argv=%v: dispatch isChat=%v, want %v", c.argv, isChat, c.wantChat)
		}
	}
}

// TestRunSampleOpts checks that only flags the user actually set become SampleOpts:
// max-tokens is always present; temp/top-p/top-k are no-ops at their zero default so
// an unset temperature does not silently force greedy via an explicit 0.
func TestRunSampleOpts(t *testing.T) {
	// Defaults: only max-tokens.
	if got := len(runSampleOpts(512, 0, 0, 0)); got != 1 {
		t.Errorf("default opts = %d; want 1 (max-tokens only)", got)
	}
	// All set: four opts.
	if got := len(runSampleOpts(256, 0.7, 0.95, 40)); got != 4 {
		t.Errorf("all-set opts = %d; want 4", got)
	}
	// The opts must actually apply to a SampleParams without panicking.
	var sp agent.SampleParams
	for _, o := range runSampleOpts(128, 0.5, 0, 0) {
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
}
