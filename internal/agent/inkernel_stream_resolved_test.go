package agent

import (
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestStreamResolvedSampleControlsAndObserverOrder(t *testing.T) {
	calls, original, stream := 0, 0, 0
	opt := func(sp *SampleParams) {
		calls++
		sp.Stop = []string{"stop"}
		sp.DecodeTokenObserver = func(string, string) { original++ }
	}
	sp := applySampleOpts(opt)
	opts := streamResolvedSampleOpts(sp)
	copied := applySampleOpts(opts...)
	copied.DecodeTokenObserver("a", "a")
	if calls != 1 || original != 1 || copied.DecodeTrace {
		t.Fatal("buffered controls/observer changed or original option replayed")
	}
	live := applySampleOpts(append(opts, WithDecodeTokenObserver(func(string, string) { stream++ }))...)
	live.DecodeTokenObserver("b", "b")
	if calls != 1 || original != 1 || stream != 1 || !live.DecodeTrace {
		t.Fatal("stream observer override ordering changed")
	}
	// Explicitly document the narrow contract: this adapter is not a deep freeze.
	sp.Stop[0] = "shared"
	if copied.Stop[0] != "shared" {
		t.Fatal("unexpected deep-copy contract")
	}
	live.Stop = nil
	again := applySampleOpts(opts...)
	if len(again.Stop) != 1 {
		t.Fatal("struct copy was overwritten by a later consumer")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCompleteStreamResolvedAdapterSourceGuard(t *testing.T) {
	data, err := os.ReadFile("inkernel_planner.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	a := strings.Index(src, "func (p *InKernelPlanner) CompleteStream(")
	if a < 0 {
		t.Fatal("CompleteStream missing")
	}
	b := strings.Index(src[a:], "// inKernelStreamProjector")
	if b < 0 {
		t.Fatal("stream boundary missing")
	}
	body := src[a : a+b]
	fold := strings.Index(body, "sp := applySampleOpts(opts...)")
	adapter := strings.Index(body, "opts = streamResolvedSampleOpts(sp)")
	sink := strings.Index(body, "if sink == nil")
	if fold < 0 || adapter <= fold || sink <= adapter || strings.Count(body, "opts = streamResolvedSampleOpts(sp)") != 1 {
		t.Fatal("adapter must run once after resolution and before all Complete branches")
	}
	if strings.Count(body, "p.Complete(") != 4 || strings.Count(body, "append(opts, WithDecodeTokenObserver(") != 2 {
		t.Fatal("completion/observer branches changed; re-audit semantics")
	}
}
