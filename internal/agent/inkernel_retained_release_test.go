package agent

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"testing"
)

type retainedReleaseProbe struct {
	compute.Backend
	calls int
	text  string
}

func (p *retainedReleaseProbe) VulkanRetainedReleaseSummary() string { p.calls++; return p.text }

// fak-test:runtime fast est=5ms lane=default
func TestInKernelRetainedReleaseReporter(t *testing.T) {
	var nilPlanner *InKernelPlanner
	for _, p := range []*InKernelPlanner{nilPlanner, {}} {
		if _, ok := p.RetainedReleaseSummary(); ok {
			t.Fatal("unsupported reported available")
		}
	}
	b := &retainedReleaseProbe{text: "status=stale-generation"}
	p := &InKernelPlanner{backend: b}
	if b.calls != 0 {
		t.Fatal("eager query")
	}
	text, ok := p.RetainedReleaseSummary()
	if !ok || text != b.text || b.calls != 1 {
		t.Fatalf("text=%q supported=%t calls=%d", text, ok, b.calls)
	}
}
