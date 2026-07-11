package ggufload

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// pinHostMemStatus swaps the loader's host-memory probe for a fixed snapshot so
// progress-line tests are deterministic on any machine (CI containers can be
// cpuset-confined, which would otherwise inject preflight lines here).
func pinHostMemStatus(t *testing.T, st compute.HostMemStatus) {
	t.Helper()
	prev := hostMemStatus
	hostMemStatus = func() compute.HostMemStatus { return st }
	t.Cleanup(func() { hostMemStatus = prev })
}

// TestLoadProgressMemConfined verifies the confinement visibility: a strict NUMA
// policy yields a one-time preflight line naming the allowed nodes, live
// rss/node-free tails on every progress line, and exactly one cliff warning once
// the confined free memory drops under the floor. This is the log trail that makes
// a CONSTRAINT_MEMORY_POLICY OOM kill legible instead of a silent vanishing.
func TestLoadProgressMemConfined(t *testing.T) {
	const gb = int64(1) << 30
	pinHostMemStatus(t, compute.HostMemStatus{
		RSS:         50 * gb,
		HostAvail:   400 * gb,
		Constrained: true,
		PolicyLabel: "bind:0",
		PolicyNodes: "0",
		PolicyFree:  2 * gb, // under the 4 GiB floor and under rss/8 — cliff territory
	})
	var buf bytes.Buffer
	p := NewLoadProfiler()
	p.Progress = &buf
	p.ProgressEvery = 50
	p.SetTotal(4)
	for i := 0; i < 4; i++ {
		p.Tick(gb)
	}
	out := buf.String()
	if !strings.Contains(out, "memory preflight") || !strings.Contains(out, "CONFINED to numa node(s) 0 (bind:0)") {
		t.Errorf("expected a confinement preflight line, got:\n%s", out)
	}
	if !strings.Contains(out, "CONSTRAINT_MEMORY_POLICY") {
		t.Errorf("preflight should name the dmesg signature to look for, got:\n%s", out)
	}
	if !strings.Contains(out, "rss 50.0 GB") || !strings.Contains(out, "node-free 2.0 GB") {
		t.Errorf("progress lines should carry live rss + confined free, got:\n%s", out)
	}
	if n := strings.Count(out, "WARNING: numa-confined memory nearly exhausted"); n != 1 {
		t.Errorf("cliff warning should fire exactly once, fired %d times:\n%s", n, out)
	}
	if strings.Count(out, "memory preflight") != 1 {
		t.Errorf("preflight should fire exactly once:\n%s", out)
	}
}

// TestLoadProgressMemUnconfined confirms the default regime stays quiet: an
// unconstrained process gets the rss tail (when known) and nothing else — no
// preflight, no warning, no node-free noise.
func TestLoadProgressMemUnconfined(t *testing.T) {
	const gb = int64(1) << 30
	pinHostMemStatus(t, compute.HostMemStatus{RSS: 3 * gb, HostAvail: 100 * gb})
	var buf bytes.Buffer
	p := NewLoadProfiler()
	p.Progress = &buf
	p.ProgressEvery = 50
	p.SetTotal(4)
	for i := 0; i < 4; i++ {
		p.Tick(gb)
	}
	out := buf.String()
	if strings.Contains(out, "preflight") || strings.Contains(out, "WARNING") || strings.Contains(out, "node-free") {
		t.Errorf("unconstrained load should not emit confinement lines:\n%s", out)
	}
	if !strings.Contains(out, "rss 3.0 GB") {
		t.Errorf("progress lines should still carry rss when known:\n%s", out)
	}
}
