package ggufload

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func pinHostMemStatusFunc(t *testing.T, fn func() compute.HostMemStatus) {
	t.Helper()
	prev := hostMemStatus
	hostMemStatus = fn
	t.Cleanup(func() { hostMemStatus = prev })
}

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

func TestLoadProfileMemorySamplesPreserveProbeAndJSONContracts(t *testing.T) {
	statuses := []compute.HostMemStatus{
		{RSS: 10, HostAvail: 0},
		{HostAvail: compute.FreeUnknown},
		{RSS: 30, HostAvail: 40, Constrained: true, PolicyFree: 20, PolicyNodes: "0-1", PolicyLabel: "bind"},
	}
	probes := 0
	pinHostMemStatusFunc(t, func() compute.HostMemStatus {
		st := statuses[probes]
		probes++
		return st
	})

	p := NewLoadProfiler()
	p.Progress = io.Discard
	p.ProgressEvery = 50
	p.SetTotal(4)
	for range 4 {
		p.Tick(1)
	}
	if probes != 3 {
		t.Fatalf("memory probes = %d, want 3 existing emitted progress points", probes)
	}
	profile := p.Snapshot("q4k", "fixture.gguf", 1)
	if profile.MemorySamplesObserved != 3 || profile.MemorySamplesDropped != 0 || profile.MaxSampledRSSBytes != 30 {
		t.Fatalf("memory summary = observed %d dropped %d max %d", profile.MemorySamplesObserved, profile.MemorySamplesDropped, profile.MaxSampledRSSBytes)
	}
	if len(profile.MemorySamples) != 3 || profile.MemorySamples[0].TensorOrdinal != 1 || profile.MemorySamples[1].TensorOrdinal != 3 || profile.MemorySamples[2].TensorOrdinal != 4 {
		t.Fatalf("memory samples = %+v, want ordinals 1,3,4", profile.MemorySamples)
	}
	if got := profile.MemorySamples[0].HostAvailBytes; got == nil || *got != 0 {
		t.Fatalf("measured zero host available = %v, want pointer to zero", got)
	}
	if profile.MemorySamples[1].HostAvailBytes != nil {
		t.Fatalf("unknown host available = %v, want nil", *profile.MemorySamples[1].HostAvailBytes)
	}
	last := profile.MemorySamples[2]
	if last.RSSBytes != 30 || last.HostAvailBytes == nil || *last.HostAvailBytes != 40 || !last.Constrained || last.PolicyFreeBytes != 20 || last.PolicyNodes != "0-1" || last.PolicyLabel != "bind" {
		t.Fatalf("last memory sample = %+v", last)
	}

	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, field := range []string{"\"memory_samples\"", "\"memory_samples_observed\":3", "\"memory_samples_dropped\":0", "\"max_sampled_rss_bytes\":30", "\"tensor_ordinal\":1", "\"elapsed_nanos\"", "\"host_avail_bytes\":0", "\"policy_free_bytes\":20"} {
		if !strings.Contains(text, field) {
			t.Errorf("profile JSON missing %s: %s", field, text)
		}
	}
	if strings.Contains(text, "\"rss_bytes\":0") || strings.Contains(text, "\"host_avail_bytes\":-1") || strings.Contains(text, "\"policy_free_bytes\":0") {
		t.Fatalf("unknown memory values must be omitted: %s", text)
	}

	profile.MemorySamples[0].TensorOrdinal = 99
	*profile.MemorySamples[0].HostAvailBytes = 99
	again := p.Snapshot("q4k", "fixture.gguf", 1)
	if again.MemorySamples[0].TensorOrdinal != 1 || again.MemorySamples[0].HostAvailBytes == nil || *again.MemorySamples[0].HostAvailBytes != 0 {
		t.Fatalf("Snapshot leaked caller mutation: %+v", again.MemorySamples[0])
	}
}

func TestLoadProfileMemorySamplesBoundedRetentionPreservesOverallMax(t *testing.T) {
	call := 0
	pinHostMemStatusFunc(t, func() compute.HostMemStatus {
		call++
		rss := int64(call)
		if call == 260 {
			rss = 10_000
		}
		return compute.HostMemStatus{RSS: rss, HostAvail: compute.FreeUnknown}
	})
	p := NewLoadProfiler()
	p.Progress = io.Discard
	p.ProgressEvery = 0.001
	p.SetTotal(300)
	for range 300 {
		p.Tick(1)
	}
	profile := p.Snapshot("q4k", "fixture.gguf", 1)
	if call != 300 || profile.MemorySamplesObserved != 300 || profile.MemorySamplesDropped != 44 || len(profile.MemorySamples) != 256 {
		t.Fatalf("retention = probes %d observed %d dropped %d retained %d", call, profile.MemorySamplesObserved, profile.MemorySamplesDropped, len(profile.MemorySamples))
	}
	if profile.MemorySamples[254].TensorOrdinal != 255 || profile.MemorySamples[255].TensorOrdinal != 300 {
		t.Fatalf("retention endpoints = %d,%d, want 255,300", profile.MemorySamples[254].TensorOrdinal, profile.MemorySamples[255].TensorOrdinal)
	}
	if profile.MaxSampledRSSBytes != 10_000 {
		t.Fatalf("max sampled RSS = %d, want dropped-sample max 10000", profile.MaxSampledRSSBytes)
	}
}

func TestLoadProfileMemorySamplesDoNotAddProbeWithoutWriter(t *testing.T) {
	probes := 0
	pinHostMemStatusFunc(t, func() compute.HostMemStatus {
		probes++
		return compute.HostMemStatus{RSS: 1, HostAvail: 2}
	})
	p := NewLoadProfiler()
	p.SetTotal(1)
	p.Tick(1)
	profile := p.Snapshot("q4k", "fixture.gguf", 1)
	if probes != 0 || profile.MemorySamplesObserved != 0 || len(profile.MemorySamples) != 0 {
		t.Fatalf("writer-free profiler probed/stored memory: probes=%d profile=%+v", probes, profile)
	}
}
