package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// TestMetalLeaseRefusalAdviceNamesHolderVerdict proves the `fak up` refusal tail
// from #13131 is verdict-shaped: a holder that is provably making progress is
// named as legitimate (NOT told to be killed), while a free/unknown lease keeps
// the historical stop-the-holder advice.
func TestMetalLeaseRefusalAdviceNamesHolderVerdict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	// No holder at all: keep the baseline advice, and never fabricate a verdict.
	advice := metalLeaseRefusalAdvice(path)
	if !strings.Contains(advice, "stop the holder process") {
		t.Fatalf("free-lease advice = %q, want the stop-the-holder baseline", advice)
	}
	if strings.Contains(advice, string(gpulease.HolderProgressLiveProgressing)) {
		t.Fatalf("free-lease advice fabricated a progressing verdict: %q", advice)
	}

	// A progressing holder must be named as legitimate and must NOT advise
	// killing it. Drive the probe verdict directly: a live, provably-progressing
	// holder cannot be staged deterministically in a test process.
	advice = metalLeaseRefusalAdviceFor(gpulease.HolderProbe{
		Verdict: gpulease.HolderProgressLiveProgressing,
		Held:    true,
		PID:     91975,
		Detail:  "holder pid is alive and consumed 20ms CPU over 250ms",
	}, path)
	if strings.Contains(advice, "stop the holder process") {
		t.Fatalf("progressing-holder advice still says stop the holder: %q", advice)
	}
	if !strings.Contains(advice, "making progress") {
		t.Fatalf("progressing-holder advice = %q, want it to name the progress", advice)
	}
	if !strings.Contains(advice, "pid 91975") {
		t.Fatalf("progressing-holder advice lost the holder identity: %q", advice)
	}
	// The advice must stay actionable: an alternative route plus the doctor hint.
	if !strings.Contains(advice, "CPU/non-Metal") || !strings.Contains(advice, "fak doctor serve") {
		t.Fatalf("progressing-holder advice lost its alternatives: %q", advice)
	}

	// A DEAD holder must NOT be named as legitimate, and must keep release advice.
	deadAdvice := metalLeaseRefusalAdviceFor(gpulease.HolderProbe{
		Verdict: gpulease.HolderProgressDead,
		Held:    true,
		PID:     4242,
		Detail:  "recorded holder pid is not a running process",
	}, path)
	if strings.Contains(deadAdvice, "making progress") {
		t.Fatalf("dead-holder advice claimed progress: %q", deadAdvice)
	}
	if !strings.Contains(deadAdvice, "no longer running") {
		t.Fatalf("dead-holder advice = %q, want it to name the dead holder", deadAdvice)
	}

	// An UNKNOWN (raced/unreadable) probe must fall back to the conservative
	// baseline rather than asserting a kill.
	unknownAdvice := metalLeaseRefusalAdviceFor(gpulease.HolderProbe{
		Verdict: gpulease.HolderProgressUnknown,
		Held:    true,
		PID:     gpulease.HolderUnknownPID,
		Detail:  "exclusive lock held but holder pid is unreadable",
	}, path)
	if !strings.Contains(unknownAdvice, "stop the holder process") {
		t.Fatalf("unknown-holder advice = %q, want the conservative baseline", unknownAdvice)
	}
}

// TestServeReservationsRowCarriesHolderVerdict proves the `fak doctor serve` row
// from #13131 surfaces the typed holder verdict, and that a progressing holder is
// advised to be waited for rather than stopped.
func TestServeReservationsRowCarriesHolderVerdict(t *testing.T) {
	progressing := serveReservationsRow(serveHostFacts{
		TotalBytes:            36 * testGiB,
		GPULeaseHeld:          true,
		GPULeasePath:          "/tmp/fak-gpu.lease",
		GPULeaseHolderVerdict: string(gpulease.HolderProgressLiveProgressing),
		GPULeaseHolderPID:     91975,
		GPULeaseHolderDetail:  "holder pid is alive and consumed 20ms CPU over 250ms",
	})
	if progressing == nil {
		t.Fatal("progressing holder row is nil")
	}
	for _, want := range []string{"pid 91975", string(gpulease.HolderProgressLiveProgressing), "wait for it to finish"} {
		if !strings.Contains(progressing.Finding+progressing.Remediation, want) {
			t.Errorf("progressing row missing %q: finding=%q remediation=%q", want, progressing.Finding, progressing.Remediation)
		}
	}
	if strings.Contains(progressing.Remediation, "stop the holder") || strings.Contains(progressing.Remediation, "release the GPU lease") {
		t.Errorf("progressing holder advised a stop/release: %q", progressing.Remediation)
	}

	stalled := serveReservationsRow(serveHostFacts{
		TotalBytes:            36 * testGiB,
		GPULeaseHeld:          true,
		GPULeasePath:          "/tmp/fak-gpu.lease",
		GPULeaseHolderVerdict: string(gpulease.HolderProgressStalled),
		GPULeaseHolderPID:     1234,
		GPULeaseHolderDetail:  "holder pid is alive but consumed only 0s CPU over 30s",
	})
	if stalled == nil || !strings.Contains(stalled.Remediation, "stalled") {
		t.Fatalf("stalled row = %+v, want a stalled remediation", stalled)
	}

	dead := serveReservationsRow(serveHostFacts{
		TotalBytes:            36 * testGiB,
		GPULeaseHeld:          true,
		GPULeasePath:          "/tmp/fak-gpu.lease",
		GPULeaseHolderVerdict: string(gpulease.HolderProgressDead),
		GPULeaseHolderPID:     4242,
		GPULeaseHolderDetail:  "recorded holder pid is not a running process",
	})
	if dead == nil || !strings.Contains(dead.Remediation, "gone") {
		t.Fatalf("dead row = %+v, want a gone-holder remediation", dead)
	}

	// An unreadable/raced probe (UNKNOWN, no pid) must keep the conservative
	// wait advice and must not claim the holder is stalled.
	unknown := serveReservationsRow(serveHostFacts{
		TotalBytes:            36 * testGiB,
		GPULeaseHeld:          true,
		GPULeasePath:          "/tmp/fak-gpu.lease",
		GPULeaseHolderVerdict: string(gpulease.HolderProgressUnknown),
		GPULeaseHolderDetail:  "holder pid is alive but its CPU-time signal is unobservable",
	})
	if unknown == nil {
		t.Fatal("unknown holder row is nil")
	}
	if strings.Contains(unknown.Remediation, "stalled") {
		t.Errorf("UNKNOWN holder advised a stalled release: %q", unknown.Remediation)
	}
	if !strings.Contains(unknown.Remediation, "wait for the active serve process") {
		t.Errorf("UNKNOWN holder lost the conservative wait advice: %q", unknown.Remediation)
	}
}

// TestProbeServeHostCarriesTypedVerdictWhenHeld proves the live probe path wires
// the holder verdict into serveHostFacts rather than only the pure row function.
func TestProbeServeHostCarriesTypedVerdictWhenHeld(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)

	lease, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("Acquire self lease: %v", err)
	}
	defer lease.Release()

	facts := probeServeHost(0, 0.1)
	if !facts.GPULeaseHeld {
		t.Fatal("probeServeHost did not observe the held lease")
	}
	if facts.GPULeaseHolderVerdict == "" {
		t.Fatal("probeServeHost left the holder verdict empty while the lease is held")
	}
	if facts.GPULeaseHolderPID != os.Getpid() {
		t.Fatalf("holder pid = %d, want %d", facts.GPULeaseHolderPID, os.Getpid())
	}
	if facts.GPULeaseHolderDetail == "" {
		t.Fatal("probeServeHost left the holder detail empty")
	}
}
