package main

import (
	"context"
	"fmt"
	"io"
	"time"
)

const (
	capturedSelfcheckContexts = 32
	capturedSelfcheckWorkers  = 4
)

// runCapturedSelfcheck exercises the smallest complete synthetic spine and
// renders only contract fields. Timing and host resource measurements remain in
// the full JSON report because they are observations, not deterministic facts.
func runCapturedSelfcheck(ctx context.Context, out io.Writer) error {
	r, err := run(ctx, config{
		Contexts:  capturedSelfcheckContexts,
		Workers:   capturedSelfcheckWorkers,
		Delay:     2 * time.Millisecond,
		Selfcheck: true,
	})
	if err != nil {
		return err
	}
	if r.Schema != "fak-microcontext-spine/1" || r.Verdict != "PASS" {
		return fmt.Errorf("unexpected contract: schema=%q verdict=%q", r.Schema, r.Verdict)
	}
	if r.Completed != capturedSelfcheckContexts || r.Failed != 0 || r.TurnCount != capturedSelfcheckContexts {
		return fmt.Errorf("incomplete work: completed=%d failed=%d turns=%d", r.Completed, r.Failed, r.TurnCount)
	}
	if r.SharedBaseInstalls != 1 || r.BaseFingerprint != canonicalBaseFingerprint() {
		return fmt.Errorf("shared base invariant failed: installs=%d fingerprint=%q", r.SharedBaseInstalls, r.BaseFingerprint)
	}
	if r.PeakInFlight < 2 || r.PeakInFlight > capturedSelfcheckWorkers {
		return fmt.Errorf("bounded concurrency invariant failed: peak=%d workers=%d", r.PeakInFlight, capturedSelfcheckWorkers)
	}

	fmt.Fprintln(out, "PASS fak-microcontext-spine/1 synthetic fixture")
	fmt.Fprintf(out, "PASS logical_contexts=%d physical_workers=%d completed=%d failed=%d\n", r.LogicalShards, r.PhysicalWorkers, r.Completed, r.Failed)
	fmt.Fprintf(out, "PASS shared_base_installs=%d turns=%d bounded_peak<=%d\n", r.SharedBaseInstalls, r.TurnCount, capturedSelfcheckWorkers)
	fmt.Fprintln(out, "CLAIM bounded fan-out and shared-base accounting only; not model quality, throughput, KV residency, or GPU evidence")
	return nil
}
