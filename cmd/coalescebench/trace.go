// trace.go adds the real-activation-trace witness path to coalescebench (#1298).
//
// The synthetic sweep in main.go projects a roofline from a seeded router. This
// file instead replays a CAPTURED (or re-derived) expert-activation trace through
// the deterministic LRU in package deepseekv4moe and emits a [SW-VERIFIED] receipt
// for that one trace. It is still not a served measurement — no hardware throughput
// claim — but the hit-rate numbers describe real routed groups rather than a
// synthetic generator's output.
//
// Fail-closed contract: an absent, unreadable, malformed, or route-less trace is a
// typed `no-trace` verdict wrapping deepseekv4moe.ErrNoTrace — never a fabricated
// hit rate.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deepseekv4moe"
)

// loadActivationTrace reads an ExpertActivationTrace from a JSON file. Every
// failure mode — empty path, read error, JSON error, route-less trace — surfaces
// as an error wrapping deepseekv4moe.ErrNoTrace so callers can fail closed with
// errors.Is(err, deepseekv4moe.ErrNoTrace).
func loadActivationTrace(path string) (deepseekv4moe.ExpertActivationTrace, error) {
	var trace deepseekv4moe.ExpertActivationTrace
	if strings.TrimSpace(path) == "" {
		return trace, fmt.Errorf("no activation trace path given: %w", deepseekv4moe.ErrNoTrace)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return trace, fmt.Errorf("read activation trace %q: %w", path, errors.Join(err, deepseekv4moe.ErrNoTrace))
	}
	if err := json.Unmarshal(raw, &trace); err != nil {
		return trace, fmt.Errorf("parse activation trace %q: %w", path, errors.Join(err, deepseekv4moe.ErrNoTrace))
	}
	if len(trace.Routes) == 0 {
		return trace, fmt.Errorf("activation trace %q has no routes: %w", path, deepseekv4moe.ErrNoTrace)
	}
	return trace, nil
}

// traceWitnessReceipt is the optional machine-readable companion to the text
// receipt. The witness struct is carried verbatim under a stable key so a consumer
// reads exactly what WitnessExpertCacheHitRate returned.
type traceWitnessReceipt struct {
	Schema  string                                  `json:"schema"`
	Witness deepseekv4moe.ExpertCacheHitRateWitness `json:"witness"`
}

// runTraceWitness loads one activation trace, replays it through the deterministic
// LRU witness, and writes a deterministic human-readable receipt to out. When
// artifactOut is non-empty an additive JSON receipt carrying the witness verbatim
// is also written. Any load failure is a typed no-trace verdict printed to out and
// returned.
func runTraceWitness(path, artifactOut string, out io.Writer) error {
	trace, err := loadActivationTrace(path)
	if err != nil {
		fmt.Fprintf(out, "coalescebench: no-trace verdict: %v\n", err)
		return err
	}

	witness, err := deepseekv4moe.WitnessExpertCacheHitRate(trace)
	if err != nil {
		fmt.Fprintf(out, "coalescebench: witness error: %v\n", err)
		return err
	}

	fmt.Fprintf(out, "[SW-VERIFIED] expert-cache hit-rate witness (schema=%s)\n", witness.Schema)
	fmt.Fprintf(out, "trace: name=%s source=%s\n", witness.TraceName, witness.TraceSource)
	fmt.Fprintf(out, "policy=%s top_k=%d cache_groups=%d cache_bytes=%d group_bytes=%d\n",
		witness.Policy, witness.TopK, witness.CacheGroups, witness.CacheBytes, witness.GroupBytes)
	fmt.Fprintf(out, "accesses=%d hits=%d misses=%d hit_rate=%.6f (hit_rate_known=%t)\n",
		witness.Accesses, witness.Hits, witness.Misses, witness.HitRate, witness.HitRateKnown)
	fmt.Fprintf(out, "byte_weighted_hit_rate=%.6f (known=%t) bytes_read_resident=%d bytes_read_streamed=%d\n",
		witness.ByteWeightedHitRate, witness.ByteWeightedHitKnown, witness.BytesReadResident, witness.BytesReadStreamed)
	fmt.Fprintf(out, "belady_optimal_hits=%d belady_optimal_misses=%d belady_regret_hits=%d belady_regret_ratio=%.6f belady_exact=%t\n",
		witness.BeladyOptimalHits, witness.BeladyOptimalMisses, witness.BeladyRegretHits, witness.BeladyRegretRatio, witness.BeladyExact)
	fmt.Fprintf(out, "peak_resident=%d\n", witness.PeakResident)
	fmt.Fprintf(out, "NOT a served measurement: SW-VERIFIED replay of a captured/re-derived activation trace; no hardware throughput claim.\n")

	if p := strings.TrimSpace(artifactOut); p != "" {
		report := traceWitnessReceipt{Schema: deepseekv4moe.ExpertCacheHitRateWitnessSchema, Witness: witness}
		raw, err := json.Marshal(report)
		if err != nil {
			return fmt.Errorf("marshal trace-witness receipt: %w", err)
		}
		if err := os.WriteFile(p, append(raw, '\n'), 0o644); err != nil {
			return fmt.Errorf("write trace-witness receipt %q: %w", p, err)
		}
	}
	return nil
}
