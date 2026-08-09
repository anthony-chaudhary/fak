package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"os"
	"time"
)

const compatSchema = "fak-microcontext-compatibility/1"

type compatReport struct {
	Schema             string     `json:"schema"`
	Verdict            string     `json:"verdict"`
	ObservedAt         string     `json:"observed_at"`
	Provenance         string     `json:"provenance"`
	Submitted          int        `json:"submitted"`
	Cancelled          int        `json:"cancelled"`
	Rejected           int        `json:"rejected"`
	Scheduled          int        `json:"scheduled"`
	Batches            int        `json:"batches"`
	SingletonFallbacks int        `json:"singleton_fallbacks"`
	MaxQueueAgeMillis  int64      `json:"max_queue_age_ms"`
	PaddingTax         float64    `json:"padding_tax"`
	BatchFill          float64    `json:"batch_fill"`
	NominalUse         float64    `json:"nominal_use"`
	BatchIDs           [][]string `json:"batch_ids"`
	Claims             []string   `json:"claims"`
	NonClaims          []string   `json:"non_claims"`
}

func buildCompatReport() (compatReport, error) {
	now := time.Now()
	keys := []microagent.CompatibilityKey{{Model: "model-a", Sampling: "temp-0", Tools: "none", Prefix: "base-a", Phase: "prefill", SequenceBucket: 128}, {Model: "model-a", Sampling: "temp-0", Tools: "read", Prefix: "base-a", Phase: "prefill", SequenceBucket: 128}, {Model: "model-b", Sampling: "temp-0", Tools: "none", Prefix: "base-b", Phase: "decode", SequenceBucket: 64}}
	var work []microagent.CompatibleWork
	for i := 0; i < 96; i++ {
		k := keys[i%len(keys)]
		work = append(work, microagent.CompatibleWork{ID: fmt.Sprintf("w-%03d", i), Key: k, Tokens: 90 + i%20, Priority: i % 4, Enqueued: now.Add(-time.Duration(i%12) * time.Millisecond)})
	}
	work = append(work, microagent.CompatibleWork{ID: "unknown", Key: microagent.CompatibilityKey{Model: "model-a"}, Tokens: 50, Enqueued: now.Add(-20 * time.Millisecond)}, microagent.CompatibleWork{ID: "cancelled", Key: keys[0], Tokens: 100, Cancelled: true})
	b, s, e := microagent.ComposeCompatible(work, microagent.CompatibilityConfig{MaxBatch: 8, MaxQueuePerClass: 64, MaxPadding: .10, StarvationAfter: 5 * time.Millisecond, Now: now})
	if e != nil {
		return compatReport{}, e
	}
	r := compatReport{Schema: compatSchema, Verdict: "PASS", ObservedAt: now.UTC().Format(time.RFC3339), Provenance: "observed deterministic controlled-kernel mixed-workload planner", Submitted: s.Submitted, Cancelled: s.Cancelled, Rejected: s.Rejected, Scheduled: s.Scheduled, Batches: s.Batches, SingletonFallbacks: s.SingletonFallbacks, MaxQueueAgeMillis: s.MaxQueueAge.Milliseconds(), PaddingTax: s.PaddingTax, BatchFill: s.BatchFill, NominalUse: s.NominalUse, Claims: []string{"compatible work coalesced while model/tool/phase classes remained isolated", "aging, cancellation, bounded queues, padding cap, and singleton fail-open were exercised"}, NonClaims: []string{"planner fill and padding are not model throughput, TTFT, or GPU slot telemetry"}}
	for _, x := range b {
		r.BatchIDs = append(r.BatchIDs, x.IDs)
	}
	if e = verifyCompatibilityReport(r); e != nil {
		r.Verdict = "FAIL"
	}
	return r, e
}
func verifyCompatibilityReport(r compatReport) error {
	if r.Schema != compatSchema || r.Verdict != "PASS" || r.Submitted != 98 || r.Scheduled != 97 || r.Cancelled != 1 || r.Rejected != 0 || r.SingletonFallbacks != 1 || r.PaddingTax > .10 || r.Batches == 0 {
		return errors.New("compatibility witness invariant failed")
	}
	return nil
}
func verifyCompatibilityArtifact(p string) error {
	b, e := os.ReadFile(p)
	if e != nil {
		return e
	}
	var r compatReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyCompatibilityReport(r)
}
