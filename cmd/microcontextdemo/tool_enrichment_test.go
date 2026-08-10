package main

import (
	"context"
	"path/filepath"
	"testing"
)

func TestToolEnrichmentSelfcheck(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tool-ledger.json")
	if err := runToolEnrichmentSelfcheck(context.Background(), path, 16); err != nil {
		t.Fatal(err)
	}
	if err := verifyToolEnrichmentArtifact(path); err != nil {
		t.Fatal(err)
	}
}

func TestReadCatalogDeniesWrites(t *testing.T) {
	c := &readCoordinator{backend: &fixtureReadBackend{attempts: map[string]int{}}, globalQuota: 1, perResourceQuota: 1, depthCap: 1, amplificationCap: 2, cache: map[string]readObservation{}, inflight: map[string]chan struct{}{}, admitted: map[string]int{}}
	o := c.run(context.Background(), readRequest{RequestID: "x", Capability: "close-issue", Resource: "issue:1", TimeoutMS: 1})
	if o.Reason != "capability_denied" || o.Dispatched {
		t.Fatalf("write escaped catalog: %+v", o)
	}
}

func TestReadDedupeAndQuota(t *testing.T) {
	b := &fixtureReadBackend{attempts: map[string]int{}}
	c := &readCoordinator{backend: b, globalQuota: 1, perResourceQuota: 1, depthCap: 1, amplificationCap: 2, cache: map[string]readObservation{}, inflight: map[string]chan struct{}{}, admitted: map[string]int{}}
	r := readRequest{RequestID: "a", Capability: "fetch-comments", Resource: "issue:1", Arguments: map[string]string{"issue_id": "1"}, TimeoutMS: 10}
	first := c.run(context.Background(), r)
	r.RequestID = "b"
	second := c.run(context.Background(), r)
	if first.Status != "observed" || !second.CacheHit || !second.Deduped || b.calls.Load() != 1 {
		t.Fatalf("dedupe failed: first=%+v second=%+v calls=%d", first, second, b.calls.Load())
	}
	r.Resource = "issue:2"
	r.Arguments = map[string]string{"issue_id": "2"}
	third := c.run(context.Background(), r)
	if third.Status != "not_run" || third.Reason != "quota_denied" || third.Dispatched {
		t.Fatalf("quota failed closed: %+v", third)
	}
}

func TestVerifyReadLedgerRejectsUnsafeClaims(t *testing.T) {
	base := readLedger{Schema: toolEnrichmentSchema, Records: 1000, SelectorWindows: 28, LogicalRequests: 33, UniqueRequests: 26, DedupeHits: 4, QuotaDenials: 1, CancelledUnopened: 2, ToolInvocations: 27, Timeouts: 1, Retries: 1, RestartCacheHits: 26, GlobalQuota: 26, PeakToolConcurrency: 16, DepthCap: 2, AmplificationCap: 4, Observed: 25, ReadbackVerified: 25, FoldCitations: make([]string, 25)}
	cases := []struct {
		name   string
		mutate func(*readLedger)
	}{
		{"cancel-dispatched", func(l *readLedger) { l.CancelledUnopenedDispatches = 1 }},
		{"restart-dispatched", func(l *readLedger) { l.RestartToolInvocations = 1 }},
		{"model-slot-pinned", func(l *readLedger) { l.ModelSlotsDuringToolWait = 1 }},
		{"depth-overflow", func(l *readLedger) { l.MaxOutputDepth = 3 }},
		{"amplification-overflow", func(l *readLedger) { l.MaxAmplification = 5 }},
		{"readback-missing", func(l *readLedger) { l.ReadbackVerified = 24 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := base
			got.FoldCitations = append([]string(nil), base.FoldCitations...)
			tc.mutate(&got)
			if verifyReadLedger(got) == nil {
				t.Fatal("unsafe ledger accepted")
			}
		})
	}
}
