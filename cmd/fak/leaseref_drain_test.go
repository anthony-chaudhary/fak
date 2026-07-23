package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestLeaserefDrainDryRunReportsExpiredMutatesNothing proves the `fak leaseref drain` verb's
// SAFE DEFAULT end to end: with no --apply it reports exactly the proven-expired session
// descriptor as the drain target, pushes nothing, reaps nothing, and leaves BOTH the live and
// the expired local refs on disk — the dry-run-by-default discipline every retention collector
// shares, so scheduling the verb can never bulk-delete a remote by accident (#5358). The live
// --apply origin mutation is proven at the library layer against a real two-clone temp remote.
func TestLeaserefDrainDryRunReportsExpiredMutatesNothing(t *testing.T) {
	dir := lockReapInitRepo(t)
	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	if _, err := store.PublishSession(ctx, leaseref.SessionDescriptor{ID: "liveX", Host: "n1", PCBState: "RUNNING", UpdatedAt: time.Now().Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("publish live: %v", err)
	}
	if _, err := store.PublishSession(ctx, leaseref.SessionDescriptor{ID: "deadX", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("publish dead: %v", err)
	}

	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"drain", "--dir", dir}); code != 0 {
		t.Fatalf("drain dry-run exit=%d stderr=%q", code, errb.String())
	}
	var res leaseref.DescriptorDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("decode drain json: %v\n%s", err, out.String())
	}
	if res.Applied || res.Pushed != 0 || res.ReapedLocal != 0 {
		t.Fatalf("dry-run must not act, got %+v", res)
	}
	if len(res.ExpiredIDs) != 1 || res.ExpiredIDs[0] != "deadX" {
		t.Fatalf("dry-run ExpiredIDs = %v, want [deadX]", res.ExpiredIDs)
	}
	if res.LiveExcluded != 1 {
		t.Fatalf("dry-run LiveExcluded = %d, want 1 (the live descriptor)", res.LiveExcluded)
	}
	// Read-only: producing the plan removes neither descriptor.
	if _, ok, _ := store.GetSession(ctx, "deadX"); !ok {
		t.Fatalf("dry-run reaped the expired descriptor — it must be read-only")
	}
	if _, ok, _ := store.GetSession(ctx, "liveX"); !ok {
		t.Fatalf("dry-run reaped the live descriptor")
	}
}
