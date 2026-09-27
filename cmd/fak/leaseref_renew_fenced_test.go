package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestLeaserefRenewRejectsStaleSameHolderGeneration(t *testing.T) {
	dir := gitInitKnownBad(t)
	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	now := time.Now()
	first, v, err := store.AcquireFenced(ctx, leaseref.Record{ID: "renew-lane", Holder: "worker", TTLSeconds: 300}, now)
	if err != nil || !v.OK {
		t.Fatalf("first acquire: verdict=%+v err=%v", v, err)
	}
	if v, err = store.ReleaseFenced(ctx, first.ID, first.Holder, first.Generation, now); err != nil || !v.OK {
		t.Fatalf("release first epoch: verdict=%+v err=%v", v, err)
	}
	second, v, err := store.AcquireFenced(ctx, leaseref.Record{ID: first.ID, Holder: first.Holder, TTLSeconds: 300}, now.Add(time.Second))
	if err != nil || !v.OK || second.Generation <= first.Generation {
		t.Fatalf("same-holder reacquire: record=%+v verdict=%+v err=%v", second, v, err)
	}

	for _, generation := range []int64{0, first.Generation} {
		var out, errb bytes.Buffer
		code := runLeaseref(&out, &errb, []string{"renew", "--dir", dir, "--id", first.ID, "--holder", first.Holder, "--generation", strconv.FormatInt(generation, 10), "--ttl", "600"})
		if code != leaserefRefused {
			t.Fatalf("renew generation %d exit=%d, want %d; out=%q err=%q", generation, code, leaserefRefused, out.String(), errb.String())
		}
		var got fencedResult
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("renew generation %d JSON: %v", generation, err)
		}
		if got.Verdict.OK || got.Verdict.Reason != leaseref.ReasonStaleLease || got.Verdict.Current != second.Generation {
			t.Fatalf("renew generation %d verdict=%+v, want STALE_LEASE current=%d", generation, got.Verdict, second.Generation)
		}
	}
	got, ok, err := store.Get(ctx, first.ID)
	if err != nil || !ok || got.Generation != second.Generation || got.TTLSeconds != second.TTLSeconds {
		t.Fatalf("stale renew changed replacement: got=%+v ok=%v err=%v", got, ok, err)
	}
}
