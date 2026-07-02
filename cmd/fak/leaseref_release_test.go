package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestLeaserefReleaseUsage pins the no-git control paths of the release verb: a missing
// --id or a missing --holder (without --force) is a usage error (exit 2).
func TestLeaserefReleaseUsage(t *testing.T) {
	for _, argv := range [][]string{
		{"release"},
		{"release", "--holder", "me"},
		{"release", "--id", "lane"},
	} {
		var out, errb bytes.Buffer
		if got := runLeaseref(&out, &errb, argv); got != 2 {
			t.Fatalf("runLeaseref(%v) = %d, want 2 (stderr=%q)", argv, got, errb.String())
		}
	}
}

// TestLeaserefReleaseEndToEnd drives the verb against a REAL git temp repo (skipped when
// git is unavailable): a wrong holder is refused STALE_LEASE (exit 3) and the lease
// survives; the right holder releases it (exit 0); releasing again is an idempotent OK;
// --force deletes without the holder check.
func TestLeaserefReleaseEndToEnd(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	rec, av, err := store.AcquireFenced(ctx, leaseref.Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "B", TTLSeconds: 3600}, time.Now())
	if err != nil || !av.OK {
		t.Fatalf("AcquireFenced: %+v %v", av, err)
	}

	// A different holder may not release B's live lease: refused STALE_LEASE, exit 3.
	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"release", "--id", "lane", "--holder", "A", "--dir", dir}); code != leaserefRefused {
		t.Fatalf("wrong-holder release exit=%d, want %d (out=%q)", code, leaserefRefused, out.String())
	}
	var v leaseref.FenceVerdict
	if err := json.Unmarshal(out.Bytes(), &v); err != nil {
		t.Fatalf("release JSON: %v\nout=%s", err, out.String())
	}
	if v.OK || v.Reason != leaseref.ReasonStaleLease || v.Holder != "B" {
		t.Fatalf("wrong-holder verdict = %+v, want refused STALE_LEASE held by B", v)
	}
	if _, ok, _ := store.Get(ctx, "lane"); !ok {
		t.Fatalf("refused release still deleted the lease")
	}

	// The holder releases its own lease, presenting its fencing token: exit 0, ref gone.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"release", "--id", "lane", "--holder", "B", "--generation", strconv.FormatInt(rec.Generation, 10), "--dir", dir}); code != 0 {
		t.Fatalf("holder release exit=%d, want 0 (out=%q stderr=%q)", code, out.String(), errb.String())
	}
	if _, ok, _ := store.Get(ctx, "lane"); ok {
		t.Fatalf("lease still present after the holder's release")
	}

	// Releasing an already-released lease is an idempotent OK (exit 0).
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"release", "--id", "lane", "--holder", "B", "--dir", dir}); code != 0 {
		t.Fatalf("idempotent release exit=%d, want 0 (out=%q)", code, out.String())
	}
	if err := json.Unmarshal(out.Bytes(), &v); err != nil {
		t.Fatalf("idempotent release JSON: %v\nout=%s", err, out.String())
	}
	if !v.OK || !strings.Contains(v.Detail, "already absent") {
		t.Fatalf("idempotent verdict = %+v, want OK already-absent", v)
	}

	// --force deletes a live record without the holder check (the operator escape).
	if _, av, err = store.AcquireFenced(ctx, leaseref.Record{ID: "wedged", TreeGlobs: []string{"y/**"}, Holder: "gone-host:dead-sess", TTLSeconds: 3600}, time.Now()); err != nil || !av.OK {
		t.Fatalf("AcquireFenced wedged: %+v %v", av, err)
	}
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"release", "--id", "wedged", "--force", "--dir", dir}); code != 0 {
		t.Fatalf("force release exit=%d, want 0 (stderr=%q)", code, errb.String())
	}
	if _, ok, _ := store.Get(ctx, "wedged"); ok {
		t.Fatalf("wedged lease still present after --force release")
	}
}
