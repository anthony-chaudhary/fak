package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestLeaserefUsageAndUnknown pins the no-git control paths: bare invocation and an
// unknown subcommand are usage errors (exit 2); help is exit 0.
func TestLeaserefUsageAndUnknown(t *testing.T) {
	cases := []struct {
		argv []string
		want int
	}{
		{nil, 2},
		{[]string{"bogus"}, 2},
		{[]string{"--help"}, 0},
		{[]string{"help"}, 0},
	}
	for _, c := range cases {
		var out, errb bytes.Buffer
		if got := runLeaseref(&out, &errb, c.argv); got != c.want {
			t.Fatalf("runLeaseref(%v) = %d, want %d (stderr=%q)", c.argv, got, c.want, errb.String())
		}
	}
}

// TestLeaserefLiveEndToEnd drives the verb against a REAL git temp repo (skipped when
// git is unavailable, e.g. the native-Windows path; runs under the WSL suite). It proves
// `fak leaseref live` emits the arbiter live_leases shape sourced from refs/fak/locks/*,
// drops an expired record, and `reap` removes it.
func TestLeaserefLiveEndToEnd(t *testing.T) {
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
	if _, err := store.Acquire(ctx, leaseref.Record{ID: "docs-lane", TreeGlobs: []string{"docs/**"}, Holder: "A:1", AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live: %v", err)
	}
	if _, err := store.Acquire(ctx, leaseref.Record{ID: "dead-lane", TreeGlobs: []string{"internal/x/**"}, Holder: "B:2", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire dead: %v", err)
	}

	// `live` emits only the non-expired lease, in the arbiter shape.
	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"live", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref live exit=%d stderr=%q", code, errb.String())
	}
	var leases []leaseref.ArbiterLease
	if err := json.Unmarshal(out.Bytes(), &leases); err != nil {
		t.Fatalf("live JSON unmarshal: %v\nout=%s", err, out.String())
	}
	if len(leases) != 1 || leases[0].Lane != "docs-lane" || leases[0].LaneKind != "cluster" {
		t.Fatalf("live leases = %+v, want exactly [{docs-lane cluster [docs/**]}]", leases)
	}
	if len(leases[0].Tree) != 1 || leases[0].Tree[0] != "docs/**" {
		t.Fatalf("live lease tree = %v, want [docs/**]", leases[0].Tree)
	}

	// `list` shows both, marking the expired one.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"list", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref list exit=%d stderr=%q", code, errb.String())
	}
	listing := out.String()
	if !strings.Contains(listing, "docs-lane") || !strings.Contains(listing, "LIVE") {
		t.Fatalf("list missing the live lease: %q", listing)
	}
	if !strings.Contains(listing, "dead-lane") || !strings.Contains(listing, "EXPIRED") {
		t.Fatalf("list missing the expired lease: %q", listing)
	}

	// `reap` deletes the one expired lease.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"reap", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref reap exit=%d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "reaped 1") {
		t.Fatalf("reap output = %q, want 'reaped 1 expired lease(s)'", out.String())
	}
	if _, ok, _ := store.Get(ctx, "dead-lane"); ok {
		t.Fatalf("dead-lane still present after reap")
	}
	if _, ok, _ := store.Get(ctx, "docs-lane"); !ok {
		t.Fatalf("reap wrongly removed the live docs-lane")
	}
}
