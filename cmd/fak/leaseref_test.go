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

	// `audit` is the read-only dry-run evidence for the reaper: it explains
	// exactly which stale lease WOULD be reaped and why, while keeping the fresh
	// lease visible but unselected.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"audit", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref audit exit=%d stderr=%q", code, errb.String())
	}
	var audit struct {
		LiveCount    int              `json:"live_count"`
		ExpiredCount int              `json:"expired_count"`
		Live         []map[string]any `json:"live"`
		WouldReap    []map[string]any `json:"would_reap"`
	}
	if err := json.Unmarshal(out.Bytes(), &audit); err != nil {
		t.Fatalf("audit JSON unmarshal: %v\nout=%s", err, out.String())
	}
	if audit.LiveCount != 1 || audit.ExpiredCount != 1 {
		t.Fatalf("audit counts live=%d expired=%d, want 1/1", audit.LiveCount, audit.ExpiredCount)
	}
	if len(audit.WouldReap) != 1 {
		t.Fatalf("audit would_reap = %+v, want exactly the stale lease", audit.WouldReap)
	}
	stale := audit.WouldReap[0]
	if stale["id"] != "dead-lane" || stale["owner"] != "B:2" || stale["lane"] != "dead-lane" {
		t.Fatalf("stale evidence = %+v, want dead-lane owner B:2 lane dead-lane", stale)
	}
	if stale["stale"] != true || stale["reason"] != "TTL_EXPIRED" {
		t.Fatalf("stale verdict = %+v, want TTL_EXPIRED stale=true", stale)
	}
	if stale["age_threshold_seconds"] != float64(10) {
		t.Fatalf("stale threshold = %+v, want ttl threshold 10s", stale["age_threshold_seconds"])
	}
	if ev, _ := stale["evidence"].(string); !strings.Contains(ev, "ttl_seconds=10") || !strings.Contains(ev, "expires_at_unix=") {
		t.Fatalf("stale evidence string = %q, want ttl/expires comparison", ev)
	}
	if len(audit.Live) != 1 || audit.Live[0]["id"] != "docs-lane" || audit.Live[0]["stale"] != false {
		t.Fatalf("audit live rows = %+v, want fresh docs-lane not selected", audit.Live)
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

// TestLeaserefLivenessEndToEnd drives `fak leaseref liveness` against a REAL git temp
// repo (skipped when git is unavailable): leases bound to a heartbeating, a lapsed, and
// no session classify peer-live / peer-dead / peer-unknown, the reader's own lease
// classifies self, and reclaimable is peer-dead-only (#2164).
func TestLeaserefLivenessEndToEnd(t *testing.T) {
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
	now := time.Now().Unix()
	for _, r := range []leaseref.Record{
		{ID: "lane-peer", TreeGlobs: []string{"a/**"}, Holder: "A", SessionID: "sess-peer", AcquiredAt: now, TTLSeconds: 3600},
		{ID: "lane-dead", TreeGlobs: []string{"b/**"}, Holder: "B", SessionID: "sess-dead", AcquiredAt: now, TTLSeconds: 3600},
		{ID: "lane-legacy", TreeGlobs: []string{"c/**"}, Holder: "C", AcquiredAt: now, TTLSeconds: 3600},
		{ID: "lane-mine", TreeGlobs: []string{"d/**"}, Holder: "D", SessionID: "sess-me", AcquiredAt: now, TTLSeconds: 3600},
	} {
		if _, err := store.Acquire(ctx, r); err != nil {
			t.Fatalf("Acquire %s: %v", r.ID, err)
		}
	}
	for _, d := range []leaseref.SessionDescriptor{
		{ID: "sess-peer", Host: "h1", PCBState: "RUNNING", UpdatedAt: now, TTLSecs: 1800},
		{ID: "sess-dead", Host: "h2", PCBState: "RUNNING", UpdatedAt: now - 7200, TTLSecs: 60},
		{ID: "sess-me", Host: "h3", PCBState: "RUNNING", UpdatedAt: now, TTLSecs: 1800},
	} {
		if _, err := store.PublishSession(ctx, d); err != nil {
			t.Fatalf("PublishSession %s: %v", d.ID, err)
		}
	}

	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"liveness", "--session", "sess-me", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref liveness exit=%d stderr=%q", code, errb.String())
	}
	var rows []leaseref.ClassifiedLease
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		t.Fatalf("liveness JSON unmarshal: %v\nout=%s", err, out.String())
	}
	got := map[string]leaseref.ClassifiedLease{}
	for _, r := range rows {
		got[r.ID] = r
	}
	want := map[string]struct {
		class       string
		reclaimable bool
	}{
		"lane-peer":   {leaseref.LivenessPeerLive, false},
		"lane-dead":   {leaseref.LivenessPeerDead, true},
		"lane-legacy": {leaseref.LivenessPeerUnknown, false},
		"lane-mine":   {leaseref.LivenessSelf, false},
	}
	if len(rows) != len(want) {
		t.Fatalf("classified %d leases, want %d: %+v", len(rows), len(want), rows)
	}
	for id, w := range want {
		row, ok := got[id]
		if !ok {
			t.Fatalf("lease %s missing from liveness view", id)
		}
		if row.Liveness != w.class || row.Reclaimable != w.reclaimable {
			t.Fatalf("%s = {%s reclaimable=%v}, want {%s reclaimable=%v} (evidence=%q)",
				id, row.Liveness, row.Reclaimable, w.class, w.reclaimable, row.Evidence)
		}
	}

	// The --session binding rides the fenced acquire verb end to end.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"acquire", "--id", "lane-cli", "--holder", "E", "--session", "sess-me", "--ttl", "3600", "--tree", "e/**", "--dir", dir}); code != 0 {
		t.Fatalf("acquire exit=%d stderr=%q out=%q", code, errb.String(), out.String())
	}
	var acq fencedResult
	if err := json.Unmarshal(out.Bytes(), &acq); err != nil {
		t.Fatalf("acquire JSON: %v\nout=%s", err, out.String())
	}
	if !acq.Verdict.OK || acq.Record == nil || acq.Record.SessionID != "sess-me" {
		t.Fatalf("acquire = %+v, want ok with session sess-me bound", acq)
	}
}

// TestLeaserefFenceEndToEnd drives the fenced acquire/fence/renew verbs against a REAL git
// temp repo, exercising the actual `git update-ref` compare-and-swap and
// `git rev-parse --show-object-format` (skipped when git is unavailable). It seeds an EXPIRED
// holder, has a new holder TRANSITION over it (generation bumps), then proves the old
// holder's fence is refused STALE_LEASE (exit 3) while the new holder's is admitted.
func TestLeaserefFenceEndToEnd(t *testing.T) {
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
	// Seed an EXPIRED generation-1 lease held by A (AcquiredAt far in the past).
	if _, err := store.Acquire(ctx, leaseref.Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", AcquiredAt: 100, TTLSeconds: 10, Generation: 1}); err != nil {
		t.Fatalf("seed expired lease: %v", err)
	}

	// B acquires: the seeded lease is expired, so this is a TRANSITION -> generation 2.
	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"acquire", "--id", "lane", "--holder", "B", "--ttl", "3600", "--tree", "x/**", "--dir", dir}); code != 0 {
		t.Fatalf("acquire exit=%d stderr=%q out=%q", code, errb.String(), out.String())
	}
	var acq fencedResult
	if err := json.Unmarshal(out.Bytes(), &acq); err != nil {
		t.Fatalf("acquire JSON: %v\nout=%s", err, out.String())
	}
	if !acq.Verdict.OK || acq.Record == nil || acq.Record.Generation != 2 || acq.Record.Holder != "B" {
		t.Fatalf("acquire transition = %+v, want ok gen=2 holder=B", acq)
	}

	// A returns and fences with its stale generation 1 -> STALE_LEASE, exit 3.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"fence", "--id", "lane", "--holder", "A", "--generation", "1", "--dir", dir}); code != leaserefRefused {
		t.Fatalf("stale fence exit=%d, want %d (out=%q)", code, leaserefRefused, out.String())
	}
	var fv leaseref.FenceVerdict
	if err := json.Unmarshal(out.Bytes(), &fv); err != nil {
		t.Fatalf("fence JSON: %v\nout=%s", err, out.String())
	}
	if fv.OK || fv.Reason != leaseref.ReasonStaleLease || fv.Current != 2 {
		t.Fatalf("stale fence verdict = %+v, want refused STALE_LEASE current=2", fv)
	}

	// B fences with its current generation 2 -> OK, exit 0.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"fence", "--id", "lane", "--holder", "B", "--generation", "2", "--dir", dir}); code != 0 {
		t.Fatalf("live fence exit=%d, want 0 (out=%q stderr=%q)", code, out.String(), errb.String())
	}

	// B renews: generation stays 2, exit 0.
	out.Reset()
	errb.Reset()
	if code := runLeaseref(&out, &errb, []string{"renew", "--id", "lane", "--holder", "B", "--dir", dir}); code != 0 {
		t.Fatalf("renew exit=%d, want 0 (out=%q stderr=%q)", code, out.String(), errb.String())
	}
	var rn fencedResult
	if err := json.Unmarshal(out.Bytes(), &rn); err != nil {
		t.Fatalf("renew JSON: %v\nout=%s", err, out.String())
	}
	if !rn.Verdict.OK || rn.Record == nil || rn.Record.Generation != 2 || rn.Record.RenewedAt == 0 {
		t.Fatalf("renew = %+v, want ok gen=2 with RenewedAt set", rn)
	}
}
