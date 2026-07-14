package main

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestGuardFleetFromDoc folds a fleetpane.FleetDoc into the gateway display shape: the
// aggregate counts come from States/Totals, the rows are read (and capped) from the
// per-machine maps, and an empty fleet reports ok=false so the block is omitted.
func TestGuardFleetFromDoc(t *testing.T) {
	// Empty fleet → omitted.
	if _, ok := guardFleetFromDoc(fleetpane.FleetDoc{}); ok {
		t.Fatal("guardFleetFromDoc(empty) ok=true, want false (block omitted)")
	}

	doc := fleetpane.FleetDoc{
		Verdict: "ACTION",
		States:  map[string]int{"OK": 1, "STALE": 1, "ACTION": 1},
		Totals:  map[string]int{"sessions": 9, "auth_blocked": 2, "version_mismatches": 1},
		Machines: []map[string]any{
			{"id": "alpha", "state": "OK", "age_min": float64(3), "sessions": float64(5), "app_version": "v9.9.9"},
			{"id": "beta", "state": "STALE", "age_min": float64(120), "sessions": 0},
			{"id": "gamma", "state": "ACTION"},
		},
	}
	f, ok := guardFleetFromDoc(doc)
	if !ok {
		t.Fatal("guardFleetFromDoc(populated) ok=false, want true")
	}
	if f.Verdict != "ACTION" || f.Machines != 3 || f.Stale != 1 || f.Action != 1 {
		t.Fatalf("aggregate = %+v, want ACTION/3/1 stale/1 action", f)
	}
	if f.Sessions != 9 || f.AuthBlocked != 2 || f.VersionMismatches != 1 {
		t.Fatalf("totals = %+v, want 9 sessions / 2 auth-blocked / 1 version-skew", f)
	}
	if len(f.Rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(f.Rows))
	}
	// Row 0 reads all fields (numbers arrive as float64 from JSON-decoded snapshots).
	// The version is asserted by pass-through — the row carries the source machine's
	// app_version verbatim — rather than frozen to a literal that would drift.
	r := f.Rows[0]
	wantVersion := doc.Machines[0]["app_version"].(string)
	if r.ID != "alpha" || r.State != "OK" || r.AgeMin != 3 || r.Sessions != 5 || r.Version != wantVersion {
		t.Fatalf("row0 = %+v, want alpha OK 3m 5sess version=%q", r, wantVersion)
	}
}

// TestGuardFleetFromDocCapsRows proves the per-machine row list is bounded to
// guardFleetMachineRows even when the fleet has more machines; the aggregate machine
// COUNT still reflects the whole fleet.
func TestGuardFleetFromDocCapsRows(t *testing.T) {
	machines := make([]map[string]any, 0, guardFleetMachineRows+5)
	for i := 0; i < guardFleetMachineRows+5; i++ {
		machines = append(machines, map[string]any{"id": string(rune('a' + i)), "state": "OK"})
	}
	f, ok := guardFleetFromDoc(fleetpane.FleetDoc{Machines: machines})
	if !ok {
		t.Fatal("ok=false, want true")
	}
	if f.Machines != guardFleetMachineRows+5 {
		t.Fatalf("Machines = %d, want %d (whole fleet)", f.Machines, guardFleetMachineRows+5)
	}
	if len(f.Rows) != guardFleetMachineRows {
		t.Fatalf("rows = %d, want cap %d", len(f.Rows), guardFleetMachineRows)
	}
}

// TestGuardFleetProviderTTLCache proves a fold is reused within the TTL window (the fleet
// view runs once) and re-run only after the window elapses, so a per-scrape /debug/vars
// poll does not re-walk the machine dir on every tick.
func TestGuardFleetProviderTTLCache(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)
	now := base
	calls := 0
	p := &guardFleetProvider{
		now: func() time.Time { return now },
		diskDoc: func() (fleetpane.FleetDoc, bool) {
			calls++
			machines := make([]map[string]any, calls)
			for i := range machines {
				machines[i] = map[string]any{"id": string(rune('a' + i)), "state": "OK"}
			}
			return fleetpane.FleetDoc{Machines: machines}, true
		},
	}

	// First pull folds once.
	if f, ok := p.pull(); !ok || f.Machines != 1 || calls != 1 {
		t.Fatalf("pull#1 = (%+v, %v), calls=%d, want machines=1 ok calls=1", f, ok, calls)
	}
	// A pull inside the TTL window reuses the fold — no new view call.
	now = base.Add(guardFleetTTL - time.Second)
	if f, ok := p.pull(); !ok || f.Machines != 1 || calls != 1 {
		t.Fatalf("pull#2 (in-window) = (%+v, %v), calls=%d, want cached machines=1 calls=1", f, ok, calls)
	}
	// Past the window, the next pull re-folds.
	now = base.Add(guardFleetTTL + time.Second)
	if f, ok := p.pull(); !ok || f.Machines != 2 || calls != 2 {
		t.Fatalf("pull#3 (post-window) = (%+v, %v), calls=%d, want refold machines=2 calls=2", f, ok, calls)
	}
}

// TestGuardFleetProviderCachesEmptyFold proves an empty fold (ok=false) is also cached for
// the TTL window, so a machine with no peers does not re-walk the (empty) dir every scrape.
func TestGuardFleetProviderCachesEmptyFold(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)
	now := base
	calls := 0
	p := &guardFleetProvider{
		now: func() time.Time { return now },
		diskDoc: func() (fleetpane.FleetDoc, bool) {
			calls++
			return fleetpane.FleetDoc{}, false
		},
	}
	if _, ok := p.pull(); ok || calls != 1 {
		t.Fatalf("pull#1 ok=%v calls=%d, want ok=false calls=1", ok, calls)
	}
	now = base.Add(guardFleetTTL / 2)
	if _, ok := p.pull(); ok || calls != 1 {
		t.Fatalf("pull#2 (in-window) ok=%v calls=%d, want cached ok=false calls=1", ok, calls)
	}
}

// spineMachine builds a live-peer machine map in the shape MachineMaps / the fold produce.
func spineMachine(id, state, generated string, sessions int) map[string]any {
	return map[string]any{
		"id": id, "state": state, "generated_utc": generated,
		"sessions": sessions, "app_version": "vspine", "source": "spine",
	}
}

// TestFoldViewUnionsDiskAndSpine proves the provider merges disk snapshots with live spine
// peers: a machine present only on disk and one present only on the network both appear.
func TestFoldViewUnionsDiskAndSpine(t *testing.T) {
	disk := fleetpane.FleetDoc{
		Verdict:  "OK",
		States:   map[string]int{"OK": 1},
		Totals:   map[string]int{"sessions": 5},
		Machines: []map[string]any{{"id": "alpha", "state": "OK", "sessions": float64(5)}},
	}
	p := &guardFleetProvider{
		diskDoc: func() (fleetpane.FleetDoc, bool) { return disk, true },
		spinePeers: func() []map[string]any {
			return []map[string]any{spineMachine("beta", "OK", "2026-07-01T12:00:00Z", 3)}
		},
	}
	f, ok := p.pull()
	if !ok {
		t.Fatal("ok=false, want true (disk ∪ spine has 2 machines)")
	}
	if f.Machines != 2 {
		t.Fatalf("Machines = %d, want 2 (alpha disk + beta spine)", f.Machines)
	}
	ids := map[string]bool{}
	for _, r := range f.Rows {
		ids[r.ID] = true
	}
	if !ids["alpha"] || !ids["beta"] {
		t.Fatalf("rows = %+v, want both alpha and beta", f.Rows)
	}
	if f.Sessions != 8 {
		t.Fatalf("Sessions = %d, want 8 (5 disk + 3 spine, recomputed)", f.Sessions)
	}
}

// TestFoldViewNewestWins proves a machine present on BOTH disk and the network dedupes to the
// fresher record (newest generated_utc), so a just-heard live heartbeat overrides a stale disk
// snapshot of the same box.
func TestFoldViewNewestWins(t *testing.T) {
	disk := fleetpane.FleetDoc{
		States:   map[string]int{"STALE": 1},
		Totals:   map[string]int{},
		Machines: []map[string]any{{"id": "alpha", "state": "STALE", "generated_utc": "2026-07-01T10:00:00Z", "sessions": float64(1)}},
	}
	p := &guardFleetProvider{
		diskDoc: func() (fleetpane.FleetDoc, bool) { return disk, true },
		spinePeers: func() []map[string]any {
			return []map[string]any{spineMachine("alpha", "OK", "2026-07-01T12:00:00Z", 4)}
		},
	}
	f, ok := p.pull()
	if !ok || f.Machines != 1 {
		t.Fatalf("pull = (%+v, %v), want 1 machine (deduped)", f, ok)
	}
	if f.Rows[0].State != "OK" || f.Rows[0].Sessions != 4 {
		t.Fatalf("deduped row = %+v, want the newer live record (OK, 4 sess)", f.Rows[0])
	}
	if f.Verdict != "OK" {
		t.Fatalf("Verdict = %q, want OK (recomputed after the newer record replaced STALE)", f.Verdict)
	}
}

// TestFoldViewSpineOnlyStillRenders is the point of the whole feature: a peer discovered over
// the network with NO disk snapshot (no shared git remote) still appears in the panel.
func TestFoldViewSpineOnlyStillRenders(t *testing.T) {
	p := &guardFleetProvider{
		diskDoc: func() (fleetpane.FleetDoc, bool) { return fleetpane.FleetDoc{}, false }, // no repo/config
		spinePeers: func() []map[string]any {
			return []map[string]any{spineMachine("cloudbox", "OK", "2026-07-01T12:00:00Z", 2)}
		},
	}
	f, ok := p.pull()
	if !ok {
		t.Fatal("ok=false, want true (a network-only peer must render)")
	}
	if f.Machines != 1 || f.Rows[0].ID != "cloudbox" {
		t.Fatalf("fleet = %+v, want 1 machine cloudbox", f)
	}
}

// TestFoldViewBothEmptyIsSilent proves the panel stays silent (block omitted) when neither
// discovery source has anything — the zero-cost / disk-only-degradation contract.
func TestFoldViewBothEmptyIsSilent(t *testing.T) {
	p := &guardFleetProvider{
		diskDoc:    func() (fleetpane.FleetDoc, bool) { return fleetpane.FleetDoc{}, false },
		spinePeers: func() []map[string]any { return nil },
	}
	if f, ok := p.pull(); ok {
		t.Fatalf("pull = (%+v, %v), want ok=false (both sources empty → silent)", f, ok)
	}
}

// TestInstallGuardFleetProviderFallsBackToDisk witnesses the production guard wiring seam.
// Disabling multicast leaves the disk provider installed; the provider may be empty in a
// scratch directory, but it is callable and does not make guard startup depend on a socket.
func TestInstallGuardFleetProviderFallsBackToDisk(t *testing.T) {
	t.Setenv("FLEET_SPINE_ENABLED", "false")
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatal(err)
	}
	installGuardFleetProvider(srv, context.Background(), nil)
	if !srv.SessionFleetProviderInstalled() {
		t.Fatal("guard fleet provider was not installed")
	}
	if _, ok := srv.SessionFleetSnapshot(); ok {
		// A repo-local machine snapshot is allowed; reaching the provider is the witness.
		return
	}
	// An empty snapshot is the expected disk-only fallback on a cold workspace. Prove the
	// provider was attached by replacing it with nil and observing the same public omission
	// contract without a panic; provider shape is covered by TestGuardFleetProviderTTLCache.
	srv.SetSessionFleetProvider(nil)
	if _, ok := srv.SessionFleetSnapshot(); ok {
		t.Fatal("fleet after detach ok=true, want omitted")
	}
}
