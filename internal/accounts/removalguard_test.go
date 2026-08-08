package accounts

import (
	"strings"
	"testing"
	"time"
)

// probeAt builds a SeatProbe with a dated row — the shape the ledger reader hands over.
func probeAt(status string, at time.Time) SeatProbe {
	return SeatProbe{Status: status, At: at, HasAt: true}
}

// TestGradeRemovalGatesTheBlipAndNotTheWall is #4676's Definition-of-Done test, both halves in
// one table: a seat whose last probe was OK (the july17-netra shape — every probe OK, one
// TRANSIENT session-level INFRA_ORG_DISABLED, tombstoned anyway) must be flagged, and a seat
// behind a real access wall ("Claude subscription access disabled" -> probe status ACCESS) must
// pass through with no gate at all.
func TestGradeRemovalGatesTheBlipAndNotTheWall(t *testing.T) {
	// Anchored on the issue's own evidence: last probe 2026-07-13 09:05Z, tombstone 15:45Z.
	now := time.Date(2026, 7, 13, 15, 45, 0, 0, time.UTC)
	lastProbe := time.Date(2026, 7, 13, 9, 5, 0, 0, time.UTC)

	cases := []struct {
		name    string
		probe   SeatProbe
		want    RemovalVerdict
		wantGap bool // expect Blip() — the removal is gated
	}{
		{"fresh OK is the blip the issue reports", probeAt("OK", lastProbe), RemovalHealthy, true},
		{"lower-case ok from a hand-written row still gates", probeAt("ok", lastProbe), RemovalHealthy, true},
		{"real access wall removes normally", probeAt("ACCESS", lastProbe), RemovalWalled, false},
		{"auth-required wall removes normally", probeAt("AUTH", lastProbe), RemovalWalled, false},
		{"billing wall removes normally", probeAt("CREDIT", lastProbe), RemovalWalled, false},
		{"usage wall is neither health nor entitlement", probeAt("LIMIT", lastProbe), RemovalUnknown, false},
		{"transport fault is not evidence of health", probeAt("TRANSPORT", lastProbe), RemovalUnknown, false},
		{"never probed never gates", SeatProbe{}, RemovalUnknown, false},
		{"undatable OK row never gates", SeatProbe{Status: "OK"}, RemovalUnknown, false},
		{"stale OK never gates", probeAt("OK", now.Add(-25*time.Hour)), RemovalUnknown, false},
		{"OK just inside the budget still gates", probeAt("OK", now.Add(-23*time.Hour)), RemovalHealthy, true},
		{"future-dated OK beyond the budget never gates", probeAt("OK", now.Add(25*time.Hour)), RemovalUnknown, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := GradeRemoval(tc.probe, now)
			if got.Verdict != tc.want {
				t.Fatalf("verdict = %q, want %q (reason: %s)", got.Verdict, tc.want, got.Reason)
			}
			if got.Blip() != tc.wantGap {
				t.Fatalf("Blip() = %t, want %t (reason: %s)", got.Blip(), tc.wantGap, got.Reason)
			}
			if strings.TrimSpace(got.Reason) == "" {
				t.Fatal("every grade must carry an operator-facing reason; got empty")
			}
		})
	}
}

// TestGradeRemovalCarriesTheMeasuredAge proves the grade reports the evidence it judged on, so
// a caller can print "last probe 6.7h ago was OK" rather than only the label.
func TestGradeRemovalCarriesTheMeasuredAge(t *testing.T) {
	now := time.Date(2026, 7, 13, 15, 45, 0, 0, time.UTC)
	got := GradeRemoval(probeAt("OK", now.Add(-400*time.Minute)), now)
	if !got.HasAge {
		t.Fatal("a dated row must yield HasAge=true")
	}
	if diff := got.AgeHours - 400.0/60.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("AgeHours = %v, want %v", got.AgeHours, 400.0/60.0)
	}
	if got.Status != "OK" {
		t.Fatalf("Status = %q, want OK", got.Status)
	}
	if !strings.Contains(got.Reason, "6.7h") {
		t.Fatalf("reason should name the measured age; got %q", got.Reason)
	}
}

// TestRestoreNoticeSurfacesTheDarkButHealthySeat replays the issue's own timeline: the seat is
// tombstoned at 15:45Z on a last probe of OK at 09:05Z, and 27h later — with zero probes since,
// exactly as reported — the status surface must still be offering the restore.
func TestRestoreNoticeSurfacesTheDarkButHealthySeat(t *testing.T) {
	tomb := time.Date(2026, 7, 13, 15, 45, 0, 0, time.UTC)
	lastProbe := time.Date(2026, 7, 13, 9, 5, 0, 0, time.UTC)
	h := Home{
		Name:            "july17-netra",
		Status:          StatusTombstoned,
		RehomeTo:        "july6-netra",
		TombstonedAt:    tomb.Format(time.RFC3339),
		TombstoneReason: "removed via `fak accounts remove`",
	}
	now := tomb.Add(27 * time.Hour)

	got, ok := RestoreNotice(h, probeAt("OK", lastProbe), now)
	if !ok {
		t.Fatal("a seat tombstoned 27h ago on an OK probe must still be offered for restore")
	}
	if got.Name != "july17-netra" || got.RehomeTo != "july6-netra" {
		t.Fatalf("notice must echo the registry handles; got %+v", got)
	}
	if got.ProbeStatus != "OK" {
		t.Fatalf("ProbeStatus = %q, want OK", got.ProbeStatus)
	}
	// The probe age is measured against the TOMBSTONE (6h40m), not against now (33h40m) — the
	// prober stopped at the tombstone, so measuring against now would age the notice out of
	// the very window it exists to cover.
	if diff := got.ProbeAgeHours - 400.0/60.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("ProbeAgeHours = %v, want the probe->tombstone gap %v", got.ProbeAgeHours, 400.0/60.0)
	}
	if diff := got.RemovedHoursAgo - 27; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("RemovedHoursAgo = %v, want 27", got.RemovedHoursAgo)
	}
	hint := got.Hint()
	for _, want := range []string{"REMOVED-BUT-HEALTHY", "july17-netra", "fak accounts restore"} {
		if !strings.Contains(hint, want) {
			t.Fatalf("hint %q missing %q", hint, want)
		}
	}
}

// TestRestoreNoticeStaysSilentWhenItWouldBeInventingEvidence pins every case where the notice
// must NOT fire — an active seat, a settled decision, a walled seat, and (the important one) a
// seat whose newest probe already predated its own removal by more than the freshness budget,
// where calling it "healthy" would assert evidence the ledger never carried.
func TestRestoreNoticeStaysSilentWhenItWouldBeInventingEvidence(t *testing.T) {
	tomb := time.Date(2026, 7, 13, 15, 45, 0, 0, time.UTC)
	stamp := tomb.Format(time.RFC3339)
	tombstoned := func(h Home) Home {
		h.Status = StatusTombstoned
		h.TombstonedAt = stamp
		return h
	}
	cases := []struct {
		name  string
		home  Home
		probe SeatProbe
		now   time.Time
	}{
		{"an active seat has nothing to restore", Home{Name: "a", Status: StatusActive}, probeAt("OK", tomb.Add(-time.Hour)), tomb.Add(time.Hour)},
		{"past the restore window the decision has settled", tombstoned(Home{Name: "b"}), probeAt("OK", tomb.Add(-time.Hour)), tomb.Add(49 * time.Hour)},
		{"a walled seat was removed for cause", tombstoned(Home{Name: "c"}), probeAt("ACCESS", tomb.Add(-time.Hour)), tomb.Add(time.Hour)},
		{"a probe older than the seat's own removal is not evidence", tombstoned(Home{Name: "d"}), probeAt("OK", tomb.Add(-30*time.Hour)), tomb.Add(time.Hour)},
		{"a never-probed seat is not healthy", tombstoned(Home{Name: "e"}), SeatProbe{}, tomb.Add(time.Hour)},
		{"an unparseable tombstone stamp cannot be windowed", Home{Name: "f", Status: StatusTombstoned, TombstonedAt: "last tuesday"}, probeAt("OK", tomb.Add(-time.Hour)), tomb.Add(time.Hour)},
		{"a missing tombstone stamp cannot be windowed", Home{Name: "g", Status: StatusTombstoned}, probeAt("OK", tomb.Add(-time.Hour)), tomb.Add(time.Hour)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got, ok := RestoreNotice(tc.home, tc.probe, tc.now); ok {
				t.Fatalf("expected no notice, got %+v", got)
			}
		})
	}
}

// TestRegistryRemovedButHealthyFoldsOnlyTheBlips proves the registry-wide fold picks exactly the
// removed-but-healthy seats, in registry order, and that a seat with no probe row at all is
// simply absent rather than optimistically included.
func TestRegistryRemovedButHealthyFoldsOnlyTheBlips(t *testing.T) {
	tomb := time.Date(2026, 7, 13, 15, 45, 0, 0, time.UTC)
	stamp := tomb.Format(time.RFC3339)
	reg := Registry{Homes: []Home{
		{Name: "live", Status: StatusActive},
		{Name: "blip", Status: StatusTombstoned, RehomeTo: "live", TombstonedAt: stamp},
		{Name: "walled", Status: StatusTombstoned, RehomeTo: "live", TombstonedAt: stamp},
		{Name: "unprobed", Status: StatusTombstoned, RehomeTo: "live", TombstonedAt: stamp},
		{Name: "blip2", Status: StatusTombstoned, RehomeTo: "live", TombstonedAt: stamp},
	}}
	probes := map[string]SeatProbe{
		"live":   probeAt("OK", tomb.Add(-time.Hour)),
		"blip":   probeAt("OK", tomb.Add(-time.Hour)),
		"walled": probeAt("ACCESS", tomb.Add(-time.Hour)),
		"blip2":  probeAt("OK", tomb.Add(-2*time.Hour)),
	}
	got := reg.RemovedButHealthy(probes, tomb.Add(3*time.Hour))
	if len(got) != 2 {
		t.Fatalf("want 2 removed-but-healthy seats, got %d: %+v", len(got), got)
	}
	if got[0].Name != "blip" || got[1].Name != "blip2" {
		t.Fatalf("fold must keep registry order; got %q, %q", got[0].Name, got[1].Name)
	}
	// A registry with no ledger at hand must yield nothing, never a false "all healthy".
	if empty := reg.RemovedButHealthy(nil, tomb.Add(3*time.Hour)); len(empty) != 0 {
		t.Fatalf("no probes must yield no notices; got %+v", empty)
	}
}
