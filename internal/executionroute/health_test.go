package executionroute

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestRouteExcludesUnhealthyFirstChoice is the core witness: a deterministic
// fixture routes AROUND an unhealthy first choice to the next healthy candidate,
// and records why the first was skipped.
func TestRouteExcludesUnhealthyFirstChoice(t *testing.T) {
	health := HealthReport{Candidates: map[string]HarnessHealth{
		"codex":  {State: HealthCooldown, Detail: "usage limit; resets 11pm", Source: "registry", AgeSeconds: 30},
		"claude": {State: HealthAvailable, FreeSlots: 2, SessionCap: 4, Source: "probe", AgeSeconds: 5},
	}}
	dec, err := Route(Request{
		HarnessCandidates: []string{"codex", "claude"},
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "claude" {
		t.Fatalf("harness=%q want claude (routed around unhealthy codex)", got)
	}
	if len(dec.Harness.Rejected) != 1 || dec.Harness.Rejected[0].Candidate != "codex" {
		t.Fatalf("rejected=%+v want one entry for codex", dec.Harness.Rejected)
	}
	if !strings.Contains(dec.Harness.Rejected[0].Reason, "cooldown") {
		t.Fatalf("codex reason=%q want a cooldown reason", dec.Harness.Rejected[0].Reason)
	}
}

// TestRouteHealthPreservesOperatorOrder proves that among equally eligible
// (healthy, requirement-satisfying) candidates the earliest operator candidate
// still wins — health never reorders equals.
func TestRouteHealthPreservesOperatorOrder(t *testing.T) {
	both := HealthReport{Candidates: map[string]HarnessHealth{
		"codex":  {State: HealthAvailable, FreeSlots: 1, SessionCap: 2, Source: "registry", AgeSeconds: 1},
		"claude": {State: HealthAvailable, FreeSlots: 1, SessionCap: 2, Source: "registry", AgeSeconds: 1},
	}}
	for _, tc := range []struct {
		order []string
		want  string
	}{
		{[]string{"claude", "codex"}, "claude"},
		{[]string{"codex", "claude"}, "codex"},
	} {
		dec, err := Route(Request{
			HarnessCandidates: tc.order,
			Health:            both,
			Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
		}, testProfiles(t), modelroute.DefaultManifest())
		if err != nil {
			t.Fatalf("order %v: %v", tc.order, err)
		}
		if got := dec.Harness.Profile.Name; got != tc.want {
			t.Fatalf("order %v -> harness=%q want %q", tc.order, got, tc.want)
		}
		if len(dec.Harness.Rejected) != 0 {
			t.Fatalf("order %v: rejected=%+v want none", tc.order, dec.Harness.Rejected)
		}
	}
}

// TestRouteHealthReasonForEverySkipped proves EVERY skipped candidate carries a
// non-empty reason, in operator order, across all three exclusion classes.
func TestRouteHealthReasonForEverySkipped(t *testing.T) {
	health := HealthReport{Candidates: map[string]HarnessHealth{
		"codex":    {State: HealthUnavailable, Detail: "auth expired", Source: "probe", AgeSeconds: 10},
		"opencode": {State: HealthDraining, Source: "registry", AgeSeconds: 4},
		"claude":   {State: HealthAvailable, FreeSlots: 3, SessionCap: 4, Source: "registry", AgeSeconds: 2},
	}}
	dec, err := Route(Request{
		HarnessCandidates: []string{"codex", "openai-generic", "claude"},
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "claude" {
		t.Fatalf("harness=%q want claude", got)
	}
	if len(dec.Harness.Rejected) != 2 {
		t.Fatalf("rejected=%+v want 2", dec.Harness.Rejected)
	}
	if dec.Harness.Rejected[0].Candidate != "codex" || dec.Harness.Rejected[1].Candidate != "openai-generic" {
		t.Fatalf("rejected order=%+v want [codex, openai-generic]", dec.Harness.Rejected)
	}
	for _, r := range dec.Harness.Rejected {
		if strings.TrimSpace(r.Reason) == "" {
			t.Fatalf("candidate %q has an empty rejection reason", r.Candidate)
		}
	}
}

// TestRouteHealthExcludesStaleReading proves the explicit freshness bound: a
// reading older than the bound is not trusted even when its state is available.
func TestRouteHealthExcludesStaleReading(t *testing.T) {
	health := HealthReport{
		MaxAgeSeconds: 60,
		Candidates: map[string]HarnessHealth{
			"codex":  {State: HealthAvailable, FreeSlots: 5, SessionCap: 5, Source: "registry", AgeSeconds: 120},
			"claude": {State: HealthAvailable, FreeSlots: 1, SessionCap: 2, Source: "probe", AgeSeconds: 3},
		},
	}
	dec, err := Route(Request{
		HarnessCandidates: []string{"codex", "claude"},
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "claude" {
		t.Fatalf("harness=%q want claude (codex reading is stale)", got)
	}
	if len(dec.Harness.Rejected) != 1 || !strings.Contains(dec.Harness.Rejected[0].Reason, "stale") {
		t.Fatalf("rejected=%+v want codex excluded as stale", dec.Harness.Rejected)
	}
}

// TestRouteHealthRequireEvidenceExcludesUnmeasured proves the fail-closed policy:
// with RequireEvidence a candidate that has NO reading is excluded.
func TestRouteHealthRequireEvidenceExcludesUnmeasured(t *testing.T) {
	health := HealthReport{
		RequireEvidence: true,
		Candidates: map[string]HarnessHealth{
			"claude": {State: HealthAvailable, FreeSlots: 1, SessionCap: 2, Source: "registry", AgeSeconds: 1},
		},
	}
	dec, err := Route(Request{
		HarnessCandidates: []string{"codex", "claude"},
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "claude" {
		t.Fatalf("harness=%q want claude (codex has no evidence)", got)
	}
	if len(dec.Harness.Rejected) != 1 || !strings.Contains(dec.Harness.Rejected[0].Reason, "no live health evidence") {
		t.Fatalf("rejected=%+v want codex excluded for missing evidence", dec.Harness.Rejected)
	}
}

// TestRouteEmptyHealthIsInert proves backward compatibility: an empty health
// report leaves selection on the static requirements alone (no exclusions).
func TestRouteEmptyHealthIsInert(t *testing.T) {
	dec, err := Route(Request{
		HarnessCandidates: []string{"codex", "claude"},
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "codex" {
		t.Fatalf("harness=%q want codex (first candidate, no health gate)", got)
	}
	if len(dec.Harness.Rejected) != 0 {
		t.Fatalf("rejected=%+v want none with an inert health report", dec.Harness.Rejected)
	}
}

// TestHealthFromFleetStatusPopulatesAndRoutes is the live-read adapter witness:
// one current fleet health source (the `fak fleet-accounts status` rows) populates
// the routing input, and that input routes around the unhealthy first choice.
func TestHealthFromFleetStatusPopulatesAndRoutes(t *testing.T) {
	age := func(m float64) *float64 { return &m }
	rows := []fleetaccounts.StatusAccount{
		{Account: ".claude-a", Tag: "a", Product: "claude", Kind: "worker", CapacityCounted: true,
			State: "usage", Reason: "usage limit; resets 11pm", Reset: "11pm", SessionCap: 3, FreeSlots: 0,
			StatusSource: "registry", RegistryAgeMin: age(2)},
		{Account: ".codex-a", Tag: "a", Product: "codex", Kind: "worker", CapacityCounted: true,
			State: "ready", SessionCap: 4, FreeSlots: 3,
			StatusSource: "probe", RegistryAgeMin: age(1)},
	}
	health := HealthFromFleetStatus(rows, 0)
	if h := health.Candidates["claude"]; h.State != HealthCooldown {
		t.Fatalf("claude health=%+v want cooldown from the usage seat", h)
	}
	if h := health.Candidates["codex"]; h.State != HealthAvailable || h.FreeSlots != 3 {
		t.Fatalf("codex health=%+v want available with 3 free slots", h)
	}
	dec, err := Route(Request{
		HarnessCandidates: []string{"claude", "codex"},
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, testProfiles(t), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Harness.Profile.Name; got != "codex" {
		t.Fatalf("harness=%q want codex (claude in cooldown per the fleet source)", got)
	}
	if len(dec.Harness.Rejected) != 1 || dec.Harness.Rejected[0].Candidate != "claude" {
		t.Fatalf("rejected=%+v want claude", dec.Harness.Rejected)
	}
}

// TestHealthFromFleetStatusFoldsSeatsToAvailable proves the many-seats-per-product
// fold: a product is available when ANY counted worker seat is ready, freshness is
// the freshest seat, provenance names every contributing source, and non-worker /
// shared-pool rows are ignored.
func TestHealthFromFleetStatusFoldsSeatsToAvailable(t *testing.T) {
	age := func(m float64) *float64 { return &m }
	rows := []fleetaccounts.StatusAccount{
		{Product: "claude", Kind: "worker", CapacityCounted: true, State: "auth", Reason: "login expired",
			SessionCap: 2, FreeSlots: 0, StatusSource: "registry", RegistryAgeMin: age(5)},
		{Product: "claude", Kind: "worker", CapacityCounted: true, State: "ready",
			SessionCap: 2, FreeSlots: 2, StatusSource: "probe", RegistryAgeMin: age(1)},
		{Product: "claude", Kind: "non-account", State: "non-account"},
		{Product: "codex", Kind: "worker", CapacityCounted: false, State: "shared-pool"},
	}
	health := HealthFromFleetStatus(rows, 0)
	h, ok := health.Candidates["claude"]
	if !ok || h.State != HealthAvailable {
		t.Fatalf("claude health=%+v ok=%v want available (one seat ready)", h, ok)
	}
	if h.FreeSlots != 2 {
		t.Fatalf("claude freeSlots=%d want 2", h.FreeSlots)
	}
	if h.AgeSeconds != 60 {
		t.Fatalf("claude age=%ds want 60 (freshest seat: min(5,1)min)", h.AgeSeconds)
	}
	if !strings.Contains(h.Source, "probe") || !strings.Contains(h.Source, "registry") {
		t.Fatalf("claude source=%q want both probe and registry named", h.Source)
	}
	if _, ok := health.Candidates["codex"]; ok {
		t.Fatalf("codex should have no reading (only a shared-pool row)")
	}
}

// testProfiles returns the built-in harness profiles, guarding the three the
// health fixtures name so a profile rename fails loudly instead of silently
// dropping a candidate to "unknown harness profile".
func testProfiles(t *testing.T) []harnessprofile.HarnessProfile {
	t.Helper()
	profiles := harnessprofile.Builtins()
	for _, want := range []string{"claude", "codex", "openai-generic"} {
		if _, ok := findProfile(want, profiles); !ok {
			t.Fatalf("built-in harness profile %q missing; fixtures need it", want)
		}
	}
	return profiles
}
