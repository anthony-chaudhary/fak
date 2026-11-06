package executionroute

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestRouteRequiresHealthEvenWhenNoReadings preserves the explicit fail-closed
// policy when an empty fleet report supplies no candidate evidence at all.
// Estimated added wall-clock: 1 ms, unmeasured; four in-memory routing calls.
// fak-test:runtime fast est=1ms lane=default
func TestRouteRequiresHealthEvenWhenNoReadings(t *testing.T) {
	t.Parallel()
	emptyFleet := HealthFromFleetStatus(nil, 60)
	emptyFleet.RequireEvidence = true
	profiles := testProfiles(t)
	for _, tc := range []struct {
		name    string
		health  HealthReport
		wantErr bool
		reason  string
	}{
		{"optional empty", HealthReport{}, false, "first candidate satisfying requirements"},
		{"required empty fleet", emptyFleet, true, ""},
		{"required empty map", HealthReport{RequireEvidence: true, Candidates: map[string]HarnessHealth{}}, true, ""},
		{"required available", HealthReport{
			RequireEvidence: true,
			Candidates: map[string]HarnessHealth{
				"codex": {State: HealthAvailable, FreeSlots: 1, SessionCap: 1, AgeSeconds: 1},
			},
		}, false, "first candidate satisfying requirements and live health"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Route(Request{
				HarnessCandidates: []string{"codex", "claude"},
				Health:            tc.health,
				Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
			}, profiles, modelroute.DefaultManifest())
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "no harness candidate satisfies") {
					t.Fatalf("err = %v, want no eligible harness", err)
				}
				if !reflect.DeepEqual(got, Decision{}) {
					t.Fatalf("missing health evidence returned a routing decision: %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.Harness.Profile.Name != "codex" || got.Harness.Reason != tc.reason || len(got.Harness.Rejected) != 0 {
				t.Fatalf("eligible first candidate changed: %+v", got.Harness)
			}
			if got.Model.Plan.Primary() == "" || got.Session.Action != SessionStart {
				t.Fatalf("successful routing lost model or session decision: %+v", got)
			}
		})
	}
}
