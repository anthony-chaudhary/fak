package routing

import (
	"reflect"
	"testing"
)

// candidate builds a minimal routing candidate for a table case.
func candidate(id, lane string) *TaskCandidate {
	return &TaskCandidate{ID: id, Lane: lane}
}

// workerEngine builds a healthy local worker-capable engine.
func workerEngine(id string) Engine {
	return Engine{
		ID:               id,
		Surface:          SurfaceLocal,
		Model:            "test-model",
		Roles:            []string{"worker"},
		Healthy:          true,
		CapacityHeadroom: 0.5,
	}
}

// TestRouteTable is the table-driven witness for the core decision. Each case
// asserts the placed/unrouted split, the winning engine, and — where it
// matters — the closed rejection reason and the audit.
func TestRouteTable(t *testing.T) {
	cases := []struct {
		name           string
		requests       []TaskRequest
		engines        []Engine
		wantEngine     string // "" means unrouted
		wantSurface    Surface
		wantUnrouted   string // closed reason when unrouted, else ""
		wantRejected   []Rejection
		wantAudit      bool
		checkAssignOne func(t *testing.T, a Assignment)
	}{
		{
			name:       "happy path picks first eligible engine",
			requests:   []TaskRequest{{Candidate: candidate("T1", "docs"), Role: "worker"}},
			engines:    []Engine{workerEngine("e1"), workerEngine("e2")},
			wantEngine: "e1",
		},
		{
			name:       "unhealthy engine skipped with reason, next eligible wins",
			requests:   []TaskRequest{{Candidate: candidate("T2", "docs"), Role: "worker"}},
			engines:    []Engine{{ID: "sick", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: false}, workerEngine("e2")},
			wantEngine: "e2",
			wantRejected: []Rejection{
				{EngineID: "sick", Reason: ReasonUnhealthy},
			},
		},
		{
			name:       "draining engine skipped",
			requests:   []TaskRequest{{Candidate: candidate("T3", "docs"), Role: "worker"}},
			engines:    []Engine{{ID: "retiring", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, Draining: true, CapacityHeadroom: 1}, workerEngine("e2")},
			wantEngine: "e2",
			wantRejected: []Rejection{
				{EngineID: "retiring", Reason: ReasonDraining},
			},
		},
		{
			name:       "role mismatch skipped",
			requests:   []TaskRequest{{Candidate: candidate("T4", "docs"), Role: "judge"}},
			engines:    []Engine{workerEngine("worker-only"), {ID: "judge", Surface: SurfaceLocal, Roles: []string{"judge"}, Healthy: true, CapacityHeadroom: 1}},
			wantEngine: "judge",
			wantRejected: []Rejection{
				{EngineID: "worker-only", Reason: ReasonRoleMismatch},
			},
		},
		{
			name:       "missing tag skipped",
			requests:   []TaskRequest{{Candidate: candidate("T5", "docs"), Role: "worker", RequireTags: []string{"gpu"}}},
			engines:    []Engine{workerEngine("no-gpu"), {ID: "gpu", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Tags: []string{"gpu"}}},
			wantEngine: "gpu",
			wantRejected: []Rejection{
				{EngineID: "no-gpu", Reason: ReasonMissingTag},
			},
		},
		{
			name:       "surface excluded skipped",
			requests:   []TaskRequest{{Candidate: candidate("T6", "docs"), Role: "worker", ExcludeSurfaces: []Surface{SurfaceHosted}}},
			engines:    []Engine{{ID: "cloud", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1}, workerEngine("local")},
			wantEngine: "local",
			wantRejected: []Rejection{
				{EngineID: "cloud", Reason: ReasonSurfaceExcluded},
			},
		},
		{
			name:       "require-local never hosted",
			requests:   []TaskRequest{{Candidate: candidate("T7", "docs"), Role: "worker", RequireLocal: true}},
			engines:    []Engine{{ID: "cloud", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1}, workerEngine("local")},
			wantEngine: "local",
			wantRejected: []Rejection{
				{EngineID: "cloud", Reason: ReasonHostedRefused},
			},
		},
		{
			name:         "require-local with only hosted is unrouted",
			requests:     []TaskRequest{{Candidate: candidate("T8", "docs"), Role: "worker", RequireLocal: true}},
			engines:      []Engine{{ID: "cloud", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1}},
			wantEngine:   "",
			wantUnrouted: ReasonNoEligibleEngine,
			wantRejected: []Rejection{
				{EngineID: "cloud", Reason: ReasonHostedRefused},
			},
		},
		{
			name:         "no headroom engine skipped",
			requests:     []TaskRequest{{Candidate: candidate("T9", "docs"), Role: "worker"}},
			engines:      []Engine{{ID: "full", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 0}, workerEngine("free")},
			wantEngine:   "free",
			wantRejected: []Rejection{{EngineID: "full", Reason: ReasonNoHeadroom}},
		},
		{
			name:         "all engines no headroom is unrouted",
			requests:     []TaskRequest{{Candidate: candidate("T10", "docs"), Role: "worker"}},
			engines:      []Engine{{ID: "full", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 0}},
			wantEngine:   "",
			wantUnrouted: ReasonNoEligibleEngine,
			wantRejected: []Rejection{{EngineID: "full", Reason: ReasonNoHeadroom}},
		},
		{
			name:         "unknown role is unrouted with closed reason",
			requests:     []TaskRequest{{Candidate: candidate("T11", "docs"), Role: "wizard"}},
			engines:      []Engine{workerEngine("e1")},
			wantEngine:   "",
			wantUnrouted: ReasonUnknownRole,
		},
		{
			name:         "nil candidate is unrouted",
			requests:     []TaskRequest{{Role: "worker"}},
			engines:      []Engine{workerEngine("e1")},
			wantEngine:   "",
			wantUnrouted: ReasonUnknownEngine,
		},
		{
			name:         "unknown surface engine is skipped",
			requests:     []TaskRequest{{Candidate: candidate("T13", "docs"), Role: "worker"}},
			engines:      []Engine{{ID: "mystery", Surface: Surface("quantum"), Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1}, workerEngine("e2")},
			wantEngine:   "e2",
			wantRejected: []Rejection{{EngineID: "mystery", Reason: ReasonUnknownEngine}},
		},
		{
			name:       "empty roles engine serves any role",
			requests:   []TaskRequest{{Candidate: candidate("T14", "docs"), Role: "judge"}},
			engines:    []Engine{{ID: "generalist", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1}},
			wantEngine: "generalist",
		},
		{
			name:         "first eligible wins even when a later engine has higher priority",
			requests:     []TaskRequest{{Candidate: candidate("T15", "docs"), Role: "worker"}},
			engines:      []Engine{workerEngine("declared-first"), {ID: "higher-priority", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Priority: 99}},
			wantEngine:   "declared-first",
			wantSurface:  SurfaceLocal,
			wantRejected: nil,
		},
		{
			name:       "audit is populated on assignment",
			requests:   []TaskRequest{{Candidate: candidate("T17", "docs"), Role: "worker"}},
			engines:    []Engine{workerEngine("e1")},
			wantEngine: "e1",
			wantAudit:  true,
		},
		{
			name:         "reason is preserved for unrouted",
			requests:     []TaskRequest{{Candidate: candidate("T18", "docs"), Role: "worker"}},
			engines:      []Engine{{ID: "sick", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: false}},
			wantEngine:   "",
			wantUnrouted: ReasonNoEligibleEngine,
			wantRejected: []Rejection{{EngineID: "sick", Reason: ReasonUnhealthy}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assignments, unrouted, err := Route(tc.requests, tc.engines)
			if err != nil {
				t.Fatalf("Route() error = %v", err)
			}

			if tc.wantEngine == "" {
				if len(assignments) != 0 {
					t.Fatalf("got %d assignments, want 0 (unrouted)", len(assignments))
				}
				if len(unrouted) != 1 {
					t.Fatalf("got %d unrouted, want 1", len(unrouted))
				}
				if unrouted[0].Reason != tc.wantUnrouted {
					t.Errorf("unrouted reason = %q, want %q", unrouted[0].Reason, tc.wantUnrouted)
				}
				if !IsKnownReason(unrouted[0].Reason) {
					t.Errorf("unrouted reason %q is outside the closed vocabulary", unrouted[0].Reason)
				}
				if !reflect.DeepEqual(unrouted[0].Rejected, tc.wantRejected) {
					t.Errorf("unrouted rejected = %+v, want %+v", unrouted[0].Rejected, tc.wantRejected)
				}
				return
			}

			if len(unrouted) != 0 {
				t.Fatalf("got %d unrouted, want 0", len(unrouted))
			}
			if len(assignments) != 1 {
				t.Fatalf("got %d assignments, want 1", len(assignments))
			}
			a := assignments[0]
			if a.EngineID != tc.wantEngine {
				t.Errorf("engine = %q, want %q", a.EngineID, tc.wantEngine)
			}
			if !a.Surface.IsKnown() {
				t.Errorf("assigned surface %q is outside the closed vocabulary", a.Surface)
			}
			if tc.wantSurface != "" && a.Surface != tc.wantSurface {
				t.Errorf("surface = %q, want %q", a.Surface, tc.wantSurface)
			}
			if tc.wantRejected != nil && !reflect.DeepEqual(a.Rejected, tc.wantRejected) {
				t.Errorf("rejected = %+v, want %+v", a.Rejected, tc.wantRejected)
			}
			for _, r := range a.Rejected {
				if !IsKnownReason(r.Reason) {
					t.Errorf("rejection reason %q is outside the closed vocabulary", r.Reason)
				}
			}
			if tc.wantAudit && len(a.AssignmentAudit) == 0 {
				t.Error("AssignmentAudit is empty, want a populated audit trail")
			}
			if len(a.AssignmentAudit) == 0 {
				t.Error("AssignmentAudit is empty on every assignment")
			}
			if a.Schema != AssignmentSchema {
				t.Errorf("schema = %q, want %q", a.Schema, AssignmentSchema)
			}
			if tc.checkAssignOne != nil {
				tc.checkAssignOne(t, a)
			}
		})
	}
}

// TestRouteMultipleRequests proves each request places independently and a
// single call preserves request order.
func TestRouteMultipleRequests(t *testing.T) {
	requests := []TaskRequest{
		{Candidate: candidate("TA", "docs"), Role: "worker"},
		{Candidate: candidate("TB", "docs"), Role: "judge"},
	}
	engines := []Engine{
		workerEngine("w"),
		{ID: "j", Surface: SurfaceLocal, Roles: []string{"judge"}, Healthy: true, CapacityHeadroom: 1},
	}

	assignments, unrouted, err := Route(requests, engines)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if len(unrouted) != 0 {
		t.Fatalf("got %d unrouted, want 0", len(unrouted))
	}
	if len(assignments) != 2 {
		t.Fatalf("got %d assignments, want 2", len(assignments))
	}
	if assignments[0].TicketID != "TA" || assignments[0].EngineID != "w" {
		t.Errorf("first assignment = %+v, want TA -> w", assignments[0])
	}
	if assignments[1].TicketID != "TB" || assignments[1].EngineID != "j" {
		t.Errorf("second assignment = %+v, want TB -> j", assignments[1])
	}
}

// TestRouteDeterministic proves the same input yields byte-identical output,
// which is the fabric's defining property.
func TestRouteDeterministic(t *testing.T) {
	requests := []TaskRequest{
		{Candidate: candidate("T1", "docs"), Role: "worker", RequireTags: []string{"gpu"}},
		{Candidate: candidate("T2", "serving"), Role: "judge", RequireLocal: true},
	}
	engines := []Engine{
		{ID: "a", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Tags: []string{"gpu"}},
		{ID: "b", Surface: SurfaceLocal, Roles: []string{"worker", "judge"}, Healthy: true, CapacityHeadroom: 0.3, Tags: []string{"gpu"}},
		{ID: "c", Surface: SurfaceLocal, Roles: []string{"judge"}, Healthy: false},
	}

	firstA, firstU, err := Route(requests, engines)
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	for i := 0; i < 10; i++ {
		gotA, gotU, err := Route(requests, engines)
		if err != nil {
			t.Fatalf("Route() iteration %d error = %v", i, err)
		}
		if !reflect.DeepEqual(gotA, firstA) {
			t.Fatalf("iteration %d assignments differ:\n got %+v\nwant %+v", i, gotA, firstA)
		}
		if !reflect.DeepEqual(gotU, firstU) {
			t.Fatalf("iteration %d unrouted differ:\n got %+v\nwant %+v", i, gotU, firstU)
		}
	}
}

// TestRouteRejectsMalformedEngineSet proves a caller bug (duplicate/empty
// engine id) fails loud rather than silently mis-dispatching.
func TestRouteRejectsMalformedEngineSet(t *testing.T) {
	req := []TaskRequest{{Candidate: candidate("T1", "docs"), Role: "worker"}}

	if _, _, err := Route(req, []Engine{workerEngine("dup"), workerEngine("dup")}); err == nil {
		t.Error("duplicate engine id should fail closed, got nil error")
	}
	if _, _, err := Route(req, []Engine{{ID: "", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1}}); err == nil {
		t.Error("empty engine id should fail closed, got nil error")
	}
}

// TestClosedVocabulariesAreClosed is the registry witness: the reason, surface,
// and role vocabularies stay closed and every token is unique.
func TestClosedVocabulariesAreClosed(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range Reasons() {
		if !IsKnownReason(r) {
			t.Errorf("Reasons() contains %q but IsKnownReason is false", r)
		}
		if seen[r] {
			t.Errorf("duplicate reason %q", r)
		}
		seen[r] = true
	}
	if IsKnownReason("NOT_A_REASON") {
		t.Error("IsKnownReason accepted an unknown token")
	}

	surfSeen := map[Surface]bool{}
	for _, s := range Surfaces() {
		if !s.IsKnown() {
			t.Errorf("Surfaces() contains %q but IsKnown is false", s)
		}
		if surfSeen[s] {
			t.Errorf("duplicate surface %q", s)
		}
		surfSeen[s] = true
	}
	if Surface("quantum").IsKnown() {
		t.Error("unknown surface reported as known")
	}

	roleSeen := map[string]bool{}
	for _, r := range KnownRoles() {
		if !IsKnownRole(r) {
			t.Errorf("KnownRoles() contains %q but IsKnownRole is false", r)
		}
		if roleSeen[r] {
			t.Errorf("duplicate role %q", r)
		}
		roleSeen[r] = true
	}
	if IsKnownRole("wizard") {
		t.Error("unknown role reported as known")
	}
}

// TestEngineHelpers covers the pure engine predicates directly.
func TestEngineHelpers(t *testing.T) {
	anyRole := Engine{}
	if !anyRole.HasRole("judge") {
		t.Error("engine with no roles should serve any role")
	}
	specific := Engine{Roles: []string{"worker"}}
	if !specific.HasRole("worker") {
		t.Error("engine should serve its declared role")
	}
	if specific.HasRole("judge") {
		t.Error("engine should not serve an undeclared role")
	}

	if !(Engine{}).HasTag(nil) {
		t.Error("empty tag requirement should be satisfied")
	}
	tagged := Engine{Tags: []string{"gpu", "big"}}
	if !tagged.HasTag([]string{"gpu"}) {
		t.Error("engine should satisfy a subset tag requirement")
	}
	if tagged.HasTag([]string{"gpu", "tpu"}) {
		t.Error("engine should not satisfy a missing tag requirement")
	}

	if !SurfaceHosted.Hosted() {
		t.Error("SurfaceHosted.Hosted() = false, want true")
	}
	if SurfaceLocal.Hosted() {
		t.Error("SurfaceLocal.Hosted() = true, want false")
	}
}
