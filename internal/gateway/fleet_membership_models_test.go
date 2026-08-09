package gateway

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// Issue #5635. Before WorkerSpec carried Models, the replica set was homogeneous
// BY CONSTRUCTION: one model string described the whole fleet, so the registry
// could route AROUND a worker (unhealthy, draining) but never TO one because of
// which model it holds. A registry holding a GLM worker and a Qwen worker would
// hand a glm-4.6 request to the Qwen worker on the very next round-robin beat and
// then report the upstream's rejection as a health problem.
//
// These tests pin the three things that fixes it: the partition, the empty-Models
// compatibility arm, and the fact that "no worker holds this model" is a DIFFERENT
// typed verdict from "no worker is healthy".

// twoModelFleet is the heterogeneous roster the issue is about: one worker holding
// GLM, one holding Qwen, both probed healthy.
func twoModelFleet(t *testing.T) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	mustAdd(t, m, WorkerSpec{ID: "glm-w", Endpoint: "h1:8000", Models: []string{"glm-4.6"}})
	mustAdd(t, m, WorkerSpec{ID: "qwen-w", Endpoint: "h2:8000", Models: []string{"qwen3-32b"}})
	m.ProbeOnce(context.Background())
	return m
}

// legacyFleet is a pre-labeling roster: every worker registered exactly the way
// every caller registered one before this field existed (no Models at all).
func legacyFleet(t *testing.T, ids ...string) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range ids {
		mustAdd(t, m, WorkerSpec{ID: id, Endpoint: id})
	}
	m.ProbeOnce(context.Background())
	return m
}

// TestFleetMembershipFiltersByModel is the issue's named witness: the two-model
// partition, the empty-Models compatibility arm, and the distinct typed error.
func TestFleetMembershipFiltersByModel(t *testing.T) {
	t.Run("partition", func(t *testing.T) {
		m := twoModelFleet(t)
		// Far more picks than the rotation period, so a model-blind round-robin
		// could not possibly stay on the right worker by luck.
		for i := 0; i < 8; i++ {
			got, err := m.PickForModel("glm-4.6")
			if err != nil {
				t.Fatalf("PickForModel(glm-4.6) #%d: %v", i, err)
			}
			if got.ID != "glm-w" {
				t.Fatalf("pick #%d for glm-4.6 landed on %q (holds %v), want glm-w", i, got.ID, got.Models)
			}
			got, err = m.PickForModel("qwen3-32b")
			if err != nil {
				t.Fatalf("PickForModel(qwen3-32b) #%d: %v", i, err)
			}
			if got.ID != "qwen-w" {
				t.Fatalf("pick #%d for qwen3-32b landed on %q (holds %v), want qwen-w", i, got.ID, got.Models)
			}
		}
		// The candidate view agrees with the pick.
		for _, tc := range []struct{ model, want string }{{"glm-4.6", "glm-w"}, {"qwen3-32b", "qwen-w"}} {
			cands, err := m.CandidatesForModel(tc.model)
			if err != nil {
				t.Fatalf("CandidatesForModel(%s): %v", tc.model, err)
			}
			if len(cands) != 1 || cands[0].ID != tc.want {
				t.Fatalf("CandidatesForModel(%s) = %+v, want exactly %s", tc.model, cands, tc.want)
			}
		}
	})

	t.Run("empty Models is unconstrained not empty", func(t *testing.T) {
		m := NewFleetMembership(MembershipConfig{
			HealthyAfter: 1, UnhealthyAfter: 2,
			Probe: func(context.Context, WorkerSpec) bool { return true },
		})
		mustAdd(t, m, WorkerSpec{ID: "glm-w", Models: []string{"glm-4.6"}})
		mustAdd(t, m, WorkerSpec{ID: "any-w"}) // the pre-labeling shape
		m.ProbeOnce(context.Background())

		// The unlabeled worker is the ONLY candidate for a model the labeled one
		// does not hold — "empty" must read as "serves everything", never as
		// "serves nothing" (a fail-closed reading here silently empties every
		// deployed configuration).
		for i := 0; i < 4; i++ {
			got, err := m.PickForModel("qwen3-32b")
			if err != nil {
				t.Fatalf("PickForModel(qwen3-32b) #%d: %v", i, err)
			}
			if got.ID != "any-w" {
				t.Fatalf("pick #%d for qwen3-32b = %q, want the unconstrained any-w", i, got.ID)
			}
		}
		// And it is ALSO a candidate for the labeled worker's model.
		cands, err := m.CandidatesForModel("glm-4.6")
		if err != nil {
			t.Fatalf("CandidatesForModel(glm-4.6): %v", err)
		}
		var ids []string
		for _, c := range cands {
			ids = append(ids, c.ID)
		}
		if !reflect.DeepEqual(ids, []string{"glm-w", "any-w"}) {
			t.Fatalf("CandidatesForModel(glm-4.6) = %v, want both glm-w and any-w", ids)
		}
	})

	t.Run("unheld model is a distinct typed error", func(t *testing.T) {
		m := twoModelFleet(t)
		_, err := m.PickForModel("llama-3.1-70b")
		if !errors.Is(err, ErrNoWorkerForModel) {
			t.Fatalf("PickForModel(unheld) err = %v, want ErrNoWorkerForModel", err)
		}
		// The whole point of the second error: a configuration mistake must not
		// masquerade as an outage. If ErrNoWorkerForModel were aliased onto (or
		// wrapped in) ErrNoHealthyWorker this assertion is what catches it.
		if errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("ErrNoWorkerForModel is aliased onto ErrNoHealthyWorker: %v", err)
		}
		// ...and not the other way round either.
		if errors.Is(ErrNoHealthyWorker, ErrNoWorkerForModel) {
			t.Fatal("ErrNoHealthyWorker matches ErrNoWorkerForModel; the two verdicts are not distinct")
		}
	})
}

// TestFleetMembershipModelFilterPrecedesHealthFilter proves the ORDERING the design
// turns on, in both directions: a holder that is down is an OUTAGE (and never falls
// through to a healthy worker holding something else), while a model nobody holds is
// a CONFIGURATION fault even when the fleet is otherwise perfectly healthy.
func TestFleetMembershipModelFilterPrecedesHealthFilter(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "glm-w", Models: []string{"glm-4.6"}})
	mustAdd(t, m, WorkerSpec{ID: "qwen-w", Models: []string{"qwen3-32b"}})
	ctx := context.Background()
	m.ProbeOnce(ctx) // both healthy

	sp.set("glm-w", false)
	m.ProbeOnce(ctx) // the only glm holder is now unhealthy

	_, err := m.PickForModel("glm-4.6")
	if !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("holder down: PickForModel(glm-4.6) err = %v, want ErrNoHealthyWorker", err)
	}
	if errors.Is(err, ErrNoWorkerForModel) {
		t.Fatalf("an outage was reported as a configuration fault: %v", err)
	}
	// It must be a verdict, not a fall-through onto the healthy Qwen worker.
	if spec, err := m.PickForModel("glm-4.6"); err == nil {
		t.Fatalf("placement fell through to %q for a model it does not hold", spec.ID)
	}

	// Meanwhile qwen-w is healthy, so the fleet is emphatically not down — yet a
	// model nobody holds still reports as configuration, not health.
	if got, err := m.PickForModel("qwen3-32b"); err != nil || got.ID != "qwen-w" {
		t.Fatalf("PickForModel(qwen3-32b) = %q, %v; want qwen-w, nil", got.ID, err)
	}
	_, err = m.PickForModel("llama-3.1-70b")
	if !errors.Is(err, ErrNoWorkerForModel) || errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("unheld model on a healthy fleet: err = %v, want ErrNoWorkerForModel only", err)
	}

	// An EMPTY roster keeps the pre-existing verdict: with nothing registered there
	// is no evidence of a model mismatch, so it stays an outage.
	empty := NewFleetMembership(MembershipConfig{})
	if _, err := empty.PickForModel("glm-4.6"); !errors.Is(err, ErrNoHealthyWorker) || errors.Is(err, ErrNoWorkerForModel) {
		t.Fatalf("empty roster: err = %v, want ErrNoHealthyWorker only", err)
	}
}

// TestFleetMembershipEmptyModelsMatchesPreLabelingBehavior is the backward-compat
// arm, and it is load-bearing: an unlabeled roster must route through the NEW code
// exactly as it did through the old. It pins the rotation against a hard-coded
// sequence AND proves the three entry points (Pick, PickForModel(""), and
// PickForModel of an arbitrary model over an unlabeled fleet) produce that same
// sequence, so no path drifted.
func TestFleetMembershipEmptyModelsMatchesPreLabelingBehavior(t *testing.T) {
	const picks = 7
	// The classic round-robin an unlabeled three-worker fleet has always produced.
	want := []string{"w1", "w2", "w3", "w1", "w2", "w3", "w1"}

	viaPick := make([]string, 0, picks)
	m1 := legacyFleet(t, "w1", "w2", "w3")
	for i := 0; i < picks; i++ {
		spec, ok := m1.Pick()
		if !ok {
			t.Fatalf("Pick() #%d reported no admissible worker on a healthy fleet", i)
		}
		viaPick = append(viaPick, spec.ID)
	}
	if !reflect.DeepEqual(viaPick, want) {
		t.Fatalf("Pick() rotation = %v, want %v", viaPick, want)
	}

	viaEmptyModel := make([]string, 0, picks)
	m2 := legacyFleet(t, "w1", "w2", "w3")
	for i := 0; i < picks; i++ {
		spec, err := m2.PickForModel("")
		if err != nil {
			t.Fatalf("PickForModel(\"\") #%d: %v", i, err)
		}
		viaEmptyModel = append(viaEmptyModel, spec.ID)
	}
	if !reflect.DeepEqual(viaEmptyModel, want) {
		t.Fatalf("PickForModel(\"\") rotation = %v, want %v (identical to Pick)", viaEmptyModel, want)
	}

	viaAnyModel := make([]string, 0, picks)
	m3 := legacyFleet(t, "w1", "w2", "w3")
	for i := 0; i < picks; i++ {
		spec, err := m3.PickForModel("some-model-nobody-labeled")
		if err != nil {
			t.Fatalf("PickForModel(any) #%d over an unlabeled fleet: %v", i, err)
		}
		viaAnyModel = append(viaAnyModel, spec.ID)
	}
	if !reflect.DeepEqual(viaAnyModel, want) {
		t.Fatalf("unlabeled fleet, model query rotation = %v, want %v", viaAnyModel, want)
	}

	// Admissible() is untouched by the new field.
	if got := admissibleIDs(m1); len(got) != 3 || !got["w1"] || !got["w2"] || !got["w3"] {
		t.Fatalf("Admissible() = %v, want all three workers", got)
	}
	// Snapshot carries the (nil) Models through without inventing a value.
	for _, st := range m1.Snapshot() {
		if st.Spec.Models != nil {
			t.Fatalf("unlabeled worker %q Snapshot Models = %v, want nil", st.Spec.ID, st.Spec.Models)
		}
	}

	// Dispatch on an empty roster keeps the pre-existing typed verdict verbatim.
	empty := NewFleetMembership(MembershipConfig{})
	if _, err := empty.Dispatch(context.Background(), func(context.Context, WorkerSpec) error { return nil }); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("empty roster Dispatch err = %v, want ErrNoHealthyWorker", err)
	}
	if _, ok := empty.Pick(); ok {
		t.Fatal("empty roster Pick() reported ok")
	}
}

// TestFleetMembershipAddNormalizesModels covers the two ways a caller's slice can
// betray the registry: retained aliasing, and blank labels that must fail OPEN.
func TestFleetMembershipAddNormalizesModels(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter: 1, UnhealthyAfter: 2,
		Probe: func(context.Context, WorkerSpec) bool { return true },
	})
	caller := []string{" glm-4.6 "} // padded, and the caller keeps the buffer
	mustAdd(t, m, WorkerSpec{ID: "glm-w", Models: caller})
	mustAdd(t, m, WorkerSpec{ID: "blank-w", Models: []string{"", "   "}})
	caller[0] = "qwen3-32b" // caller reuses its own backing array afterwards
	m.ProbeOnce(ctx)

	// The label was trimmed at the door, and the caller's later mutation did not
	// reach the registry.
	if got, err := m.PickForModel("glm-4.6"); err != nil || got.ID != "glm-w" {
		t.Fatalf("PickForModel(glm-4.6) = %q, %v; want glm-w, nil", got.ID, err)
	}
	// blank-w normalized to unconstrained, so it — and only it — serves qwen.
	for i := 0; i < 3; i++ {
		got, err := m.PickForModel("qwen3-32b")
		if err != nil {
			t.Fatalf("PickForModel(qwen3-32b) #%d: %v", i, err)
		}
		if got.ID != "blank-w" {
			t.Fatalf("pick #%d = %q; a blank label must normalize to UNCONSTRAINED and a caller mutation must not leak in", i, got.ID)
		}
	}
	for _, st := range m.Snapshot() {
		if st.Spec.ID == "blank-w" && st.Spec.Models != nil {
			t.Fatalf("blank-w Models = %v, want nil (unconstrained)", st.Spec.Models)
		}
		if st.Spec.ID == "glm-w" && !reflect.DeepEqual(st.Spec.Models, []string{"glm-4.6"}) {
			t.Fatalf("glm-w Models = %v, want [glm-4.6]", st.Spec.Models)
		}
	}
}

// TestFleetMembershipDispatchForModelNeverDialsAWrongUpstream closes the done
// condition's last clause: an unheld model returns the typed verdict without
// dialing anything, and failover re-places only onto OTHER holders of the same
// model.
func TestFleetMembershipDispatchForModelNeverDialsAWrongUpstream(t *testing.T) {
	ctx := context.Background()

	t.Run("unheld model dials nothing", func(t *testing.T) {
		m := twoModelFleet(t)
		dialed := 0
		_, err := m.DispatchForModel(ctx, "llama-3.1-70b", func(context.Context, WorkerSpec) error {
			dialed++
			return nil
		})
		if dialed != 0 {
			t.Fatalf("dialed %d upstream(s) for a model no worker holds; want 0", dialed)
		}
		if !errors.Is(err, ErrNoWorkerForModel) || errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("DispatchForModel(unheld) err = %v, want ErrNoWorkerForModel only", err)
		}
	})

	t.Run("failover stays inside the model", func(t *testing.T) {
		m := NewFleetMembership(MembershipConfig{
			HealthyAfter: 1, UnhealthyAfter: 5,
			Probe: func(context.Context, WorkerSpec) bool { return true },
		})
		mustAdd(t, m, WorkerSpec{ID: "glm-a", Models: []string{"glm-4.6"}})
		mustAdd(t, m, WorkerSpec{ID: "qwen-w", Models: []string{"qwen3-32b"}})
		mustAdd(t, m, WorkerSpec{ID: "glm-b", Models: []string{"glm-4.6"}})
		m.ProbeOnce(ctx)

		var visited []string
		spec, err := m.DispatchForModel(ctx, "glm-4.6", func(_ context.Context, s WorkerSpec) error {
			visited = append(visited, s.ID)
			if s.ID == "glm-a" {
				return errors.New("upstream reset")
			}
			return nil
		})
		if err != nil {
			t.Fatalf("DispatchForModel(glm-4.6): %v", err)
		}
		if spec.ID != "glm-b" {
			t.Fatalf("failover served %q, want glm-b", spec.ID)
		}
		if !reflect.DeepEqual(visited, []string{"glm-a", "glm-b"}) {
			t.Fatalf("failover visited %v; it must stay inside the model and never touch qwen-w", visited)
		}
	})

	t.Run("all holders failing is still an outage", func(t *testing.T) {
		m := NewFleetMembership(MembershipConfig{
			HealthyAfter: 1, UnhealthyAfter: 5,
			Probe: func(context.Context, WorkerSpec) bool { return true },
		})
		mustAdd(t, m, WorkerSpec{ID: "glm-a", Models: []string{"glm-4.6"}})
		mustAdd(t, m, WorkerSpec{ID: "qwen-w", Models: []string{"qwen3-32b"}})
		m.ProbeOnce(ctx)
		var visited []string
		_, err := m.DispatchForModel(ctx, "glm-4.6", func(_ context.Context, s WorkerSpec) error {
			visited = append(visited, s.ID)
			return errors.New("upstream reset")
		})
		if !errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("every holder failing: err = %v, want ErrNoHealthyWorker", err)
		}
		if errors.Is(err, ErrNoWorkerForModel) {
			t.Fatalf("a real outage was typed as a configuration fault: %v", err)
		}
		if !reflect.DeepEqual(visited, []string{"glm-a"}) {
			t.Fatalf("visited %v; failover must not spill onto the qwen worker", visited)
		}
	})
}
