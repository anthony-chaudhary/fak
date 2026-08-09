package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// livenessProbe is a real network probe rather than the package's scriptedProbe: the
// issue's done condition is stated over two httptest endpoints, one answering and one
// not, so the states in the produced report are OBSERVED rather than injected.
func livenessProbe(ctx context.Context, spec WorkerSpec) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, spec.Endpoint, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// TestAProbedFleetProducesAServingReport is the producer half of the issue #5636 done
// condition: a FleetMembership probing two endpoints, one up and one down, yields a
// fak.modelroute.serving.v1 document whose entries MATCH the observed states.
//
// Before this test there was nothing in the tree that produced that schema at all —
// grepping it returned only _test.go fixtures and one doc sample, so the placement
// ladder's fleet rung was gated by a file a human remembered to refresh.
func TestAProbedFleetProducesAServingReport(t *testing.T) {
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer live.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	dead.Close() // closed: nothing answers here at all

	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: livenessProbe})
	if err := m.Add(WorkerSpec{ID: "worker-up", Endpoint: live.URL}); err != nil {
		t.Fatal(err)
	}
	if err := m.Add(WorkerSpec{ID: "worker-down", Endpoint: dead.URL}); err != nil {
		t.Fatal(err)
	}
	m.ProbeOnce(context.Background())

	now := time.Unix(1_700_000_000, 0)
	rep, err := ServingReportFromSnapshot(m.Snapshot(), ServingReportOptions{
		Now:    now,
		MaxAge: 90 * time.Second,
		Covers: []modelroute.PlacementZone{modelroute.ZoneFleet},
	})
	if err != nil {
		t.Fatalf("ServingReportFromSnapshot: %v", err)
	}
	if rep.Schema != modelroute.ServingReportSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, modelroute.ServingReportSchema)
	}
	if err := rep.Validate(); err != nil {
		t.Fatalf("produced report does not validate: %v", err)
	}
	if got := rep.Models["worker-up"].State; got != modelroute.ServingUp {
		t.Errorf("worker-up observed %q, want %q", got, modelroute.ServingUp)
	}
	if got := rep.Models["worker-down"].State; got != modelroute.ServingDown {
		t.Errorf("worker-down observed %q, want %q", got, modelroute.ServingDown)
	}
	if rep.AsOfUnix != now.Unix() {
		t.Errorf("as-of %d, want %d", rep.AsOfUnix, now.Unix())
	}
}

// TestAProducedReportAlwaysCarriesABoundAndAStamp is the anti-immortality rule. A
// report with no as-of stamp or no declared TTL is honored at any age forever, which
// is precisely how a producer that dies on a Sunday keeps a dead rung open. The
// producer must refuse to emit one rather than write a document that can never go
// stale.
func TestAProducedReportAlwaysCarriesABoundAndAStamp(t *testing.T) {
	snap := []WorkerStatus{{Spec: WorkerSpec{ID: "w1"}, Health: HealthHealthy}}
	if _, err := ServingReportFromSnapshot(snap, ServingReportOptions{Now: time.Unix(1, 0)}); err == nil {
		t.Error("a report was produced with no freshness bound; it would be honored at any age forever")
	}
	if _, err := ServingReportFromSnapshot(snap, ServingReportOptions{MaxAge: time.Minute}); err == nil {
		t.Error("a report was produced with no as-of stamp; nothing can measure its age")
	}
	// An EMPTY membership is the one case with nothing to go stale, so it is allowed
	// through as the honest zero report.
	if _, err := ServingReportFromSnapshot(nil, ServingReportOptions{Now: time.Unix(1, 0)}); err != nil {
		t.Errorf("an empty snapshot should produce the zero report, got %v", err)
	}
}

// TestDrainingReadsAsDownNotDegraded pins the one state mapping that is easy to get
// backwards. modelroute treats DEGRADED as "still serving, still takes work" — a host
// under load must not shed its tokens to a vendor. A DRAINING worker is the opposite:
// drain exists precisely to stop new work reaching it. Mapping drain to degraded would
// keep routing work at a worker that is being taken out of service.
func TestDrainingReadsAsDownNotDegraded(t *testing.T) {
	snap := []WorkerStatus{
		{Spec: WorkerSpec{ID: "healthy"}, Health: HealthHealthy},
		{Spec: WorkerSpec{ID: "draining"}, Health: HealthHealthy, Draining: true},
		{Spec: WorkerSpec{ID: "unprobed"}, Health: HealthUnknown},
		{Spec: WorkerSpec{ID: "sick"}, Health: HealthUnhealthy},
	}
	rep, err := ServingReportFromSnapshot(snap, ServingReportOptions{Now: time.Unix(100, 0), MaxAge: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]modelroute.ServingState{
		"healthy":  modelroute.ServingUp,
		"draining": modelroute.ServingDown,
		"unprobed": modelroute.ServingUnknown,
		"sick":     modelroute.ServingDown,
	}
	for id, wantState := range want {
		if got := rep.Models[id].State; got != wantState {
			t.Errorf("%s: observed %q, want %q", id, got, wantState)
		}
	}
	if rep.Models["draining"].State == modelroute.ServingDegraded {
		t.Error("a draining worker read as degraded, which still takes work — drain exists to stop exactly that")
	}
}

// TestAnUnprobedFleetIsNeverReportedHealthy is the fail-closed floor for the producer.
// A membership whose probe has not run yet knows nothing, and a report that called
// that "up" would manufacture the stale positive this whole issue exists to prevent.
func TestAnUnprobedFleetIsNeverReportedHealthy(t *testing.T) {
	m := NewFleetMembership(MembershipConfig{Probe: nil})
	if err := m.Add(WorkerSpec{ID: "never-probed", Endpoint: "http://127.0.0.1:1/v1"}); err != nil {
		t.Fatal(err)
	}
	rep, err := ServingReportFromSnapshot(m.Snapshot(), ServingReportOptions{Now: time.Unix(100, 0), MaxAge: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if got := rep.Models["never-probed"].State; got != modelroute.ServingUnknown {
		t.Errorf("an unprobed worker observed %q, want %q", got, modelroute.ServingUnknown)
	}
}

// TestTheProducerDoesNotClaimCoverageItCannotHave guards the one field that turns
// SILENCE into a gate. Declaring ZoneFleet asserts "this membership is the whole fleet
// rung", which a single gateway process cannot know — another host's workers are
// invisible to it. Claiming it by default would pass over every candidate this
// instance happens not to serve.
func TestTheProducerDoesNotClaimCoverageItCannotHave(t *testing.T) {
	rep, err := ServingReportFromSnapshot(
		[]WorkerStatus{{Spec: WorkerSpec{ID: "w1"}, Health: HealthHealthy}},
		ServingReportOptions{Now: time.Unix(100, 0), MaxAge: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Covers) != 0 {
		t.Errorf("the producer claimed coverage %v nobody declared; silence on those rungs now gates candidates this instance never observed", rep.Covers)
	}
}

// TestTheProducedReportGoesStaleOnItsOwnBound closes the loop with the consumer rule.
// A produced report is not more trustworthy than a hand-written one — it is a claim
// with a timestamp, and the same TTL applies. Read one tick past its own declared
// bound it must degrade to unknown.
func TestTheProducedReportGoesStaleOnItsOwnBound(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	rep, err := ServingReportFromSnapshot(
		[]WorkerStatus{{Spec: WorkerSpec{ID: "w1"}, Health: HealthHealthy}},
		ServingReportOptions{Now: now, MaxAge: 60 * time.Second, Covers: []modelroute.PlacementZone{modelroute.ZoneFleet}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if rep.StaleAsOf(now.Unix() + 30) {
		t.Error("a 30s-old report read as stale under a 60s bound")
	}
	if !rep.StaleAsOf(now.Unix() + 61) {
		t.Fatal("a report past its OWN declared bound did not read as stale; a producer that dies leaves an immortal positive")
	}
	if got := rep.DegradeStale(now.Unix() + 61).Models["w1"].State; got != modelroute.ServingUnknown {
		t.Errorf("stale produced report degraded to %q, want %q", got, modelroute.ServingUnknown)
	}
}

// TestWriteServingReportIsAtomicAndReadable checks the emission. A torn document is
// worse than none — the consumer refuses unknown fields, so a half-written file reads
// as a hard error on a placement path that had been working.
func TestWriteServingReportIsAtomicAndReadable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "serving.json")
	rep, err := ServingReportFromSnapshot(
		[]WorkerStatus{{Spec: WorkerSpec{ID: "w1"}, Health: HealthHealthy}},
		ServingReportOptions{Now: time.Unix(1_700_000_000, 0), MaxAge: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteServingReport(path, rep); err != nil {
		t.Fatalf("WriteServingReport: %v", err)
	}
	// Rewrite in place: the second write must replace, not append or fail.
	if err := WriteServingReport(path, rep); err != nil {
		t.Fatalf("WriteServingReport (rewrite): %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), modelroute.ServingReportSchema) {
		t.Errorf("written document does not carry the schema:\n%s", raw)
	}
	var back modelroute.ServingReport
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	// The consumer refuses unknown fields; the producer must not emit any.
	dec.DisallowUnknownFields()
	if err := dec.Decode(&back); err != nil {
		t.Fatalf("the consumer's strict decoder rejects the producer's own output: %v", err)
	}
	if dec.More() {
		t.Error("more than one JSON document was written")
	}
	if err := back.Validate(); err != nil {
		t.Fatalf("round-tripped report does not validate: %v", err)
	}
	if back.Models["w1"].State != modelroute.ServingUp {
		t.Errorf("round-trip lost the observation: %+v", back.Models)
	}
	// No strand left behind: a crashed temp must not look like a report.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("write left strands beside the report: %v", names)
	}
}

// TestDefaultServingReportPathIsDerivedNotInvented pins that the well-known location
// comes off the existing fleet state dir rather than a new flag, and that it returns
// empty (rather than guessing) when nothing declares one — the caller fails closed.
func TestDefaultServingReportPathIsDerivedNotInvented(t *testing.T) {
	t.Setenv("FLEET_STATE_DIR", filepath.Join("X:", "fleet"))
	got := DefaultServingReportPath()
	if got == "" {
		t.Fatal("FLEET_STATE_DIR is set and no default serving path resolved")
	}
	if !strings.HasPrefix(got, filepath.Join("X:", "fleet")) {
		t.Errorf("default path %q is not under the declared FLEET_STATE_DIR", got)
	}
	if filepath.Base(got) != "serving.json" {
		t.Errorf("default path %q does not end in serving.json", got)
	}
	t.Setenv("FLEET_STATE_DIR", "")
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("LOCALAPPDATA", "")
	if p := DefaultServingReportPath(); p != "" {
		t.Errorf("a default path %q was invented with nothing declaring a state dir", p)
	}
}

// TestTheReportIsKeyedByTheModelsTheRouterFiltersOn pins the default keying against the
// labels WorkerSpec.Models carries (#5635). Without this the producer keyed every entry
// by worker id, so a report emitted from a labeled fleet named nothing the roster binds
// and the freshness gate it exists to feed passed over every candidate.
//
// It also pins the direction the fallback must NOT take. Empty Models means
// UNCONSTRAINED in the registry, so mirroring that here would report an unlabeled worker
// as evidence about every model in the report — and because the worst state wins, one
// unlabeled host going down would mark healthy models down. The fallback must stay
// narrow (the worker's own id) so an unlabeled worker gates nothing rather than gating
// the wrong candidate.
func TestTheReportIsKeyedByTheModelsTheRouterFiltersOn(t *testing.T) {
	snap := []WorkerStatus{
		{Spec: WorkerSpec{ID: "w-glm", Models: []string{"glm-4.6", "glm-4.6-air"}}, Health: HealthHealthy},
		{Spec: WorkerSpec{ID: "w-qwen", Models: []string{"qwen3-coder"}}, Health: HealthUnhealthy},
		{Spec: WorkerSpec{ID: "w-unlabeled"}, Health: HealthUnhealthy},
	}
	rep, err := ServingReportFromSnapshot(snap, ServingReportOptions{Now: time.Unix(1_700_000_000, 0), MaxAge: time.Minute})
	if err != nil {
		t.Fatalf("ServingReportFromSnapshot: %v", err)
	}
	for model, want := range map[string]modelroute.ServingState{
		"glm-4.6":     modelroute.ServingUp,
		"glm-4.6-air": modelroute.ServingUp,
		"qwen3-coder": modelroute.ServingDown,
	} {
		obs, ok := rep.Models[model]
		if !ok {
			t.Fatalf("report carries no observation for %q; keys = %v", model, servingReportKeys(rep))
		}
		if obs.State != want {
			t.Errorf("%s observed %q, want %q", model, obs.State, want)
		}
	}
	if _, ok := rep.Models["w-glm"]; ok {
		t.Error("a labeled worker was reported under its worker id; the roster binds model ids, so that entry gates nothing")
	}
	// The unlabeled DOWN worker must not have poisoned the labeled models.
	if got := rep.Models["glm-4.6"].State; got != modelroute.ServingUp {
		t.Errorf("glm-4.6 became %q after an unlabeled worker went down; the fallback is reporting one worker as evidence about every model", got)
	}
	if _, ok := rep.Models["w-unlabeled"]; !ok {
		t.Error("an unlabeled worker was dropped entirely; it should report under its own id so the unbound-observation diagnostic can see it")
	}
	// The seam is still the caller's when neither id is what the roster binds.
	custom, err := ServingReportFromSnapshot(snap, ServingReportOptions{
		Now:       time.Unix(1_700_000_000, 0),
		MaxAge:    time.Minute,
		ModelsFor: func(st WorkerStatus) []string { return []string{"routed/" + st.Spec.ID} },
	})
	if err != nil {
		t.Fatalf("ServingReportFromSnapshot with ModelsFor: %v", err)
	}
	if _, ok := custom.Models["routed/w-glm"]; !ok {
		t.Errorf("an explicit ModelsFor was ignored; keys = %v", servingReportKeys(custom))
	}
	if _, ok := custom.Models["glm-4.6"]; ok {
		t.Error("an explicit ModelsFor was merged with the default instead of replacing it")
	}
}

func servingReportKeys(rep modelroute.ServingReport) []string {
	out := make([]string, 0, len(rep.Models))
	for k := range rep.Models {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
