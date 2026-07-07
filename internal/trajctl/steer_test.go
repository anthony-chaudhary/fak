package trajctl

// steer_test.go — issue #2540: the re-anchor nudge rung. The done condition is
// tested literally: a synthetic drifting session receives exactly one re-anchor
// nudge, a healthy session receives none, and both decisions appear in the
// ledger. The gateway integration test drives the rung end-to-end against a
// fake session steer route (httptest) and asserts the ledger rows.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func steerObjective(id string) Objective {
	return Objective{
		ID:        id,
		Statement: "ship the re-anchor nudge rung with its regime gate",
		Plan:      []PlanPhase{{ID: "p1", Title: "gate"}, {ID: "p2", Title: "deliver"}},
		Status:    StatusActive,
	}
}

func w3Progress(objID string, value float64, ms int64) ScoreRow {
	return ScoreRow{
		ObjectiveID: objID,
		Value:       value,
		Method:      CommitScorerMethod,
		Version:     "test",
		Witness:     W3,
		UnixMillis:  ms,
	}
}

func w2Divergence(objID string, ms int64) ScoreRow {
	return ScoreRow{
		ObjectiveID: objID,
		Value:       1,
		Method:      ActivityDivergenceScorerMethod,
		Version:     "test",
		Witness:     W2,
		UnixMillis:  ms,
	}
}

// driftingState folds a declining witnessed progress curve — the DRIFT regime.
func driftingState(objID string) State {
	return State{
		Objectives: map[string]Objective{objID: steerObjective(objID)},
		Scores:     []ScoreRow{w3Progress(objID, 0.6, 1000), w3Progress(objID, 0.3, 2000)},
	}
}

// healthyState folds a rising witnessed progress curve — the no-action regime.
func healthyState(objID string) State {
	return State{
		Objectives: map[string]Objective{objID: steerObjective(objID)},
		Scores:     []ScoreRow{w3Progress(objID, 0.3, 1000), w3Progress(objID, 0.6, 2000)},
	}
}

// countingSteer records deliveries and fails while failures > 0.
type countingSteer struct {
	mu       sync.Mutex
	calls    int
	lastID   string
	lastText string
	failures int
}

func (c *countingSteer) fn() SteerFunc {
	return func(sessionID, text string) error {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.calls++
		c.lastID = sessionID
		c.lastText = text
		if c.failures > 0 {
			c.failures--
			return errStub("channel down")
		}
		return nil
	}
}

type errStub string

func (e errStub) Error() string { return string(e) }

func TestRegimeGateHealthyNoNudge(t *testing.T) {
	st := healthyState("obj-h")
	rec := &countingSteer{}
	ds := st.SteerSweep(Stamp{SessionID: "sess-h"}, 3000, rec.fn())
	if len(ds) != 1 {
		t.Fatalf("decisions = %d, want 1", len(ds))
	}
	d := ds[0]
	if d.Action != ActionNone || d.Signal != SignalHealthy {
		t.Fatalf("decision = %+v, want none/HEALTHY", d)
	}
	if !strings.Contains(d.Reason, "healthy") {
		t.Errorf("reason %q does not name the healthy regime", d.Reason)
	}
	if rec.calls != 0 {
		t.Errorf("healthy session was steered %d times, want 0", rec.calls)
	}
}

func TestDriftNudgesExactlyOnce(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, ObjectiveRecord(steerObjective("obj-d"))); err != nil {
		t.Fatal(err)
	}
	for _, row := range []ScoreRow{w3Progress("obj-d", 0.6, 1000), w3Progress("obj-d", 0.3, 2000)} {
		if err := Append(ledger, ScoreRecord(row)); err != nil {
			t.Fatal(err)
		}
	}
	rec := &countingSteer{}

	// Turn boundary 1: the drift is nudged, once.
	st := Fold(ReadLedgerFile(ledger))
	ds := st.SteerSweep(Stamp{SessionID: "sess-d", RunID: "run-1"}, 3000, rec.fn())
	if len(ds) != 1 || ds[0].Action != ActionNudge || ds[0].Signal != SignalDrift {
		t.Fatalf("sweep 1 = %+v, want one nudge on DRIFT", ds)
	}
	if !ds[0].Delivered || ds[0].DeliverErr != "" {
		t.Fatalf("nudge not delivered: %+v", ds[0])
	}
	for _, want := range []string{"re-anchor", "DRIFT", "ship the re-anchor nudge rung", "p1 (gate)", "0.60 -> 0.30"} {
		if !strings.Contains(ds[0].Packet, want) {
			t.Errorf("packet missing %q:\n%s", want, ds[0].Packet)
		}
	}
	if rec.calls != 1 || rec.lastID != "sess-d" || rec.lastText != ds[0].Packet {
		t.Fatalf("delivery = %d calls to %q, want exactly one carrying the packet", rec.calls, rec.lastID)
	}
	if n, err := AppendSteerDecisions(ledger, ds); err != nil || n != 1 {
		t.Fatalf("append = %d, %v", n, err)
	}

	// Turn boundary 2: same persisting drift — the episode holds, no second nudge.
	st = Fold(ReadLedgerFile(ledger))
	ds = st.SteerSweep(Stamp{SessionID: "sess-d", RunID: "run-1"}, 4000, rec.fn())
	if len(ds) != 1 || ds[0].Action != ActionNone {
		t.Fatalf("sweep 2 = %+v, want a held none decision", ds)
	}
	if !strings.Contains(ds[0].Reason, "outstanding") {
		t.Errorf("hold reason = %q", ds[0].Reason)
	}
	if rec.calls != 1 {
		t.Fatalf("session steered %d times, want exactly 1", rec.calls)
	}
	if _, err := AppendSteerDecisions(ledger, ds); err != nil {
		t.Fatal(err)
	}

	// Both decisions are in the ledger: the acted nudge and the held none.
	st = Fold(ReadLedgerFile(ledger))
	steers := st.SteersFor("obj-d")
	if len(steers) != 2 || steers[0].Action != ActionNudge || !steers[0].Delivered || steers[1].Action != ActionNone {
		t.Fatalf("ledgered steers = %+v, want [delivered nudge, held none]", steers)
	}
}

func TestStallNudges(t *testing.T) {
	st := State{
		Objectives: map[string]Objective{"obj-s": steerObjective("obj-s")},
		Scores: []ScoreRow{
			w3Progress("obj-s", 0.5, 1000),
			w3Progress("obj-s", 0.5, 2000),
			w2Divergence("obj-s", 2000),
		},
	}
	rec := &countingSteer{}
	ds := st.SteerSweep(Stamp{SessionID: "sess-s"}, 3000, rec.fn())
	if len(ds) != 1 || ds[0].Action != ActionNudge || ds[0].Signal != SignalStall {
		t.Fatalf("decisions = %+v, want one nudge on STALL", ds)
	}
	if rec.calls != 1 {
		t.Fatalf("calls = %d, want 1", rec.calls)
	}
}

func TestRecoveryRearmsTheGate(t *testing.T) {
	st := driftingState("obj-r")
	// Episode 1 nudged and the curve later recovered (a ledgered HEALTHY
	// decision); a fresh drift is a new episode and may nudge again.
	st.Steers = []SteerDecision{
		{ObjectiveID: "obj-r", Action: ActionNudge, Signal: SignalDrift, Reason: "r", Packet: "p", Delivered: true},
		{ObjectiveID: "obj-r", Action: ActionNone, Signal: SignalHealthy, Reason: "healthy"},
	}
	oc, _ := st.CurveFor("obj-r")
	if d := st.DecideNudge(oc); d.Action != ActionNudge {
		t.Fatalf("recovered episode did not re-arm: %+v", d)
	}

	// Without the recovery row the episode still holds.
	st.Steers = st.Steers[:1]
	if d := st.DecideNudge(oc); d.Action != ActionNone {
		t.Fatalf("un-recovered episode re-nudged: %+v", d)
	}
}

func TestFailedDeliveryStaysArmed(t *testing.T) {
	st := driftingState("obj-f")
	rec := &countingSteer{failures: 1}

	ds := st.SteerSweep(Stamp{SessionID: "sess-f"}, 3000, rec.fn())
	if len(ds) != 1 || ds[0].Action != ActionNudge || ds[0].Delivered || ds[0].DeliverErr == "" {
		t.Fatalf("failed delivery row = %+v, want an undelivered nudge with the error captured", ds)
	}

	// The undelivered nudge does not consume the episode: the next boundary retries.
	st.Steers = ds
	ds = st.SteerSweep(Stamp{SessionID: "sess-f"}, 4000, rec.fn())
	if len(ds) != 1 || ds[0].Action != ActionNudge || !ds[0].Delivered {
		t.Fatalf("retry = %+v, want a delivered nudge", ds)
	}
	if rec.calls != 2 {
		t.Fatalf("calls = %d, want 2", rec.calls)
	}

	// A nil channel is captured fail-open, never a panic.
	st = driftingState("obj-f")
	ds = st.SteerSweep(Stamp{SessionID: "sess-f"}, 5000, nil)
	if len(ds) != 1 || ds[0].Delivered || ds[0].DeliverErr == "" {
		t.Fatalf("nil-channel row = %+v", ds)
	}

	// A panicking channel is captured fail-open too.
	ds = st.SteerSweep(Stamp{SessionID: "sess-f"}, 6000, func(string, string) error { panic("boom") })
	if len(ds) != 1 || ds[0].Delivered || !strings.Contains(ds[0].DeliverErr, "panicked") {
		t.Fatalf("panicking-channel row = %+v", ds)
	}
}

// TestGatewaySteerIntegration is the issue's named witness: the rung drives a
// fake session's steer route end-to-end. The drifting session receives exactly
// one re-anchor POST on the existing channel shape, the healthy session
// receives none, and both decisions land in the ledger.
func TestGatewaySteerIntegration(t *testing.T) {
	type post struct {
		method, path, text string
	}
	var mu sync.Mutex
	var posts []post
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("steer body did not decode: %v", err)
		}
		mu.Lock()
		posts = append(posts, post{r.Method, r.URL.Path, body.Text})
		mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
		if err := json.NewEncoder(w).Encode(map[string]any{"trace_id": "sess-drift", "steered": true}); err != nil {
			t.Errorf("encode ack: %v", err)
		}
	}))
	defer srv.Close()
	deliver := GatewaySteer(srv.URL, "", srv.Client())

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")

	drift := driftingState("obj-gw-d")
	if _, err := AppendSteerDecisions(ledger, drift.SteerSweep(Stamp{SessionID: "sess-drift"}, 3000, deliver)); err != nil {
		t.Fatal(err)
	}
	healthy := healthyState("obj-gw-h")
	if _, err := AppendSteerDecisions(ledger, healthy.SteerSweep(Stamp{SessionID: "sess-healthy"}, 3000, deliver)); err != nil {
		t.Fatal(err)
	}

	// Exactly one nudge crossed the wire, on the existing channel shape, to the
	// drifting session.
	mu.Lock()
	defer mu.Unlock()
	if len(posts) != 1 {
		t.Fatalf("steer channel saw %d posts, want exactly 1: %+v", len(posts), posts)
	}
	if posts[0].method != http.MethodPost || posts[0].path != "/v1/fak/session/sess-drift/steer" {
		t.Fatalf("post = %s %s, want POST /v1/fak/session/sess-drift/steer", posts[0].method, posts[0].path)
	}
	if !strings.Contains(posts[0].text, "re-anchor") || !strings.Contains(posts[0].text, "ship the re-anchor nudge rung") {
		t.Errorf("wire packet missing the re-anchor content:\n%s", posts[0].text)
	}

	// Both decisions — the delivered nudge and the healthy no-action — are ledgered.
	st := Fold(ReadLedgerFile(ledger))
	nudges := st.SteersFor("obj-gw-d")
	if len(nudges) != 1 || nudges[0].Action != ActionNudge || !nudges[0].Delivered || nudges[0].Signal != SignalDrift {
		t.Fatalf("drift ledger rows = %+v, want one delivered nudge on DRIFT", nudges)
	}
	holds := st.SteersFor("obj-gw-h")
	if len(holds) != 1 || holds[0].Action != ActionNone || holds[0].Signal != SignalHealthy {
		t.Fatalf("healthy ledger rows = %+v, want one ledgered no-action", holds)
	}
}

func TestGatewaySteerSurfacesRefusal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "steer refused: tainted", http.StatusUnprocessableEntity)
	}))
	defer srv.Close()
	err := GatewaySteer(srv.URL, "", srv.Client())("sess-x", "packet")
	if err == nil || !strings.Contains(err.Error(), "422") || !strings.Contains(err.Error(), "tainted") {
		t.Fatalf("err = %v, want the 422 refusal surfaced", err)
	}
}

func TestDetourOverrunIsNotThisRung(t *testing.T) {
	st := driftingState("obj-o")
	d := st.DecideNudge(ObjectiveCurve{ObjectiveID: "obj-o", Signal: SignalDetourOverrun})
	if d.Action != ActionNone || !strings.Contains(d.Reason, "return-to-main") {
		t.Fatalf("DETOUR_OVERRUN decision = %+v, want a ledgered none naming the return-to-main rung", d)
	}
}

func TestSteerRowValidation(t *testing.T) {
	good := SteerDecision{ObjectiveID: "o", Action: ActionNudge, Signal: SignalDrift, Reason: "r", Packet: "p"}
	if err := Validate(SteerRecord(good)); err != nil {
		t.Fatalf("good steer row refused: %v", err)
	}
	bad := []SteerDecision{
		{Action: ActionNudge, Signal: SignalDrift, Reason: "r", Packet: "p"},                    // no objective
		{ObjectiveID: "o", Action: "shout", Signal: SignalDrift, Reason: "r"},                   // foreign action
		{ObjectiveID: "o", Action: ActionNudge, Signal: "WAT", Reason: "r"},                     // foreign signal
		{ObjectiveID: "o", Action: ActionNudge, Signal: SignalDrift, Reason: ""},                // no reason
		{ObjectiveID: "o", Action: ActionNudge, Signal: SignalDrift, Reason: "r"},               // nudge without packet
		{ObjectiveID: "o", Action: ActionNone, Signal: SignalHealthy, Reason: "r", Packet: "p"}, // none with packet
	}
	for i, d := range bad {
		if err := Validate(SteerRecord(d)); err == nil {
			t.Errorf("bad steer row %d admitted: %+v", i, d)
		}
	}

	// A steer row round-trips through the ledger and folds into State.Steers;
	// exclusivity holds for the other kinds.
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, SteerRecord(good)); err != nil {
		t.Fatal(err)
	}
	st := Fold(ReadLedgerFile(ledger))
	if len(st.Steers) != 1 || st.Steers[0] != good {
		t.Fatalf("folded steers = %+v", st.Steers)
	}
	mixed := Row{Schema: Schema, Kind: KindScore, Score: &ScoreRow{ObjectiveID: "o", Method: "m", Version: "1", Witness: W3}, Steer: &good}
	if err := Validate(mixed); err == nil {
		t.Error("score row carrying a steer payload admitted")
	}
}
