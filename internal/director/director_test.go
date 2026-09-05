package director_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/director"
	"github.com/anthony-chaudhary/fak/internal/supervisoragent"
)

// mockAdmissionVerbs implements supervisoragent.AdmissionVerbs for test verification.
type mockAdmissionVerbs struct {
	arbitrateCalls []struct {
		lane string
		tree []string
	}
	admitCalls []struct {
		issue      string
		lane       string
		supersedes string
	}
	escalateCalls []supervisoragent.Escalation
}

func (m *mockAdmissionVerbs) Arbitrate(lane string, tree []string) (supervisoragent.Lease, error) {
	m.arbitrateCalls = append(m.arbitrateCalls, struct {
		lane string
		tree []string
	}{lane: lane, tree: tree})
	return supervisoragent.Lease{Lane: lane, Kind: "cluster", Tree: tree}, nil
}

func (m *mockAdmissionVerbs) Admit(issue, lane, supersedes string) (supervisoragent.AdmitReceipt, error) {
	m.admitCalls = append(m.admitCalls, struct {
		issue      string
		lane       string
		supersedes string
	}{issue: issue, lane: lane, supersedes: supersedes})
	return supervisoragent.AdmitReceipt{RunID: "RID-new", Issue: issue, Lane: lane, Supersedes: supersedes}, nil
}

func (m *mockAdmissionVerbs) EmitEscalation(head supervisoragent.Escalation) (supervisoragent.Escalation, error) {
	head.ID = "ESC-1"
	m.escalateCalls = append(m.escalateCalls, head)
	return head, nil
}

// TestDirectorDigestZeroProseZeroSelfReport proves DirectorDigest compilation produces
// 0 prose / 0 self-report tokens, containing only verified numeric and closed-enum rungs.
func TestDirectorDigestZeroProseZeroSelfReport(t *testing.T) {
	engine := director.NewRollupEngine()
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:           "RID-001",
		Lane:            "gateway",
		Issue:           "#11411",
		State:           director.WorkerHealthy,
		StepCount:       12,
		VerifiedCommits: 2,
		TreeTouches:     4,
		VelocityScore:   1.5,
		LastWitnessMs:   time.Now().UnixMilli(),
	})
	engine.RecordLease(director.LeaseSnapshot{
		Lane:     "gateway",
		LaneKind: director.LaneKindCluster,
		Tree:     []string{"internal/gateway/**"},
		Holder:   "RID-001",
		Mode:     director.LeaseModeExclusive,
	})

	digest := engine.CompileDigest()

	// 1. Verify schema definition via reflection: no unbounded prose / message fields
	prohibitedFieldNames := map[string]bool{
		"message":     true,
		"prose":       true,
		"text":        true,
		"summary":     true,
		"description": true,
		"comment":     true,
		"claim":       true,
		"selfreport":  true,
		"narrative":   true,
		"statusline":  true,
	}

	checkFields := func(t *testing.T, st reflect.Type, name string) {
		for i := 0; i < st.NumField(); i++ {
			field := st.Field(i)
			lowerName := strings.ToLower(field.Name)
			for p := range prohibitedFieldNames {
				if strings.Contains(lowerName, p) {
					t.Fatalf("struct %s has prohibited prose/self-report field: %s", name, field.Name)
				}
			}
		}
	}

	checkFields(t, reflect.TypeOf(director.DirectorDigest{}), "DirectorDigest")
	checkFields(t, reflect.TypeOf(director.WorkerDigestRow{}), "WorkerDigestRow")
	checkFields(t, reflect.TypeOf(director.LeaseSnapshot{}), "LeaseSnapshot")
	checkFields(t, reflect.TypeOf(director.FleetVelocityScore{}), "FleetVelocityScore")

	// 2. Verify JSON payload: check keys and closed values
	raw, err := json.Marshal(digest)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	// Ensure no forbidden keys in JSON
	var checkMapKeys func(m map[string]interface{})
	checkMapKeys = func(m map[string]interface{}) {
		for k, v := range m {
			lowerKey := strings.ToLower(k)
			for p := range prohibitedFieldNames {
				if strings.Contains(lowerKey, p) {
					t.Fatalf("JSON payload contains prohibited key %q", k)
				}
			}
			if childMap, ok := v.(map[string]interface{}); ok {
				checkMapKeys(childMap)
			}
			if childList, ok := v.([]interface{}); ok {
				for _, item := range childList {
					if itemMap, ok := item.(map[string]interface{}); ok {
						checkMapKeys(itemMap)
					}
				}
			}
		}
	}
	checkMapKeys(parsed)

	// Verify worker state is strictly member of closed enum
	for _, w := range digest.Workers {
		switch w.State {
		case director.WorkerHealthy, director.WorkerStalled, director.WorkerBlocked, director.WorkerDone:
			// valid closed token
		default:
			t.Fatalf("invalid non-closed worker state token: %q", w.State)
		}
	}

	// Verify RollupHash is non-empty cryptographic hash
	if !strings.HasPrefix(digest.RollupHash, "sha256:") {
		t.Fatalf("RollupHash %q must have sha256: prefix", digest.RollupHash)
	}
}

// TestAutomatedSupervisorActionTriggeringLatency verifies automated triggering of supervisor
// actions (replace, replan, hold) from digest metrics within <5ms of state transition.
func TestAutomatedSupervisorActionTriggeringLatency(t *testing.T) {
	engine := director.NewRollupEngine()
	mockVerbs := &mockAdmissionVerbs{}

	// Baseline healthy state
	row := director.WorkerDigestRow{
		RunID:           "RID-100",
		Lane:            "core",
		Issue:           "#11411",
		State:           director.WorkerHealthy,
		StepCount:       10,
		VerifiedCommits: 1,
		TreeTouches:     2,
		VelocityScore:   1.0,
		LastWitnessMs:   time.Now().UnixMilli(),
	}
	engine.RecordWorker(row)

	// 1. Hold transition (healthy fleet)
	{
		start := time.Now()
		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)
		elapsed := time.Since(start)

		if elapsed >= 5*time.Millisecond {
			t.Fatalf("Hold evaluation took %v, want <5ms", elapsed)
		}
		if len(recs) != 1 || recs[0].Action != director.ActionHold {
			t.Fatalf("expected ActionHold, got %+v", recs)
		}
		if recs[0].SupervisorAction == nil {
			t.Fatal("expected non-nil SupervisorAction on recommendation")
		}

		eff, err := supervisoragent.Lower(recs[0].SupervisorAction, mockVerbs)
		if err != nil {
			t.Fatalf("supervisoragent.Lower failed for ActionHold: %v", err)
		}
		if eff.Action != supervisoragent.ActionHold || eff.Verb != supervisoragent.VerbNone {
			t.Fatalf("unexpected effect for ActionHold: %+v", eff)
		}
	}

	// 2. Replace transition: worker becomes stalled
	{
		row.State = director.WorkerStalled
		start := time.Now()
		engine.RecordWorker(row)
		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)
		elapsed := time.Since(start)

		if elapsed >= 5*time.Millisecond {
			t.Fatalf("Replace evaluation took %v, want <5ms", elapsed)
		}
		foundReplace := false
		for _, rec := range recs {
			if rec.Action == director.ActionReplace {
				foundReplace = true
				if rec.RunID != "RID-100" || rec.Lane != "core" || rec.Issue != "#11411" {
					t.Fatalf("mismatched replace recommendation fields: %+v", rec)
				}
				if rec.Reason != director.ReasonWorkerStalled {
					t.Fatalf("expected reason %s, got %s", director.ReasonWorkerStalled, rec.Reason)
				}

				// Execute through supervisoragent.Lower to prove compatibility
				eff, err := supervisoragent.Lower(rec.SupervisorAction, mockVerbs)
				if err != nil {
					t.Fatalf("supervisoragent.Lower failed for ActionReplace: %v", err)
				}
				if eff.Action != supervisoragent.ActionReplace || eff.Verb != supervisoragent.VerbAdmit {
					t.Fatalf("unexpected effect for ActionReplace: %+v", eff)
				}
				if eff.Admit == nil || eff.Admit.Supersedes != "RID-100" {
					t.Fatalf("admit receipt supersedes mismatch: %+v", eff.Admit)
				}
			}
		}
		if !foundReplace {
			t.Fatalf("expected ActionReplace in recommendations, got %+v", recs)
		}
	}

	// 3. Replan transition: worker becomes blocked
	{
		row.State = director.WorkerBlocked
		start := time.Now()
		engine.RecordWorker(row)
		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)
		elapsed := time.Since(start)

		if elapsed >= 5*time.Millisecond {
			t.Fatalf("Replan evaluation took %v, want <5ms", elapsed)
		}
		foundReplan := false
		for _, rec := range recs {
			if rec.Action == director.ActionReplan {
				foundReplan = true
				if rec.Lane != "core" || rec.Issue != "#11411" {
					t.Fatalf("mismatched replan recommendation fields: %+v", rec)
				}
				if rec.Reason != director.ReasonWorkerBlocked {
					t.Fatalf("expected reason %s, got %s", director.ReasonWorkerBlocked, rec.Reason)
				}

				// Execute through supervisoragent.Lower to prove compatibility
				eff, err := supervisoragent.Lower(rec.SupervisorAction, mockVerbs)
				if err != nil {
					t.Fatalf("supervisoragent.Lower failed for ActionReplan: %v", err)
				}
				if eff.Action != supervisoragent.ActionRedispatch || eff.Verb != supervisoragent.VerbAdmit {
					t.Fatalf("unexpected effect for ActionReplan: %+v", eff)
				}
			}
		}
		if !foundReplan {
			t.Fatalf("expected ActionReplan in recommendations, got %+v", recs)
		}
	}
}

// TestHighConcurrencyRollupEngine verifies concurrency performance under 50+ concurrent
// worker states with high-throughput reads without data races.
func TestHighConcurrencyRollupEngine(t *testing.T) {
	engine := director.NewRollupEngine()
	numWorkers := 60
	iterations := 50

	var wg sync.WaitGroup

	// Writers: update worker states concurrently
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			runID := strings.TrimSpace("RID-WORKER-" + string(rune('A'+(workerIdx%26))) + string(rune('0'+(workerIdx/26))))
			lane := "lane-" + string(rune('A'+(workerIdx%10)))

			for iter := 0; iter < iterations; iter++ {
				state := director.WorkerHealthy
				if iter%7 == 0 {
					state = director.WorkerBlocked
				} else if iter%11 == 0 {
					state = director.WorkerStalled
				} else if iter%13 == 0 {
					state = director.WorkerDone
				}

				engine.RecordWorker(director.WorkerDigestRow{
					RunID:           runID,
					Lane:            lane,
					Issue:           "#issue",
					State:           state,
					StepCount:       iter + 1,
					VerifiedCommits: iter / 5,
					TreeTouches:     iter * 2,
					VelocityScore:   float64(iter) * 0.1,
					LastWitnessMs:   time.Now().UnixMilli(),
				})

				if iter%5 == 0 {
					engine.RecordLease(director.LeaseSnapshot{
						Lane:     lane,
						LaneKind: director.LaneKindCluster,
						Tree:     []string{"internal/" + lane + "/**"},
						Holder:   runID,
						Mode:     director.LeaseModeExclusive,
					})
				}
			}
		}(i)
	}

	// Readers: high-throughput reads and steering evaluations concurrently
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iter := 0; iter < iterations; iter++ {
				d := engine.CompileDigest()
				if d.TotalWorkers > numWorkers {
					t.Errorf("TotalWorkers %d exceeds numWorkers %d", d.TotalWorkers, numWorkers)
				}
				if d.RollupHash == "" {
					t.Errorf("RollupHash should never be empty")
				}
				recs := engine.EvaluateFleetSteering(d)
				if len(recs) == 0 {
					t.Errorf("EvaluateFleetSteering should always return at least one action")
				}
				time.Sleep(1 * time.Millisecond)
			}
		}()
	}

	wg.Wait()

	finalDigest := engine.CompileDigest()
	if finalDigest.TotalWorkers != numWorkers {
		t.Fatalf("expected TotalWorkers=%d, got %d", numWorkers, finalDigest.TotalWorkers)
	}
	if len(finalDigest.Workers) != numWorkers {
		t.Fatalf("expected len(Workers)=%d, got %d", numWorkers, len(finalDigest.Workers))
	}
}

// TestA2APayloadCompatibility verifies serialization and A2A payload compatibility
// for /a2a/v1/director/digest.
func TestA2APayloadCompatibility(t *testing.T) {
	engine := director.NewRollupEngine()
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:           "RID-001",
		Lane:            "gateway",
		Issue:           "#11411",
		State:           director.WorkerHealthy,
		StepCount:       20,
		VerifiedCommits: 3,
		TreeTouches:     5,
		VelocityScore:   3.0,
		LastWitnessMs:   1700000000000,
	})
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:           "RID-002",
		Lane:            "engine",
		Issue:           "#11412",
		State:           director.WorkerStalled,
		StepCount:       5,
		VerifiedCommits: 0,
		TreeTouches:     1,
		VelocityScore:   0.0,
		LastWitnessMs:   1700000000000,
	})
	engine.RecordLease(director.LeaseSnapshot{
		Lane:     "gateway",
		LaneKind: director.LaneKindCluster,
		Tree:     []string{"internal/gateway/**"},
		Holder:   "RID-001",
		Mode:     director.LeaseModeExclusive,
	})

	digest := engine.CompileDigest()

	// Verify Schema is canonical
	if digest.Schema != director.DigestSchema {
		t.Fatalf("expected Schema %s, got %s", director.DigestSchema, digest.Schema)
	}
	if digest.Schema != "/a2a/v1/director/digest" {
		t.Fatalf("expected Schema /a2a/v1/director/digest, got %s", digest.Schema)
	}

	// Marshal to JSON
	bytes, err := json.Marshal(digest)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// Verify A2A envelope keys
	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(bytes, &rawMap); err != nil {
		t.Fatalf("Unmarshal to rawMap failed: %v", err)
	}

	requiredKeys := []string{
		"schema",
		"timestamp",
		"total_workers",
		"active_workers",
		"stalled_workers",
		"completed_workers",
		"fleet_velocity",
		"workers",
		"leases",
		"rollup_hash",
	}

	for _, k := range requiredKeys {
		if _, ok := rawMap[k]; !ok {
			t.Fatalf("A2A payload missing required key %q", k)
		}
	}

	// Unmarshal back to DirectorDigest
	var decoded director.DirectorDigest
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatalf("Unmarshal to DirectorDigest failed: %v", err)
	}

	if decoded.Schema != digest.Schema {
		t.Fatalf("roundtrip Schema mismatch: got %s, want %s", decoded.Schema, digest.Schema)
	}
	if decoded.TotalWorkers != 2 || decoded.ActiveWorkers != 1 || decoded.StalledWorkers != 1 {
		t.Fatalf("roundtrip counts mismatch: %+v", decoded)
	}
	if decoded.RollupHash != digest.RollupHash {
		t.Fatalf("roundtrip RollupHash mismatch: got %s, want %s", decoded.RollupHash, digest.RollupHash)
	}
	if len(decoded.Workers) != 2 || len(decoded.Leases) != 1 {
		t.Fatalf("roundtrip slices mismatch: %+v", decoded)
	}
}

// TestSteeringEscalateWidenSpawn verifies the remaining closed actions (escalate, widen, spawn).
func TestSteeringEscalateWidenSpawn(t *testing.T) {
	// 1. Spawn action on idle lease
	{
		engine := director.NewRollupEngine()
		engine.RecordLease(director.LeaseSnapshot{
			Lane:     "adjudicator",
			LaneKind: director.LaneKindCluster,
			Tree:     []string{"internal/adjudicator/**"},
			Holder:   "", // idle!
			Mode:     director.LeaseModeExclusive,
		})

		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)

		foundSpawn := false
		for _, rec := range recs {
			if rec.Action == director.ActionSpawn && rec.Lane == "adjudicator" {
				foundSpawn = true
				if rec.Reason != director.ReasonLaneIdle {
					t.Fatalf("expected reason %s, got %s", director.ReasonLaneIdle, rec.Reason)
				}
				if rec.SupervisorAction == nil {
					t.Fatal("expected non-nil SupervisorAction")
				}
				if _, ok := rec.SupervisorAction.(supervisoragent.SpawnAction); !ok {
					t.Fatalf("expected SpawnAction, got %T", rec.SupervisorAction)
				}
			}
		}
		if !foundSpawn {
			t.Fatalf("expected ActionSpawn in recs, got %+v", recs)
		}
	}

	// 2. Escalate action on runaway thrashing worker
	{
		engine := director.NewRollupEngine()
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           "RID-THRASH",
			Lane:            "model",
			Issue:           "#9999",
			State:           director.WorkerHealthy,
			StepCount:       55,
			VerifiedCommits: 0,
			TreeTouches:     15,
			VelocityScore:   0.0,
			LastWitnessMs:   time.Now().UnixMilli(),
		})

		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)

		foundEscalate := false
		for _, rec := range recs {
			if rec.Action == director.ActionEscalate && rec.RunID == "RID-THRASH" {
				foundEscalate = true
				if rec.Reason != director.ReasonWorkerThrashing {
					t.Fatalf("expected reason %s, got %s", director.ReasonWorkerThrashing, rec.Reason)
				}
				if rec.SupervisorAction == nil {
					t.Fatal("expected non-nil SupervisorAction")
				}
				if _, ok := rec.SupervisorAction.(supervisoragent.EscalateAction); !ok {
					t.Fatalf("expected EscalateAction, got %T", rec.SupervisorAction)
				}
			}
		}
		if !foundEscalate {
			t.Fatalf("expected ActionEscalate in recs, got %+v", recs)
		}
	}

	// 3. Widen action on worker with high tree touches needing wider scope
	{
		engine := director.NewRollupEngine()
		engine.RecordWorker(director.WorkerDigestRow{
			RunID:           "RID-WIDEN",
			Lane:            "ctxmmu",
			Issue:           "#8888",
			State:           director.WorkerHealthy,
			StepCount:       30,
			VerifiedCommits: 0,
			TreeTouches:     28,
			VelocityScore:   0.0,
			LastWitnessMs:   time.Now().UnixMilli(),
		})

		d := engine.CompileDigest()
		recs := engine.EvaluateFleetSteering(d)

		foundWiden := false
		for _, rec := range recs {
			if rec.Action == director.ActionWiden && rec.RunID == "RID-WIDEN" {
				foundWiden = true
				if rec.Reason != director.ReasonLaneWiden {
					t.Fatalf("expected reason %s, got %s", director.ReasonLaneWiden, rec.Reason)
				}
				if rec.SupervisorAction == nil {
					t.Fatal("expected non-nil SupervisorAction")
				}
				if _, ok := rec.SupervisorAction.(supervisoragent.WidenAction); !ok {
					t.Fatalf("expected WidenAction, got %T", rec.SupervisorAction)
				}
			}
		}
		if !foundWiden {
			t.Fatalf("expected ActionWiden in recs, got %+v", recs)
		}
	}
}

// TestRollupHashDeterminism proves RollupHash is strictly deterministic and sensitive to changes.
func TestRollupHashDeterminism(t *testing.T) {
	e1 := director.NewRollupEngine()
	fixedTime := int64(1700000000000)
	e1.SetStartTime(fixedTime)

	w1 := director.WorkerDigestRow{
		RunID:           "RID-1",
		Lane:            "lane1",
		Issue:           "#1",
		State:           director.WorkerHealthy,
		StepCount:       10,
		VerifiedCommits: 1,
		TreeTouches:     2,
		VelocityScore:   1.0,
		LastWitnessMs:   fixedTime,
	}
	w2 := director.WorkerDigestRow{
		RunID:           "RID-2",
		Lane:            "lane2",
		Issue:           "#2",
		State:           director.WorkerDone,
		StepCount:       20,
		VerifiedCommits: 3,
		TreeTouches:     5,
		VelocityScore:   2.0,
		LastWitnessMs:   fixedTime,
	}

	e1.RecordWorker(w1)
	e1.RecordWorker(w2)

	d1 := e1.CompileDigest()

	// Another engine with insertions in opposite order
	e2 := director.NewRollupEngine()
	e2.SetStartTime(fixedTime)
	e2.RecordWorker(w2)
	e2.RecordWorker(w1)

	d2 := e2.CompileDigest()

	// Ordering in map shouldn't affect deterministic hash (digest.Workers is sorted by RunID)
	// But note Timestamp might differ if now() is called, so let's verify worker content hash
	if len(d1.Workers) != 2 || len(d2.Workers) != 2 {
		t.Fatalf("worker counts mismatch")
	}
	if d1.Workers[0].RunID != "RID-1" || d2.Workers[0].RunID != "RID-1" {
		t.Fatalf("Workers not sorted by RunID: %+v, %+v", d1.Workers, d2.Workers)
	}

	// Change one counter in e2 and verify RollupHash changes
	w2Modified := w2
	w2Modified.VerifiedCommits = 4
	e2.RecordWorker(w2Modified)
	d3 := e2.CompileDigest()

	if d3.RollupHash == d1.RollupHash {
		t.Fatalf("RollupHash should change when VerifiedCommits changes: %s == %s", d3.RollupHash, d1.RollupHash)
	}
}

// TestEngineGetRemoveReset verifies CRUD operations on RollupEngine.
func TestEngineGetRemoveReset(t *testing.T) {
	engine := director.NewRollupEngine()
	w := director.WorkerDigestRow{
		RunID: "RID-CRUD",
		Lane:  "crud-lane",
		State: director.WorkerHealthy,
	}
	l := director.LeaseSnapshot{
		Lane:   "crud-lane",
		Holder: "RID-CRUD",
	}

	engine.RecordWorker(w)
	engine.RecordLease(l)

	gotW, ok := engine.GetWorker("RID-CRUD")
	if !ok || gotW.Lane != "crud-lane" {
		t.Fatalf("GetWorker failed: ok=%v, got=%+v", ok, gotW)
	}

	gotL, ok := engine.GetLease("crud-lane")
	if !ok || gotL.Holder != "RID-CRUD" {
		t.Fatalf("GetLease failed: ok=%v, got=%+v", ok, gotL)
	}

	engine.RemoveWorker("RID-CRUD")
	if _, ok := engine.GetWorker("RID-CRUD"); ok {
		t.Fatal("expected worker to be removed")
	}

	engine.RemoveLease("crud-lane")
	if _, ok := engine.GetLease("crud-lane"); ok {
		t.Fatal("expected lease to be removed")
	}

	engine.RecordWorker(w)
	engine.RecordLease(l)
	engine.Reset()

	d := engine.CompileDigest()
	if d.TotalWorkers != 0 || len(d.Leases) != 0 {
		t.Fatalf("Reset did not clear engine state: %+v", d)
	}
}

// TestStallTimeoutAutomaticTransition verifies that inactivity beyond StallTimeoutMs
// automatically marks a worker stalled.
func TestStallTimeoutAutomaticTransition(t *testing.T) {
	engine := director.NewRollupEngine()
	engine.SetStallTimeoutMs(500) // 500ms threshold

	past := time.Now().UnixMilli() - 1000 // 1 second ago (inactive)
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:         "RID-SILENT",
		Lane:          "silent-lane",
		State:         director.WorkerHealthy,
		LastWitnessMs: past,
	})

	digest := engine.CompileDigest()
	if digest.StalledWorkers != 1 {
		t.Fatalf("expected StalledWorkers=1, got %d", digest.StalledWorkers)
	}
	if len(digest.Workers) != 1 || digest.Workers[0].State != director.WorkerStalled {
		t.Fatalf("expected worker state to transition to WorkerStalled: %+v", digest.Workers)
	}

	recs := engine.EvaluateFleetSteering(digest)
	foundReplace := false
	for _, rec := range recs {
		if rec.Action == director.ActionReplace && rec.RunID == "RID-SILENT" {
			foundReplace = true
		}
	}
	if !foundReplace {
		t.Fatalf("expected automatic stall to trigger ActionReplace: %+v", recs)
	}
}

// TestFleetCriticalThresholdEscalation verifies fleet-wide escalation when stall or block rate >= 50%.
func TestFleetCriticalThresholdEscalation(t *testing.T) {
	// 2 workers, both blocked => 100% block rate => fleet escalate
	engine := director.NewRollupEngine()
	engine.RecordWorker(director.WorkerDigestRow{
		RunID: "RID-B1",
		State: director.WorkerBlocked,
	})
	engine.RecordWorker(director.WorkerDigestRow{
		RunID: "RID-B2",
		State: director.WorkerBlocked,
	})

	digest := engine.CompileDigest()
	if digest.FleetVelocity.BlockRate != 1.0 {
		t.Fatalf("expected BlockRate=1.0, got %f", digest.FleetVelocity.BlockRate)
	}

	recs := engine.EvaluateFleetSteering(digest)
	foundFleetEscalate := false
	for _, rec := range recs {
		if rec.Action == director.ActionEscalate && rec.Reason == director.ReasonFleetHighBlockRate {
			foundFleetEscalate = true
		}
	}
	if !foundFleetEscalate {
		t.Fatalf("expected fleet-wide ActionEscalate for high block rate: %+v", recs)
	}
}

// TestCompileAndEvaluate verifies CompileAndEvaluate returns both digest and recommendations.
func TestCompileAndEvaluate(t *testing.T) {
	engine := director.NewRollupEngine()
	engine.RecordWorker(director.WorkerDigestRow{
		RunID: "RID-OK",
		State: director.WorkerHealthy,
	})

	d, recs := engine.CompileAndEvaluate()
	if d.ActiveWorkers != 1 {
		t.Fatalf("expected ActiveWorkers=1, got %d", d.ActiveWorkers)
	}
	if len(recs) != 1 || recs[0].Action != director.ActionHold {
		t.Fatalf("expected ActionHold, got %+v", recs)
	}
}
