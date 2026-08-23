package ultracoderesume

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/workflow"
)

const fixtureSchema = "fak-ultracode-crash-fixture/1"

type crashFixture struct {
	Schema                string                     `json:"schema"`
	Plan                  orchestration.WorkflowPlan `json:"plan"`
	Completed             []string                   `json:"completed"`
	Partial               string                     `json:"partial"`
	DependencyInvalidated string                     `json:"dependency_invalidated"`
	PrivateMarkers        []string                   `json:"private_markers"`
}

func loadCrashFixture(t *testing.T) crashFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "crash-boundaries.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture crashFixture
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fixture); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if fixture.Schema != fixtureSchema {
		t.Fatalf("fixture schema = %q", fixture.Schema)
	}
	return fixture
}

type primedRun struct {
	fixture crashFixture
	store   Store
	graph   GraphReceipt
}

func primeCrashRun(t *testing.T, mirrorObserve bool) primedRun {
	t.Helper()
	fixture := loadCrashFixture(t)
	store := Store{Dir: t.TempDir()}
	graph, err := store.Initialize(fixture.Plan)
	if err != nil {
		t.Fatalf("initialize: %v", err)
	}
	appendCompleted(t, store, graph, "observe-config", 1, Admission{},
		ExecutionReceipt{ReceiptKey: "receipt:observe-v1", WitnessKey: "witness:observe-v1"}, 100, mirrorObserve, nil)
	appendCompleted(t, store, graph, "effect-land", 1,
		effectAdmission(t, "effect-land", "effect-receipt:land-v1", "lease:land-v1", "cap:land-v1", 10_000),
		ExecutionReceipt{ReceiptKey: "receipt:land-v1", WitnessKey: "witness:land-v1"}, 110, false, nil)
	appendStarted(t, store, graph, "effect-partial", 1,
		effectAdmission(t, "effect-partial", "effect-receipt:partial-v1", "lease:partial-v1", "cap:partial-v1", 10_000), 120)
	return primedRun{fixture: fixture, store: store, graph: graph}
}

func appendStarted(t *testing.T, store Store, graph GraphReceipt, node string, attempt int, admission Admission, ts int64) {
	t.Helper()
	receipt, err := NewAttemptReceipt(graph, node, AttemptRecord{
		Attempt: attempt, Status: AttemptStarted, Admission: admission, TSMS: ts,
	})
	if err != nil {
		t.Fatalf("new started receipt for %s: %v", node, err)
	}
	if err := store.AppendAttempt(receipt); err != nil {
		t.Fatalf("append started receipt for %s: %v", node, err)
	}
}

func appendCompleted(t *testing.T, store Store, graph GraphReceipt, node string, attempt int, admission Admission, result ExecutionReceipt, ts int64, mirror bool, deps map[string]string) {
	t.Helper()
	appendStarted(t, store, graph, node, attempt, admission, ts-1)
	receipt, err := NewAttemptReceipt(graph, node, AttemptRecord{
		Attempt: attempt, Status: AttemptCompleted, Admission: admission, Result: result, TSMS: ts,
	})
	if err != nil {
		t.Fatalf("new completed receipt for %s: %v", node, err)
	}
	if err := store.AppendAttempt(receipt); err != nil {
		t.Fatalf("append completed receipt for %s: %v", node, err)
	}
	if !mirror {
		return
	}
	flow := graph.workflowGraph()
	var flowNode workflow.Node
	for _, candidate := range flow.Nodes {
		if candidate.ID == node {
			flowNode = candidate
			break
		}
	}
	depHashes := make(map[string]string, len(deps))
	for id, output := range deps {
		depHashes[id] = workflow.HashOutput(output)
	}
	if err := store.appendWorkflow(workflow.Entry{
		Run: graph.GraphID, Step: node, Kind: workflow.StepEffectful,
		InputsHash: workflow.StepInputsHash(flowNode, depHashes), EpochHash: graph.GraphEpoch,
		OutputHash: workflow.HashOutput(result.ReceiptKey), Output: result.ReceiptKey,
		Claim: result.WitnessKey, TSMS: ts,
	}); err != nil {
		t.Fatalf("append workflow mirror for %s: %v", node, err)
	}
}

func witnessSet(keys ...string) Corroborate {
	allowed := make(map[string]bool, len(keys))
	for _, key := range keys {
		allowed[key] = true
	}
	return func(_ context.Context, _ string, key string) (string, bool) {
		if !allowed[key] {
			return "", false
		}
		return "fixture-readback:" + key, true
	}
}

func TestCrashResumeSelectivelyRerunsAndThenIsIdempotent(t *testing.T) {
	run := primeCrashRun(t, true)
	corr := witnessSet(
		"witness:observe-v1", "witness:land-v1",
		"witness:partial-v2", "witness:reconcile-v1",
	)
	var executed []string
	report, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
		NowMS:       1_000,
		Corroborate: corr,
		Acquire: func(_ context.Context, node GraphNode, previous string) (Admission, error) {
			if node.ID != run.fixture.Partial || previous != "lease:partial-v1" {
				t.Fatalf("unexpected lease request node=%s previous=%s", node.ID, previous)
			}
			return effectAdmission(t, node.ID, "effect-receipt:partial-v2", "lease:partial-v2", "cap:partial-v2", 20_000), nil
		},
		Runner: func(_ context.Context, node GraphNode, _ Admission) (ExecutionReceipt, error) {
			executed = append(executed, node.ID)
			switch node.ID {
			case run.fixture.Partial:
				return ExecutionReceipt{ReceiptKey: "receipt:partial-v2", WitnessKey: "witness:partial-v2"}, nil
			case run.fixture.DependencyInvalidated:
				return ExecutionReceipt{ReceiptKey: "receipt:reconcile-v1", WitnessKey: "witness:reconcile-v1"}, nil
			default:
				t.Fatalf("completed node %s was repeated", node.ID)
				return ExecutionReceipt{}, nil
			}
		},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if report.Skipped != 2 || report.Rerun != 2 || report.Refused != 0 {
		t.Fatalf("resume counts = skip %d rerun %d refuse %d; report=%+v", report.Skipped, report.Rerun, report.Refused, report)
	}
	if !reflect.DeepEqual(executed, []string{"effect-partial", "reconcile"}) {
		t.Fatalf("executed = %v", executed)
	}
	byID := NodeReportsByID(report)
	if byID["observe-config"].Disposition != DispositionSkip || byID["effect-land"].Disposition != DispositionSkip {
		t.Fatalf("completed nodes were not skipped: %+v", byID)
	}
	partial := byID[run.fixture.Partial]
	if partial.Reason != ReasonIncompleteAttempt || partial.PreviousLeaseID != "lease:partial-v1" || partial.CurrentLeaseID != "lease:partial-v2" {
		t.Fatalf("partial row = %+v", partial)
	}
	if partial.PriorEffectReceiptID != "effect-receipt:partial-v1" || partial.CurrentEffectReceiptID != "effect-receipt:partial-v2" {
		t.Fatalf("partial effect receipt mapping = %+v", partial)
	}
	if got := byID[run.fixture.DependencyInvalidated].Reason; got != ReasonDependencyInvalid {
		t.Fatalf("dependent reason = %q", got)
	}

	second, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
		NowMS: 2_000, Corroborate: corr,
		Acquire: func(context.Context, GraphNode, string) (Admission, error) {
			t.Fatal("idempotent resume acquired a lease")
			return Admission{}, nil
		},
		Runner: func(context.Context, GraphNode, Admission) (ExecutionReceipt, error) {
			t.Fatal("idempotent resume executed a node")
			return ExecutionReceipt{}, nil
		},
	})
	if err != nil || second.Skipped != 4 || second.Rerun != 0 || second.Refused != 0 {
		t.Fatalf("second resume: err=%v report=%+v", err, second)
	}
	firstJSON, err := StableJSON(second)
	if err != nil {
		t.Fatalf("stable json: %v", err)
	}
	inspected, err := Inspect(context.Background(), run.store, run.fixture.Plan, corr)
	if err != nil {
		t.Fatalf("inspect: %v", err)
	}
	secondJSON, _ := StableJSON(inspected)
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("reconstruction is not deterministic:\n%s\n---\n%s", firstJSON, secondJSON)
	}

	attempts, err := run.store.ReadAttempts()
	if err != nil {
		t.Fatalf("read attempts: %v", err)
	}
	type attemptKey struct {
		node    string
		attempt int
		status  AttemptStatus
	}
	wantAttempts := map[attemptKey]bool{
		{"observe-config", 1, AttemptStarted}:   false,
		{"observe-config", 1, AttemptCompleted}: false,
		{"effect-land", 1, AttemptStarted}:      false,
		{"effect-land", 1, AttemptCompleted}:    false,
		{"effect-partial", 1, AttemptStarted}:   false,
		{"effect-partial", 2, AttemptStarted}:   false,
		{"effect-partial", 2, AttemptCompleted}: false,
		{"reconcile", 1, AttemptStarted}:        false,
		{"reconcile", 1, AttemptCompleted}:      false,
	}
	for _, attempt := range attempts {
		key := attemptKey{attempt.NodeID, attempt.Attempt, attempt.Status}
		if _, ok := wantAttempts[key]; !ok {
			t.Fatalf("unexpected attempt row: %+v", attempt)
		}
		wantAttempts[key] = true
	}
	for key, seen := range wantAttempts {
		if !seen {
			t.Errorf("missing attempt row: %+v", key)
		}
	}
	journal, err := run.store.ReadWorkflowJournal()
	if err != nil || len(journal) != 3 {
		t.Fatalf("workflow rows=%d err=%v, want 3 (one completion recovered from receipt-only crash)", len(journal), err)
	}
}

func TestCrashReceiptsAndReportRedactPlanBodies(t *testing.T) {
	run := primeCrashRun(t, true)
	report, err := Inspect(context.Background(), run.store, run.fixture.Plan,
		witnessSet("witness:observe-v1", "witness:land-v1"))
	if err != nil {
		t.Fatalf("inspect: %v", err)
	}
	reportJSON, _ := StableJSON(report)
	graphJSON, err := os.ReadFile(filepath.Join(run.store.Dir, GraphFile))
	if err != nil {
		t.Fatalf("read graph receipt: %v", err)
	}
	attemptJSON, err := os.ReadFile(filepath.Join(run.store.Dir, AttemptsFile))
	if err != nil {
		t.Fatalf("read attempts: %v", err)
	}
	all := string(reportJSON) + string(graphJSON) + string(attemptJSON)
	for _, marker := range run.fixture.PrivateMarkers {
		if strings.Contains(all, marker) {
			t.Fatalf("durable resume artifacts leaked private marker %q", marker)
		}
	}
}

func TestCorruptAttemptJournalRefusesWithoutExecuting(t *testing.T) {
	run := primeCrashRun(t, true)
	f, err := os.OpenFile(filepath.Join(run.store.Dir, AttemptsFile), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open attempts: %v", err)
	}
	_, _ = f.WriteString("{corrupt-after-controller-crash\n")
	_ = f.Close()
	report, err := Inspect(context.Background(), run.store, run.fixture.Plan, witnessSet())
	assertRefusal(t, report, err, RefusalAttemptCorrupt)
}

func TestChangedGraphIdentityRefuses(t *testing.T) {
	run := primeCrashRun(t, true)
	changed := run.fixture.Plan
	changed.EngineRef = "executionroute:changed"
	report, err := Inspect(context.Background(), run.store, changed, witnessSet())
	assertRefusal(t, report, err, RefusalGraphIdentity)
}

func TestIncompatibleGraphAndAttemptSchemasRefuse(t *testing.T) {
	t.Run("graph", func(t *testing.T) {
		run := primeCrashRun(t, true)
		graph := run.graph
		graph.Schema = "fak-ultracode-graph-receipt/2"
		writeJSONFile(t, filepath.Join(run.store.Dir, GraphFile), graph)
		report, err := Inspect(context.Background(), run.store, run.fixture.Plan, witnessSet())
		assertRefusal(t, report, err, RefusalGraphSchema)
	})
	t.Run("epoch", func(t *testing.T) {
		run := primeCrashRun(t, true)
		graph := run.graph
		graph.GraphEpoch = "sha256:changed-epoch"
		writeJSONFile(t, filepath.Join(run.store.Dir, GraphFile), graph)
		report, err := Inspect(context.Background(), run.store, run.fixture.Plan, witnessSet())
		assertRefusal(t, report, err, RefusalGraphEpoch)
	})
	t.Run("attempt", func(t *testing.T) {
		run := primeCrashRun(t, true)
		rows, err := run.store.ReadAttempts()
		if err != nil {
			t.Fatalf("read attempts: %v", err)
		}
		rows[0].Schema = "fak-ultracode-node-attempt/2"
		writeJSONLines(t, filepath.Join(run.store.Dir, AttemptsFile), rows)
		report, err := Inspect(context.Background(), run.store, run.fixture.Plan, witnessSet())
		assertRefusal(t, report, err, RefusalAttemptSchema)
	})
}

func TestMissingReceiptRerunsWithoutRepeatingCompletedEffect(t *testing.T) {
	run := primeCrashRun(t, false)
	rows, err := run.store.ReadAttempts()
	if err != nil {
		t.Fatalf("read attempts: %v", err)
	}
	kept := rows[:0]
	for _, row := range rows {
		if row.NodeID != "observe-config" {
			kept = append(kept, row)
		}
	}
	writeJSONLines(t, filepath.Join(run.store.Dir, AttemptsFile), kept)

	var executed []string
	report, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
		NowMS: 1_000,
		Corroborate: witnessSet(
			"witness:land-v1", "witness:observe-config-v1",
			"witness:effect-partial-v1", "witness:reconcile-v1",
		),
		Acquire: func(_ context.Context, node GraphNode, _ string) (Admission, error) {
			return effectAdmission(t, node.ID, "effect-receipt:"+node.ID+"-fresh", "lease:"+node.ID+"-fresh", "cap:"+node.ID+"-fresh", 20_000), nil
		},
		Runner: func(_ context.Context, node GraphNode, _ Admission) (ExecutionReceipt, error) {
			executed = append(executed, node.ID)
			return ExecutionReceipt{ReceiptKey: "receipt:" + node.ID + "-v1", WitnessKey: "witness:" + node.ID + "-v1"}, nil
		},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	row := NodeReportsByID(report)["observe-config"]
	if row.Disposition != DispositionRerun || row.Reason != ReasonMissingReceipt {
		t.Fatalf("missing receipt row = %+v", row)
	}
	if contains(executed, "effect-land") {
		t.Fatalf("completed effect repeated while repairing missing node: %v", executed)
	}
}

func TestInvalidReadReceiptRerunsButInvalidCompletedEffectRefuses(t *testing.T) {
	t.Run("observe reruns selectively", func(t *testing.T) {
		run := primeCrashRun(t, false)
		corruptAttemptID(t, run.store, "observe-config", AttemptCompleted)
		var executed []string
		report, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
			NowMS:       1_000,
			Corroborate: witnessSet("witness:land-v1", "witness:observe-config-v2", "witness:effect-partial-v2", "witness:reconcile-v2"),
			Acquire: func(_ context.Context, node GraphNode, _ string) (Admission, error) {
				return effectAdmission(t, node.ID, "effect-receipt:"+node.ID+"-v2", "lease:"+node.ID+"-v2", "cap:"+node.ID+"-v2", 20_000), nil
			},
			Runner: func(_ context.Context, node GraphNode, _ Admission) (ExecutionReceipt, error) {
				executed = append(executed, node.ID)
				return ExecutionReceipt{ReceiptKey: "receipt:" + node.ID + "-v2", WitnessKey: "witness:" + node.ID + "-v2"}, nil
			},
		})
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		row := NodeReportsByID(report)["observe-config"]
		if row.Disposition != DispositionRerun || row.Reason != ReasonInvalidReceipt {
			t.Fatalf("invalid observe row = %+v", row)
		}
		if contains(executed, "effect-land") {
			t.Fatalf("valid completed effect was repeated: %v", executed)
		}
	})
	t.Run("effect refuses ambiguous rerun", func(t *testing.T) {
		run := primeCrashRun(t, false)
		corruptAttemptID(t, run.store, "effect-land", AttemptCompleted)
		report, err := Inspect(context.Background(), run.store, run.fixture.Plan,
			witnessSet("witness:observe-v1", "witness:land-v1"))
		if err != nil {
			t.Fatalf("inspect: %v", err)
		}
		row := NodeReportsByID(report)["effect-land"]
		if row.Disposition != DispositionRefuse || row.Reason != ReasonAmbiguousEffect {
			t.Fatalf("invalid effect row = %+v", row)
		}
	})
}

func TestUnwitnessedCompletedEffectIsNotBlindlyRerun(t *testing.T) {
	run := primeCrashRun(t, true)
	var executed []string
	report, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
		NowMS:       1_000,
		Corroborate: witnessSet("witness:observe-v1", "witness:partial-v2", "witness:reconcile-v1"),
		Acquire: func(_ context.Context, node GraphNode, _ string) (Admission, error) {
			return effectAdmission(t, node.ID, "effect-receipt:"+node.ID+"-v2", "lease:"+node.ID+"-v2", "cap:"+node.ID+"-v2", 20_000), nil
		},
		Runner: func(_ context.Context, node GraphNode, _ Admission) (ExecutionReceipt, error) {
			executed = append(executed, node.ID)
			return ExecutionReceipt{ReceiptKey: "receipt:" + node.ID + "-v2", WitnessKey: "witness:" + node.ID + "-v2"}, nil
		},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	row := NodeReportsByID(report)["effect-land"]
	if row.Disposition != DispositionRefuse || row.Reason != ReasonAmbiguousEffect || contains(executed, "effect-land") {
		t.Fatalf("unwitnessed effect was not fail-closed: row=%+v executed=%v", row, executed)
	}
}

func TestExpiredFreshCapabilityRefusesEffectAndDependent(t *testing.T) {
	run := primeCrashRun(t, true)
	report, err := Resume(context.Background(), run.store, run.fixture.Plan, Options{
		NowMS:       5_000,
		Corroborate: witnessSet("witness:observe-v1", "witness:land-v1"),
		Acquire: func(_ context.Context, _ GraphNode, _ string) (Admission, error) {
			return effectAdmission(t, "effect-partial", "effect-receipt:partial-v2", "lease:partial-v2", "cap:expired", 5_000), nil
		},
		Runner: func(context.Context, GraphNode, Admission) (ExecutionReceipt, error) {
			t.Fatal("expired capability reached runner")
			return ExecutionReceipt{}, nil
		},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	byID := NodeReportsByID(report)
	if byID[run.fixture.Partial].Reason != ReasonCapabilityExpired || byID[run.fixture.DependencyInvalidated].Reason != ReasonDependencyRefused {
		t.Fatalf("expired capability projection = %+v", byID)
	}
}

func corruptAttemptID(t *testing.T, store Store, node string, status AttemptStatus) {
	t.Helper()
	rows, err := store.ReadAttempts()
	if err != nil {
		t.Fatalf("read attempts: %v", err)
	}
	found := false
	for i := range rows {
		if rows[i].NodeID == node && rows[i].Status == status {
			rows[i].ReceiptID = "sha256:damaged"
			found = true
		}
	}
	if !found {
		t.Fatalf("attempt %s/%s not found", node, status)
	}
	writeJSONLines(t, filepath.Join(store.Dir, AttemptsFile), rows)
}

func effectAdmission(t *testing.T, nodeID, receiptID, leaseID, capabilityID string, expiresMS int64) Admission {
	t.Helper()
	admission, err := AdmissionFromEffectSuccessor(nodeID, orchestration.EffectSuccessorReceipt{
		Schema: orchestration.EffectSuccessorReceiptSchema,
		ID:     receiptID, NodeID: nodeID, LeaseID: leaseID,
	}, capabilityID, expiresMS)
	if err != nil {
		t.Fatalf("effect admission for %s: %v", nodeID, err)
	}
	return admission
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", filepath.Base(path), err)
	}
}

func writeJSONLines(t *testing.T, path string, rows []AttemptReceipt) {
	t.Helper()
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			t.Fatalf("encode row: %v", err)
		}
	}
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatalf("write attempts: %v", err)
	}
}

func assertRefusal(t *testing.T, report Report, err error, want RefusalReason) {
	t.Helper()
	var refusal *Refusal
	if !errors.As(err, &refusal) {
		t.Fatalf("error = %v, want refusal %s", err, want)
	}
	if refusal.Reason != want || report.Refusal == nil || report.Refusal.Reason != want {
		t.Fatalf("refusal=%+v report=%+v, want %s", refusal, report.Refusal, want)
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
