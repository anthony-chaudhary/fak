package experiments

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func TestNativePerfTransactionKeepAndReplay(t *testing.T) {
	r := nativePerfRequest(t, false)
	got, err := RunNativePerfTransaction(r)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decision != NativePerfKeep || got.DefaultStateReadback.Selected != "candidate" || got.DefaultStateReadback.Revision != r.GateRequest.Candidate.Revision {
		t.Fatalf("KEEP journal/default mismatch: %+v", got)
	}
	if got.ProfileDigest == "" || got.ObligationGraphDigest == "" || got.BaselineReceiptDigest == "" || got.CandidateReceiptDigest == "" || got.GateRequestDigest == "" || got.ReplayDigest == "" {
		t.Fatalf("missing journal digests: %+v", got)
	}
	if got.SelectedObligationID != r.ObligationID || got.SelectedCandidateID != r.CandidateID || got.GateOutput.Classification != nativeperf.GatePass {
		t.Fatalf("selection/gate output mismatch: %+v", got)
	}
	if err := ReplayNativePerfTransaction(r, got); err != nil {
		t.Fatalf("replay: %v", err)
	}

	path := filepath.Join(t.TempDir(), "decisions.jsonl")
	if err := AppendNativePerfDecision(path, got); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := ParseNativePerfDecisionLedger(string(content))
	if err != nil || len(rows) != 1 || rows[0].ReplayDigest != got.ReplayDigest {
		t.Fatalf("parse appended row: rows=%+v err=%v", rows, err)
	}
	if err := AppendNativePerfDecision(path, got); err == nil {
		t.Fatal("duplicate immutable decision append succeeded")
	}
}

func TestNativePerfTransactionRejectRetainsBaseline(t *testing.T) {
	r := nativePerfRequest(t, true)
	got, err := RunNativePerfTransaction(r)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decision != NativePerfReject || got.DefaultStateReadback.Selected != "baseline" || got.DefaultStateReadback.Revision != r.GateRequest.LastAccepted.Revision {
		t.Fatalf("REJECT did not retain baseline: %+v", got)
	}
	if got.GateOutput.Classification != nativeperf.GateRegression {
		t.Fatalf("gate classification=%q want %q", got.GateOutput.Classification, nativeperf.GateRegression)
	}
	if err := ReplayNativePerfTransaction(r, got); err != nil {
		t.Fatalf("replay: %v", err)
	}
}

func TestNativePerfTransactionRefusesGraphAndDefaultMismatch(t *testing.T) {
	r := nativePerfRequest(t, false)
	r.ObligationGraphDigest = "sha256:" + strings.Repeat("0", 64)
	if _, err := RunNativePerfTransaction(r); err == nil || !strings.Contains(err.Error(), "graph digest mismatch") {
		t.Fatalf("graph mismatch error=%v", err)
	}

	r = nativePerfRequest(t, false)
	r.DefaultReadback = DefaultReadback{Selected: "baseline", Revision: r.GateRequest.LastAccepted.Revision}
	if _, err := RunNativePerfTransaction(r); err == nil || !strings.Contains(err.Error(), "default-state readback mismatch") {
		t.Fatalf("default mismatch error=%v", err)
	}

	r = nativePerfRequest(t, false)
	r.GateRequest.Candidate.Execution.Engine = "llama.cpp"
	if _, err := RunNativePerfTransaction(r); err == nil {
		t.Fatal("receipt envelope mismatch was accepted")
	}
}

func TestNativePerfDecisionReplayAndLedgerTamperFail(t *testing.T) {
	r := nativePerfRequest(t, false)
	got, err := RunNativePerfTransaction(r)
	if err != nil {
		t.Fatal(err)
	}
	tampered := got
	tampered.SelectedCandidateID = "issue-9999"
	if err := ReplayNativePerfTransaction(r, tampered); err == nil {
		t.Fatal("tampered replay succeeded")
	}
	data, err := json.Marshal(tampered)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseNativePerfDecisionLedger(string(data) + "\n"); err == nil || !strings.Contains(err.Error(), "replay digest mismatch") {
		t.Fatalf("tampered ledger error=%v", err)
	}
	valid, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	unknown := strings.TrimSuffix(string(valid), "}") + `,"unknown":true}` + "\n"
	if _, err := ParseNativePerfDecisionLedger(unknown); err == nil {
		t.Fatal("unknown ledger field was accepted")
	}
	if _, err := ParseNativePerfDecisionLedger(string(valid)); err == nil {
		t.Fatal("unterminated ledger row was accepted")
	}
}

func nativePerfRequest(t *testing.T, reject bool) NativePerfTransactionRequest {
	t.Helper()
	profileData, err := os.ReadFile(filepath.Join("..", "nativeperf", "testdata", "native-performance-profile", "synthetic-cuda-bandwidth-bound.json"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := nativeperf.DecodeProfile(profileData)
	if err != nil {
		t.Fatal(err)
	}
	classification, err := nativeperf.ClassifyProfile(nativeperf.ActiveGraph(), profile)
	if err != nil {
		t.Fatal(err)
	}
	candidateID := "issue-8635-candidate"
	obligationID := "qwen38-cuda-nativeperf-candidate"
	graph := newmodel.NativeObligationGraph{
		Schema: "fak-native-obligation-graph/v1", Engine: "fak-native",
		ManifestDigest: "sha256:" + strings.Repeat("1", 64), DescriptorDigest: "sha256:" + strings.Repeat("2", 64), EnvelopeDigest: "sha256:" + strings.Repeat("3", 64),
		Nodes: []newmodel.NativeObligation{{
			ID: obligationID, Class: "fusion", Operation: classification.RecommendedLeverID, Eligible: true,
			Backend:          newmodel.NativeBackendObligation{Engine: "fak-native", Platform: "cuda", Backend: "cuda", Operation: classification.RecommendedLeverID},
			PromotionWitness: newmodel.NativePromotionWitness{ID: candidateID, Kind: "nativeperf-gate"},
		}},
	}
	graphDigest, err := digestJSON(graph)
	if err != nil {
		t.Fatal(err)
	}
	baseline := nativePerfReceipt(t, nativeperf.RoleBaseline, classification.RecommendedLeverID)
	candidate := nativePerfReceipt(t, nativeperf.RoleCandidate, classification.RecommendedLeverID)
	baseline.Revision = "baseline-r1"
	candidate.Revision = "candidate-r2"
	for i := range baseline.Repetitions {
		baseline.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
		candidate.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
		if reject {
			candidate.Repetitions[i].TokensPerSecond = 85
		}
	}
	gate := nativeperf.GateRequest{
		Schema: nativeperf.GateRequestSchema,
		Policy: nativeperf.GatePolicy{
			Schema: nativeperf.GatePolicySchema, EnvelopeID: baseline.EnvelopeID, ChangedLeverID: baseline.ChangedLeverID,
			AcceptedRevision: baseline.Revision, MinimumRepetitions: 3, MaximumNoisePercent: 2,
			InvestigateDropPercent: 2, RegressionDropPercent: 5, MinimumThroughput: 90,
			MaximumPeakBytes: 1200, QualityMetric: "exact_match", MinimumQualityScore: 1,
			QualityHigherIsBetter: true, RequiredEngine: "fak-native", RequiredForwardPath: baseline.Execution.ForwardPath,
		},
		LastAccepted: baseline, Candidate: candidate,
	}
	readback := DefaultReadback{Selected: "candidate", Revision: candidate.Revision}
	if reject {
		readback = DefaultReadback{Selected: "baseline", Revision: baseline.Revision}
	}
	return NativePerfTransactionRequest{
		Profile: profile, ObligationGraph: graph, ObligationGraphDigest: graphDigest,
		ObligationID: obligationID, CandidateID: candidateID, GateRequest: gate, DefaultReadback: readback,
	}
}

func nativePerfReceipt(t *testing.T, role, lever string) nativeperf.ExperimentReceipt {
	t.Helper()
	r, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
	if err != nil {
		t.Fatal(err)
	}
	r.Role = role
	r.Revision = "revision"
	r.Machine.ScrubbedID = "lab-node-class-a"
	r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
	r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
	r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r10+gaaaaaaa"}}
	r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
	r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
	for i := range r.Repetitions {
		r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 100 + float64(i%2)}
	}
	if role == nativeperf.RoleCandidate {
		r.ChangedAxes = []string{"lever:" + lever}
	}
	return r
}
