package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/experiments"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// fak-test:justify why=integration when=changed:cmd/fak/experiments*
// fak-test:runtime fast est=100ms lane=default
func TestExperimentsNativePerformanceTransactionRoundTrip(t *testing.T) {
	for _, reject := range []bool{false, true} {
		name := "keep"
		wantDecision, wantSelected := experiments.NativePerfKeep, "candidate"
		if reject {
			name, wantDecision, wantSelected = "reject", experiments.NativePerfReject, "baseline"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			request := nativeperfCLIRequest(t, reject)
			input, ledger, decision := filepath.Join(dir, "request.json"), filepath.Join(dir, "decisions.jsonl"), filepath.Join(dir, "decision.json")
			writeExperimentJSON(t, input, request)
			original := mustReadExperimentFile(t, input)
			var result nativeperfCLIResponse
			nativeperfCLIRun(t, 0, &result, "evaluate", "--file", input, "--ledger", ledger, "--json")
			nativeperfCLIAssertScope(t, result)
			if result.Journal.Decision != wantDecision || result.Journal.DefaultStateReadback.Selected != wantSelected {
				t.Fatalf("decision=%q selected=%q, want %q/%q", result.Journal.Decision, result.Journal.DefaultStateReadback.Selected, wantDecision, wantSelected)
			}
			if err := experiments.ReplayNativePerfTransaction(request, result.Journal); err != nil {
				t.Fatalf("CLI journal does not replay through the existing transaction: %v", err)
			}
			content := mustReadExperimentFile(t, ledger)
			rows, err := experiments.ParseNativePerfDecisionLedger(string(content))
			if err != nil || len(rows) != 1 || rows[0].ReplayDigest != result.Journal.ReplayDigest {
				t.Fatalf("immutable journal append: rows=%d err=%v", len(rows), err)
			}
			writeExperimentJSON(t, decision, result.Journal)
			var replay nativeperfCLIResponse
			nativeperfCLIRun(t, 0, &replay, "replay", "--file", input, "--decision", decision, "--json")
			nativeperfCLIAssertScope(t, replay)
			if replay.Replayed == nil || !*replay.Replayed || replay.Journal.ReplayDigest != result.Journal.ReplayDigest {
				t.Fatal("ordinary replay did not confirm the exact journal")
			}
			human := nativeperfCLIRun(t, 0, nil, "replay", "--file", input, "--decision", decision)
			for _, label := range []string{"supplied-files", "hardware_verified=false", "default_observation=caller-supplied", "serving_mutated=false"} {
				if !bytes.Contains(human, []byte(label)) {
					t.Fatalf("human output omitted evidence boundary %q", label)
				}
			}
			nativeperfCLIRun(t, 1, nil, "evaluate", "--file", input, "--ledger", ledger, "--json")
			if !bytes.Equal(content, mustReadExperimentFile(t, ledger)) || !bytes.Equal(original, mustReadExperimentFile(t, input)) {
				t.Fatal("replay or duplicate evaluation mutated immutable evidence")
			}
		})
	}
}

// fak-test:justify why=guard when=changed:cmd/fak/experiments* twin=TestExperimentsNativePerformanceTransactionRoundTrip
// fak-test:runtime fast est=300ms lane=default
func TestExperimentsNativePerformanceInvalidEvidenceCannotAppend(t *testing.T) {
	for _, name := range []string{"malformed", "unknown_field", "trailing_object", "oversized", "profile_missing_fallback", "profile_null_fallback", "baseline_missing_fallback", "baseline_null_fallback", "candidate_missing_fallback", "candidate_null_fallback", "graph_digest", "fallback", "default_mismatch", "tampered_replay", "trailing_replay", "corrupt_ledger_trailer", "corrupt_ledger_brace"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			request := nativeperfCLIRequest(t, false)
			journal, err := experiments.RunNativePerfTransaction(request)
			if err != nil {
				t.Fatal(err)
			}
			input, ledger, decision := filepath.Join(dir, "request.json"), filepath.Join(dir, "decisions.jsonl"), filepath.Join(dir, "decision.json")
			if err := experiments.AppendNativePerfDecision(ledger, journal); err != nil {
				t.Fatal(err)
			}
			before := mustReadExperimentFile(t, ledger)
			code, mode := 2, "evaluate"
			var raw []byte
			switch name {
			case "malformed":
				raw = []byte("{")
			case "graph_digest":
				request.ObligationGraphDigest, code = cmdTestDigest("wrong graph"), 1
			case "fallback":
				request.GateRequest.Candidate.Execution.Engine, code = "llama.cpp", 1
			case "default_mismatch":
				request.DefaultReadback = experiments.DefaultReadback{Selected: "baseline", Revision: request.GateRequest.LastAccepted.Revision}
				code = 1
			case "corrupt_ledger_trailer", "corrupt_ledger_brace":
				trailer := " ]\n"
				if name == "corrupt_ledger_brace" {
					trailer = " }\n"
				}
				before = append(bytes.TrimSuffix(before, []byte("\n")), []byte(trailer)...)
				if err := os.WriteFile(ledger, before, 0o600); err != nil {
					t.Fatal(err)
				}
				request.GateRequest.Candidate.Revision += "-next"
				request.DefaultReadback.Revision, code = request.GateRequest.Candidate.Revision, 1
			case "tampered_replay", "trailing_replay":
				mode = "replay"
				if name == "tampered_replay" {
					journal.SelectedCandidateID, code = "tampered-candidate", 1
				}
			}
			if raw == nil {
				raw, err = json.Marshal(request)
				if err != nil {
					t.Fatal(err)
				}
			}
			if name == "unknown_field" {
				raw = append(bytes.TrimSuffix(raw, []byte("}")), []byte(",\"unknown\":true}")...)
			}
			if name == "trailing_object" {
				raw = append(raw, []byte("\n{}")...)
			}
			if name == "oversized" {
				raw = append(bytes.Repeat([]byte(" "), experiments.MaxReceiptLineBytes), raw...)
			}
			if strings.HasSuffix(name, "_fallback") {
				var object map[string]any
				if err := json.Unmarshal(raw, &object); err != nil {
					t.Fatal(err)
				}
				identity := object["profile"].(map[string]any)["execution"].(map[string]any)
				if !strings.HasPrefix(name, "profile_") {
					role := "last_accepted"
					if strings.HasPrefix(name, "candidate_") {
						role = "candidate"
					}
					identity = object["gate_request"].(map[string]any)[role].(map[string]any)["execution"].(map[string]any)
				}
				if strings.Contains(name, "_missing_") {
					delete(identity, "fallback_count")
				} else {
					identity["fallback_count"] = nil
				}
				raw, err = json.Marshal(object)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(input, raw, 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{mode, "--file", input, "--ledger", ledger, "--json"}
			var decisionBefore []byte
			if mode == "replay" {
				writeExperimentJSON(t, decision, journal)
				if name == "trailing_replay" {
					b := append(mustReadExperimentFile(t, decision), []byte("\n{}")...)
					if err := os.WriteFile(decision, b, 0o600); err != nil {
						t.Fatal(err)
					}
				}
				decisionBefore = mustReadExperimentFile(t, decision)
				args = []string{mode, "--file", input, "--decision", decision, "--json"}
			}
			nativeperfCLIRun(t, code, nil, args...)
			if !bytes.Equal(before, mustReadExperimentFile(t, ledger)) || !bytes.Equal(raw, mustReadExperimentFile(t, input)) {
				t.Fatal("rejected evidence changed ledger or input bytes")
			}
			if mode == "replay" && !bytes.Equal(decisionBefore, mustReadExperimentFile(t, decision)) {
				t.Fatal("rejected replay changed the supplied journal")
			}
		})
	}
}

// fak-test:justify why=integration when=changed:cmd/fak/experiments*
// fak-test:runtime fast est=300ms lane=default
func TestExperimentsNativePerformanceDecisionReaderBoundsAndUnknownTail(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "decisions.jsonl")
	const rowCount = 23
	var fixture []byte
	for i := 0; i < rowCount; i++ {
		r := nativeperfCLIRequest(t, false)
		r.GateRequest.Candidate.Revision += strconv.Itoa(i)
		r.DefaultReadback.Revision = r.GateRequest.Candidate.Revision
		row, err := experiments.RunNativePerfTransaction(r)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		fixture = append(fixture, append(encoded, '\n')...)
	}
	if err := os.WriteFile(ledger, fixture, 0o600); err != nil {
		t.Fatal(err)
	}
	content := mustReadExperimentFile(t, ledger)
	latestRevision := "synthetic-candidate" + strconv.Itoa(rowCount-1)
	var defaults nativeperfCLIResponse
	nativeperfCLIRun(t, 0, &defaults, "decisions", "--ledger", ledger, "--json")
	if defaults.RowLimit != 20 {
		t.Fatalf("default decision row limit = %d, want 20", defaults.RowLimit)
	}
	firstRevision := "synthetic-candidate" + strconv.Itoa(rowCount-defaults.RowLimit)
	if len(defaults.Decisions) != defaults.RowLimit || defaults.Decisions[0].DefaultStateReadback.Revision != firstRevision || defaults.Decisions[defaults.RowLimit-1].DefaultStateReadback.Revision != latestRevision {
		t.Fatalf("default reader hid the latest decisions or reordered the retained tail: returned=%d rows_read=%d", defaults.Returned, defaults.RowsRead)
	}
	var countBound nativeperfCLIResponse
	nativeperfCLIRun(t, 0, &countBound, "decisions", "--ledger", ledger, "--limit", "1", "--json")
	nativeperfCLIAssertScope(t, countBound)
	if len(countBound.Decisions) != 1 || countBound.Returned != 1 || countBound.RowLimit != 1 || countBound.RowsRead != rowCount || countBound.TotalRows == nil || *countBound.TotalRows != rowCount || countBound.OmittedRows == nil || *countBound.OmittedRows != rowCount-1 || !countBound.RowTruncated || countBound.ByteTruncated || !countBound.Truncated || countBound.Decisions[0].DefaultStateReadback.Revision != latestRevision {
		t.Fatalf("count bound lost complete census or latest row: returned=%d rows_read=%d truncated=%t", countBound.Returned, countBound.RowsRead, countBound.Truncated)
	}
	firstLine := bytes.IndexByte(content, '\n') + 1
	capBytes := firstLine + firstLine/2
	var byteBound nativeperfCLIResponse
	nativeperfCLIRun(t, 0, &byteBound, "decisions", "--ledger", ledger, "--limit", "3", "--max-bytes", strconv.Itoa(capBytes), "--json")
	if len(byteBound.Decisions) != 1 || byteBound.RowsRead != 1 || !byteBound.ByteTruncated || !byteBound.Truncated || byteBound.TotalRows != nil || byteBound.OmittedRows != nil || byteBound.ByteLimit != capBytes || byteBound.BytesRead < capBytes || byteBound.BytesRead > capBytes+1 || byteBound.WindowStart <= 0 || byteBound.OmittedBytes <= 0 || byteBound.FileBytes != len(content) || byteBound.Decisions[0].DefaultStateReadback.Revision != latestRevision {
		t.Fatalf("byte-limited tail hid the latest row or overstated unseen history: rows_read=%d bytes_read=%d window_start=%d file_bytes=%d", byteBound.RowsRead, byteBound.BytesRead, byteBound.WindowStart, byteBound.FileBytes)
	}
	var exact nativeperfCLIResponse
	nativeperfCLIRun(t, 0, &exact, "decisions", "--ledger", ledger, "--limit", "100", "--max-bytes", strconv.Itoa(len(content)), "--json")
	if exact.Truncated || exact.Returned != rowCount || exact.TotalRows == nil || *exact.TotalRows != rowCount || exact.BytesRead != len(content) || exact.Present == nil || !*exact.Present {
		t.Fatalf("exact byte boundary was not complete: returned=%d rows_read=%d truncated=%t", exact.Returned, exact.RowsRead, exact.Truncated)
	}
	missing := filepath.Join(dir, "missing.jsonl")
	var empty nativeperfCLIResponse
	nativeperfCLIRun(t, 0, &empty, "decisions", "--ledger", missing, "--json")
	if empty.Present == nil || *empty.Present || empty.Returned != 0 || empty.TotalRows != nil || empty.OmittedRows != nil {
		t.Fatal("missing ledger was reported as an observed empty file")
	}
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Fatalf("reader created a missing ledger: %v", err)
	}
	if err := os.WriteFile(missing, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	nativeperfCLIRun(t, 0, &empty, "decisions", "--ledger", missing, "--json")
	if empty.Present == nil || !*empty.Present || empty.Returned != 0 || empty.TotalRows == nil || *empty.TotalRows != 0 {
		t.Fatal("existing empty ledger lost its observed presence")
	}
	nativeperfCLIRun(t, 1, nil, "decisions", "--ledger", ledger, "--max-bytes", "1", "--json")
	if !bytes.Equal(content, mustReadExperimentFile(t, ledger)) {
		t.Fatal("bounded readers mutated the ledger")
	}
	for _, args := range [][]string{{"--limit", "0"}, {"--limit", "101"}, {"--max-bytes", "0"}, {"--max-bytes", strconv.Itoa(16<<20 + 1)}} {
		nativeperfCLIRun(t, 2, nil, append([]string{"decisions", "--ledger", ledger, "--json"}, args...)...)
	}
	for _, bad := range []string{string(content) + "{}\n", strings.TrimSuffix(string(content), "\n"), strings.TrimSuffix(string(content), "\n") + " ]\n", strings.TrimSuffix(string(content), "\n") + " }\n"} {
		if err := os.WriteFile(ledger, []byte(bad), 0o600); err != nil {
			t.Fatal(err)
		}
		nativeperfCLIRun(t, 1, nil, "decisions", "--ledger", ledger, "--json")
		if !bytes.Equal([]byte(bad), mustReadExperimentFile(t, ledger)) {
			t.Fatal("invalid journal reader mutated evidence")
		}
	}
}

type nativeperfCLIResponse struct {
	EvidenceScope      string                                  `json:"evidence_scope"`
	HardwareVerified   *bool                                   `json:"hardware_verified"`
	DefaultObservation string                                  `json:"default_observation"`
	ServingMutated     *bool                                   `json:"serving_mutated"`
	Journal            experiments.NativePerfDecisionJournal   `json:"journal"`
	Replayed           *bool                                   `json:"replayed"`
	Decisions          []experiments.NativePerfDecisionJournal `json:"decisions"`
	Present            *bool                                   `json:"present"`
	RowsRead           int                                     `json:"rows_read"`
	Returned           int                                     `json:"returned"`
	RowLimit           int                                     `json:"row_limit"`
	BytesRead          int                                     `json:"bytes_read"`
	ByteLimit          int                                     `json:"byte_limit"`
	WindowStart        int                                     `json:"window_start"`
	FileBytes          int                                     `json:"file_bytes"`
	OmittedBytes       int                                     `json:"omitted_bytes"`
	ByteTruncated      bool                                    `json:"byte_truncated"`
	RowTruncated       bool                                    `json:"row_truncated"`
	Truncated          bool                                    `json:"truncated"`
	TotalRows          *int                                    `json:"total_rows"`
	OmittedRows        *int                                    `json:"omitted_rows"`
}

func nativeperfCLIRun(t *testing.T, wantCode int, result any, args ...string) []byte {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runExperiments(&stdout, &stderr, append([]string{"native-performance"}, args...))
	if code != wantCode {
		t.Fatalf("native-performance %s exit=%d, want %d; stderr=%s stdout=%s", args[0], code, wantCode, &stderr, &stdout)
	}
	if result != nil {
		if err := json.Unmarshal(stdout.Bytes(), result); err != nil {
			t.Fatalf("decode native-performance response: %v; output=%s", err, &stdout)
		}
	}
	return stdout.Bytes()
}

func nativeperfCLIAssertScope(t *testing.T, got nativeperfCLIResponse) {
	t.Helper()
	if got.EvidenceScope != "supplied-files" || got.DefaultObservation != "caller-supplied" || got.HardwareVerified == nil || *got.HardwareVerified || got.ServingMutated == nil || *got.ServingMutated {
		t.Fatalf("file evidence overstated observation or effects: %+v", got)
	}
}

// This synthetic policy fixture exercises CLI wiring; it is not a compiled onboarding graph or a physical performance witness.
func nativeperfCLIRequest(t *testing.T, reject bool) experiments.NativePerfTransactionRequest {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "internal", "nativeperf", "testdata", "native-performance-profile", "synthetic-cuda-bandwidth-bound.json"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := nativeperf.DecodeProfile(data)
	if err != nil {
		t.Fatal(err)
	}
	classification, err := nativeperf.ClassifyProfile(nativeperf.ActiveGraph(), profile)
	if err != nil {
		t.Fatal(err)
	}
	lever := classification.RecommendedLeverID
	graph := newmodel.NativeObligationGraph{
		Schema: "fak-native-obligation-graph/v1", Engine: "fak-native",
		ManifestDigest: cmdTestDigest("synthetic manifest"), DescriptorDigest: cmdTestDigest("synthetic descriptor"), EnvelopeDigest: cmdTestDigest("synthetic envelope"),
		Nodes: []newmodel.NativeObligation{{ID: "synthetic-obligation", Class: "fusion", Operation: lever, Eligible: true,
			Backend:          newmodel.NativeBackendObligation{Engine: "fak-native", Platform: "cuda", Backend: "cuda", Operation: lever},
			PromotionWitness: newmodel.NativePromotionWitness{ID: "synthetic-candidate", Kind: "nativeperf-gate"}}},
	}
	graphBytes, err := json.Marshal(graph)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
	if err != nil {
		t.Fatal(err)
	}
	baseline.Role, baseline.Revision = nativeperf.RoleBaseline, "synthetic-baseline"
	baseline.Machine.ScrubbedID = "synthetic-host"
	baseline.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
	baseline.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
	baseline.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "synthetic-r1"}}
	baseline.Commands = []string{"synthetic fixture; no hardware command executed"}
	baseline.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "synthetic-profile.json", SHA256: strings.Repeat("a", 64)}}
	for i := range baseline.Repetitions {
		baseline.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 100 + float64(i%2)}
	}
	candidate := baseline
	candidate.Role, candidate.Revision = nativeperf.RoleCandidate, "synthetic-candidate"
	candidate.ChangedAxes = []string{"lever:" + lever}
	candidate.Repetitions = append([]nativeperf.Repetition(nil), baseline.Repetitions...)
	readback := experiments.DefaultReadback{Selected: "candidate", Revision: candidate.Revision}
	if reject {
		for i := range candidate.Repetitions {
			candidate.Repetitions[i].TokensPerSecond = 85
		}
		readback = experiments.DefaultReadback{Selected: "baseline", Revision: baseline.Revision}
	}
	return experiments.NativePerfTransactionRequest{Profile: profile, ObligationGraph: graph, ObligationGraphDigest: cmdTestDigest(string(graphBytes)),
		ObligationID: "synthetic-obligation", CandidateID: "synthetic-candidate", DefaultReadback: readback,
		GateRequest: nativeperf.GateRequest{Schema: nativeperf.GateRequestSchema, LastAccepted: baseline, Candidate: candidate,
			Policy: nativeperf.GatePolicy{Schema: nativeperf.GatePolicySchema, EnvelopeID: baseline.EnvelopeID, ChangedLeverID: lever,
				AcceptedRevision: baseline.Revision, MinimumRepetitions: 3, MaximumNoisePercent: 2, InvestigateDropPercent: 2, RegressionDropPercent: 5,
				MinimumThroughput: 90, MaximumPeakBytes: 1200, QualityMetric: "exact_match", MinimumQualityScore: 1, QualityHigherIsBetter: true,
				RequiredEngine: "fak-native", RequiredForwardPath: baseline.Execution.ForwardPath}}}
}
