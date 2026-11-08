package optimize

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// smokeTrace is the repo-root frozen trace, addressed relative to this package
// dir (internal/optimize) — `go test` runs with CWD at the package.
const smokeTrace = "../../testdata/tau2/tau2-smoke.json"

var pinnedNow = time.Date(2026, 11, 7, 12, 0, 0, 0, time.UTC)

// syntheticGraph builds a minimal valid graph: one metal envelope, one banded
// rung, and three levers — an enabled base plus two dependency-ready
// unwitnessed candidates (synth.a, synth.b in graph order).
func syntheticGraph() nativeperf.Graph {
	env := nativeperf.Envelope{
		ID: "synth-p32-t64", Model: "m", ModelRevision: "r", ArtifactSHA256: "a", Quantization: "Q4_K_M",
		Hardware: "hw", MemoryGiB: 36, Engine: "fak-native", Backend: "metal", ForwardPath: "fp",
		PromptTokens: 32, DecodeTokens: 64, Repetitions: 3, Temperature: 0,
		ThroughputMetric: "end_to_end_decode_tokens_per_second", ComparabilityRule: "rule",
	}
	apply := nativeperf.Applicability{EnvelopeID: env.ID, Platform: "darwin/arm64", Backend: "metal"}
	issue := nativeperf.IssueRef{Number: 1, URL: "https://github.com/anthony-chaudhary/fak/issues/1"}
	base := func(id string, deps []string) nativeperf.Lever {
		return nativeperf.Lever{
			ID: id, Name: id, Applicability: apply, Enabled: true, Status: nativeperf.StatusPresent, DependencyIDs: deps,
			Expected:    nativeperf.PlanningEffect{Summary: "s", Classification: nativeperf.ClassHypothesis, Provenance: "p"},
			OwningIssue: issue, NextWitness: "w",
		}
	}
	cand := func(id string) nativeperf.Lever {
		l := base(id, []string{"synth.base"})
		l.Enabled = false
		l.Status = nativeperf.StatusPartial
		return l
	}
	return nativeperf.Graph{
		Schema:     nativeperf.Schema,
		Envelope:   env,
		Comparison: nativeperf.Throughput{TokensPerSecond: 6.9, Classification: nativeperf.ClassComparison, Engine: "llama.cpp b9828", Comparability: nativeperf.ComparabilityApproximate, ObservedOn: "2026-01-01", Provenance: "p"},
		Rungs: []nativeperf.Rung{{
			ID: "rung-a", Name: "A", Enabled: true, Status: nativeperf.StatusPresent,
			Expected:  nativeperf.ExpectedRange{FloorTokensPerSecond: 2, RoofTokensPerSecond: 4, Classification: nativeperf.ClassHypothesis, Provenance: "p"},
			NextIssue: issue,
		}},
		Features:  []nativeperf.Feature{{ID: "feat-a", Name: "A", RungID: "rung-a", Enabled: true, Status: nativeperf.StatusPresent, Observable: "obs"}},
		Envelopes: []nativeperf.Envelope{env},
		Levers:    []nativeperf.Lever{base("synth.base", nil), cand("synth.a"), cand("synth.b")},
	}
}

// withSyntheticBands points the planning-band table at the synthetic rung for
// the duration of the test.
func withSyntheticBands(t *testing.T) {
	t.Helper()
	old := planningRungs
	planningRungs = map[string]string{"synth.a": "rung-a", "synth.b": "rung-a"}
	t.Cleanup(func() { planningRungs = old })
}

func refusalClass(t *testing.T, err error) string {
	t.Helper()
	if err == nil {
		t.Fatalf("expected OptimizeRefusal, got nil error")
	}
	var r *OptimizeRefusal
	if !errors.As(err, &r) {
		t.Fatalf("error = %v, want *OptimizeRefusal", err)
	}
	return r.Class
}

func TestPlanSelectsReadyLeverAndEmitsCommands(t *testing.T) {
	withSyntheticBands(t)
	cfg := Config{Graph: syntheticGraph(), Now: func() time.Time { return pinnedNow }}
	OptimizePlan, err := BuildPlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if OptimizePlan.LeverID != "synth.a" || OptimizePlan.Envelope != "synth-p32-t64" || !OptimizePlan.DependencyReady {
		t.Fatalf("unexpected OptimizePlan selection: %+v", OptimizePlan)
	}
	if OptimizePlan.ExpectedFloor == nil || *OptimizePlan.ExpectedFloor != 2 || OptimizePlan.ExpectedRoof == nil || *OptimizePlan.ExpectedRoof != 4 {
		t.Fatalf("planning band = %v..%v, want 2..4", OptimizePlan.ExpectedFloor, OptimizePlan.ExpectedRoof)
	}
	wantMeasure := "fak optimize measure --lever synth.a --trace " + DefaultTracePath + " --out synth.a.json"
	wantVerify := "fak optimize verify --baseline synth.a.baseline.json --candidate synth.a.json --json"
	if OptimizePlan.MeasureCmd != wantMeasure || OptimizePlan.VerifyCmd != wantVerify {
		t.Fatalf("commands drifted: measure=%q verify=%q", OptimizePlan.MeasureCmd, OptimizePlan.VerifyCmd)
	}
	if OptimizePlan.GeneratedAt != pinnedNow.Format(time.RFC3339) {
		t.Fatalf("generated_at = %q", OptimizePlan.GeneratedAt)
	}
}

func TestPlanActiveGraphProjectsCommittedBand(t *testing.T) {
	OptimizePlan, err := BuildPlan(Config{Graph: nativeperf.ActiveGraph()})
	if err != nil {
		t.Fatal(err)
	}
	if OptimizePlan.LeverID != "metal.command-buffer-amortization" {
		t.Fatalf("next committed lever = %q", OptimizePlan.LeverID)
	}
	if OptimizePlan.ExpectedFloor == nil || *OptimizePlan.ExpectedFloor != 5 || OptimizePlan.ExpectedRoof == nil || *OptimizePlan.ExpectedRoof != 6.966061 {
		t.Fatalf("committed band = %v..%v, want 5..6.966061", OptimizePlan.ExpectedFloor, OptimizePlan.ExpectedRoof)
	}
}

func TestPlanRefusesWhenNoReadyLever(t *testing.T) {
	graph := syntheticGraph()
	for i := range graph.Levers {
		graph.Levers[i].Enabled = true
		graph.Levers[i].Status = nativeperf.StatusPresent
	}
	_, err := BuildPlan(Config{Graph: graph})
	if got := refusalClass(t, err); got != ClassNoReadyLever {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassNoReadyLever)
	}
}

func TestMeasureRefusesUnknownLever(t *testing.T) {
	cfg := Config{Graph: syntheticGraph()}
	_, err := Measure(context.Background(), cfg, "nope", smokeTrace, "")
	if got := refusalClass(t, err); got != ClassUnknownLever {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassUnknownLever)
	}
}

func TestMeasureRefusesMissingTrace(t *testing.T) {
	cfg := Config{Graph: syntheticGraph(), Mappings: map[string]string{"synth.a": "vdso"}}
	missing := filepath.Join(t.TempDir(), "absent.json")
	_, err := Measure(context.Background(), cfg, "synth.a", missing, "")
	if got := refusalClass(t, err); got != ClassTraceUnreadable {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassTraceUnreadable)
	}
}

func TestMeasureRefusesWithoutArmMapping(t *testing.T) {
	// The committed graph's levers are Metal/CUDA hardware levers: none maps to
	// an ablate runtime feature, so measure must refuse with the mapping class.
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	_, err := Measure(context.Background(), cfg, "metal.command-buffer-amortization", smokeTrace, "")
	if got := refusalClass(t, err); got != ClassNoArmMapping {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassNoArmMapping)
	}
}

func TestMeasureRefusesEnvGatedMapping(t *testing.T) {
	// An env-gated feature cannot be flipped in-process; mapping one must refuse
	// rather than measure a silent no-op.
	cfg := Config{Graph: syntheticGraph(), Mappings: map[string]string{"synth.a": "normgate"}}
	_, err := Measure(context.Background(), cfg, "synth.a", smokeTrace, "")
	if got := refusalClass(t, err); got != ClassNoArmMapping {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassNoArmMapping)
	}
}

func TestMeasureRunsMappedSweepAndBindsWorkload(t *testing.T) {
	cfg := Config{Graph: syntheticGraph(), Mappings: map[string]string{"synth.a": "vdso"}, Now: func() time.Time { return pinnedNow }}
	rec, err := Measure(context.Background(), cfg, "synth.a", smokeTrace, "")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(smokeTrace)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if rec.Schema != ReceiptSchema || rec.LeverID != "synth.a" || rec.EngineID != "mock" {
		t.Fatalf("OptimizeReceipt identity drifted: %+v", rec)
	}
	if rec.TraceSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("trace sha256 = %q", rec.TraceSHA256)
	}
	if rec.BaselineTokS <= 0 || rec.CandidateTokS <= 0 {
		t.Fatalf("throughput must be positive from real replays: %+v", rec)
	}
	if rec.Reps != 1 || rec.MeasuredAt != pinnedNow.Format(time.RFC3339) {
		t.Fatalf("reps/measured_at drifted: %+v", rec)
	}
}

func writeReceipt(t *testing.T, dir, name string, rec OptimizeReceipt) string {
	t.Helper()
	path := filepath.Join(dir, name)
	raw, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestVerifyAcceptRejectAndQualityFail(t *testing.T) {
	withSyntheticBands(t)
	cfg := Config{Graph: syntheticGraph()}
	dir := t.TempDir()
	baseRec := OptimizeReceipt{Schema: ReceiptSchema, LeverID: "synth.a", TraceSHA256: "sha-base", BaselineTokS: 3}
	candRec := OptimizeReceipt{Schema: ReceiptSchema, LeverID: "synth.a", TraceSHA256: "sha-base", CandidateTokS: 3.5}
	basePath := writeReceipt(t, dir, "base.json", baseRec)

	t.Run("accept", func(t *testing.T) {
		v, err := Verify(cfg, basePath, writeReceipt(t, dir, "cand.json", candRec), "")
		if err != nil {
			t.Fatal(err)
		}
		if v.Verdict != VerdictAccept || v.Quality != QualityAbsent || len(v.Reasons) == 0 {
			t.Fatalf("OptimizeVerdict = %+v", v)
		}
	})
	t.Run("reject below floor", func(t *testing.T) {
		weak := candRec
		weak.CandidateTokS = 1.5
		v, err := Verify(cfg, basePath, writeReceipt(t, dir, "weak.json", weak), "")
		if err != nil {
			t.Fatal(err)
		}
		if v.Verdict != VerdictReject {
			t.Fatalf("OptimizeVerdict = %+v", v)
		}
	})
	t.Run("quality fail forces reject", func(t *testing.T) {
		qpath := filepath.Join(dir, "quality.json")
		if err := os.WriteFile(qpath, []byte(`{"pass": false}`), 0o644); err != nil {
			t.Fatal(err)
		}
		v, err := Verify(cfg, basePath, writeReceipt(t, dir, "cand.json", candRec), qpath)
		if err != nil {
			t.Fatal(err)
		}
		if v.Verdict != VerdictReject || v.Quality != QualityFail {
			t.Fatalf("OptimizeVerdict = %+v", v)
		}
	})
}

func TestVerifyRefusesInvalidPairs(t *testing.T) {
	withSyntheticBands(t)
	cfg := Config{Graph: syntheticGraph()}
	dir := t.TempDir()
	baseRec := OptimizeReceipt{Schema: ReceiptSchema, LeverID: "synth.a", TraceSHA256: "sha-1", BaselineTokS: 3}
	candRec := OptimizeReceipt{Schema: ReceiptSchema, LeverID: "synth.a", TraceSHA256: "sha-1", CandidateTokS: 3.5}
	basePath := writeReceipt(t, dir, "base.json", baseRec)

	cases := []struct {
		name string
		mut  func(*OptimizeReceipt)
	}{
		{"different lever", func(r *OptimizeReceipt) { r.LeverID = "synth.b" }},
		{"different workload", func(r *OptimizeReceipt) { r.TraceSHA256 = "sha-2" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mut := candRec
			tc.mut(&mut)
			_, err := Verify(cfg, basePath, writeReceipt(t, dir, "mut.json", mut), "")
			if got := refusalClass(t, err); got != ClassInvalidInput {
				t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
			}
		})
	}
	t.Run("unreadable input", func(t *testing.T) {
		_, err := Verify(cfg, filepath.Join(dir, "absent.json"), basePath, "")
		if got := refusalClass(t, err); got != ClassInvalidInput {
			t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
		}
	})
	t.Run("mixed kinds", func(t *testing.T) {
		foreign := filepath.Join(dir, "foreign.json")
		if err := os.WriteFile(foreign, []byte(`{"schema":"fak-native-performance-OptimizeReceipt/v3","role":"baseline"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := Verify(cfg, foreign, writeReceipt(t, dir, "cand.json", candRec), "")
		if got := refusalClass(t, err); got != ClassInvalidInput {
			t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
		}
	})
}

// nativeReceipt builds a structurally valid ExperimentReceipt pair member for
// the committed graph's first ready lever, mirroring the fixture shape the
// nativeperf OptimizeReceipt tests use.
func nativeReceipt(t *testing.T, role, lever string, tps float64) nativeperf.ExperimentReceipt {
	t.Helper()
	r, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
	if err != nil {
		t.Fatal(err)
	}
	r.Role, r.Revision, r.Machine.ScrubbedID = role, "rev-1", "lab-node-class-a"
	r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
	r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
	r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+gaaaaaaa"}}
	r.Commands = []string{"fak run-model --native --OptimizeReceipt-out OptimizeReceipt.json"}
	r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
	for i := range r.Repetitions {
		r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: tps}
	}
	if role == nativeperf.RoleCandidate {
		r.ChangedAxes = []string{"lever:" + lever}
	}
	return r
}

func writeNativeReceipt(t *testing.T, dir, name string, r nativeperf.ExperimentReceipt) string {
	t.Helper()
	path := filepath.Join(dir, name)
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestVerifyNativeperfCriterionAcceptsGainAboveAllGates(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	base := writeNativeReceipt(t, dir, "base.json", nativeReceipt(t, nativeperf.RoleBaseline, "metal.command-buffer-amortization", 100))
	cand := writeNativeReceipt(t, dir, "cand.json", nativeReceipt(t, nativeperf.RoleCandidate, "metal.command-buffer-amortization", 105))
	v, err := Verify(cfg, base, cand, "")
	if err != nil {
		t.Fatal(err)
	}
	if v.Verdict != VerdictAccept {
		t.Fatalf("OptimizeVerdict = %+v", v)
	}
}

func TestVerifyNativeperfCriterionRejectsRegression(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	base := writeNativeReceipt(t, dir, "base.json", nativeReceipt(t, nativeperf.RoleBaseline, "metal.command-buffer-amortization", 100))
	cand := writeNativeReceipt(t, dir, "cand.json", nativeReceipt(t, nativeperf.RoleCandidate, "metal.command-buffer-amortization", 90))
	v, err := Verify(cfg, base, cand, "")
	if err != nil {
		t.Fatal(err)
	}
	if v.Verdict != VerdictReject {
		t.Fatalf("OptimizeVerdict = %+v", v)
	}
}

func TestRunLoopDryRunStopsAfterPlan(t *testing.T) {
	withSyntheticBands(t)
	cfg := Config{Graph: syntheticGraph(), Mappings: map[string]string{"synth.a": "vdso", "synth.b": "vdso"}}
	lines, err := RunLoop(context.Background(), cfg, 3, true, smokeTrace)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 1 || lines[0].Verdict != LedgerPlanned || lines[0].LeverID != "synth.a" || lines[0].Metric != MetricDecodeTokS {
		t.Fatalf("dry-run ledger = %+v", lines)
	}
	if lines[0].Baseline != nil || lines[0].Candidate != nil || lines[0].DeltaPct != nil {
		t.Fatalf("dry-run line must not carry throughput fields: %+v", lines[0])
	}
}

func TestRunLoopRefusesNonPositiveIterations(t *testing.T) {
	cfg := Config{Graph: syntheticGraph()}
	for _, n := range []int{0, -1} {
		_, err := RunLoop(context.Background(), cfg, n, false, smokeTrace)
		if got := refusalClass(t, err); got != ClassNonPositiveIterations {
			t.Fatalf("maxIters=%d: OptimizeRefusal class = %q, want %q", n, got, ClassNonPositiveIterations)
		}
	}
}

func TestRunLoopStopsOnTypedRefusal(t *testing.T) {
	// Committed graph: the ready lever has no arm mapping, so iteration one
	// refuses and the loop stops without fabricating a measurement.
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	lines, err := RunLoop(context.Background(), cfg, 2, false, smokeTrace)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 1 || lines[0].Verdict != LedgerRefused || lines[0].Refusal != ClassNoArmMapping {
		t.Fatalf("ledger = %+v", lines)
	}
}

func TestMarkWitnessedAdvancesSelection(t *testing.T) {
	withSyntheticBands(t)
	graph := syntheticGraph()
	rec := &OptimizeReceipt{LeverID: "synth.a", TraceSHA256: "sha-x", BaselineTokS: 3, CandidateTokS: 3.5}
	markWitnessed(&graph, rec)
	if err := nativeperf.Validate(graph); err != nil {
		t.Fatal(err)
	}
	next, err := nativeperf.NextLever(graph)
	if err != nil {
		t.Fatal(err)
	}
	if next == nil || next.ID != "synth.b" {
		t.Fatalf("selection after witness = %+v, want synth.b", next)
	}
}

const rtx5090EnvelopeID = "qwen38-27b-q4km-rtx5090-p1-decode"

func TestPlanEnvelopeFilterSelectsRTX5090ReadyLever(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: rtx5090EnvelopeID, Now: func() time.Time { return pinnedNow }}
	plan, err := BuildPlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if plan.LeverID != "rtx5090.q8_1-activation-quant" || plan.Envelope != rtx5090EnvelopeID || !plan.DependencyReady {
		t.Fatalf("unexpected envelope-filtered plan: %+v", plan)
	}
	wantMeasure := "fak optimize measure --lever rtx5090.q8_1-activation-quant --envelope " + rtx5090EnvelopeID + " --trace " + DefaultTracePath + " --out rtx5090.q8_1-activation-quant.json"
	wantVerify := "fak optimize verify --baseline rtx5090.q8_1-activation-quant.baseline.json --candidate rtx5090.q8_1-activation-quant.json --envelope " + rtx5090EnvelopeID + " --json"
	if plan.MeasureCmd != wantMeasure || plan.VerifyCmd != wantVerify {
		t.Fatalf("commands drifted: measure=%q verify=%q", plan.MeasureCmd, plan.VerifyCmd)
	}
	if plan.GeneratedAt != pinnedNow.Format(time.RFC3339) {
		t.Fatalf("generated_at = %q", plan.GeneratedAt)
	}
}

func TestPlanEnvelopeFilterSelectsA100ReadyLever(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: "qwen38-27b-q4k-a100-p1-decode"}
	plan, err := BuildPlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if plan.LeverID != "cuda.q8_1-activation-quant" || plan.Envelope != "qwen38-27b-q4k-a100-p1-decode" {
		t.Fatalf("unexpected A100 filtered plan: %+v", plan)
	}
}

func TestPlanUnknownEnvelopeRefusesTyped(t *testing.T) {
	_, err := BuildPlan(Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: "no-such-envelope"})
	if got := refusalClass(t, err); got != ClassUnknownEnvelope {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassUnknownEnvelope)
	}
}

func TestPlanKnownEnvelopeWithoutReadyLeverRefuses(t *testing.T) {
	graph := syntheticGraph()
	envB := nativeperf.Envelope{
		ID: "synth-env-b", Model: "m", ModelRevision: "r", ArtifactSHA256: "a", Quantization: "Q4_K_M",
		Hardware: "hw", MemoryGiB: 36, Engine: "fak-native", Backend: "metal", ForwardPath: "fp",
		PromptTokens: 32, DecodeTokens: 64, Repetitions: 3, Temperature: 0,
		ThroughputMetric: "end_to_end_decode_tokens_per_second", ComparabilityRule: "rule",
	}
	applyB := nativeperf.Applicability{EnvelopeID: envB.ID, Platform: "darwin/arm64", Backend: "metal"}
	issue := nativeperf.IssueRef{Number: 1, URL: "https://github.com/anthony-chaudhary/fak/issues/1"}
	done := nativeperf.Lever{
		ID: "synth.done", Name: "done", Applicability: applyB, Enabled: true, Status: nativeperf.StatusPresent,
		Expected:    nativeperf.PlanningEffect{Summary: "s", Classification: nativeperf.ClassHypothesis, Provenance: "p"},
		OwningIssue: issue, NextWitness: "w",
	}
	graph.Envelopes = append(graph.Envelopes, envB)
	graph.Levers = append(graph.Levers, done)
	_, err := BuildPlan(Config{Graph: graph, EnvelopeID: envB.ID})
	if got := refusalClass(t, err); got != ClassNoReadyLever {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassNoReadyLever)
	}
}

// rtx5090ReceiptPair builds a minimal valid baseline/candidate ExperimentReceipt
// pair for a committed RTX 5090 lever, mirroring the fixture shape the
// nativeperf receipt tests use (BaselineTemplate plus filled capture fields).
func rtx5090ReceiptPair(t *testing.T, lever string, baseTPS, candTPS float64) []nativeperf.ExperimentReceipt {
	t.Helper()
	mk := func(role string, tps float64) nativeperf.ExperimentReceipt {
		r, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
		if err != nil {
			t.Fatal(err)
		}
		r.Role = role
		r.Revision = "rev-" + role
		r.Machine.ScrubbedID = "lab-node-class-a"
		r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
		r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
		r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+gaaaaaaa"}}
		r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
		r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
		for i := range r.Repetitions {
			r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: tps}
		}
		if role == nativeperf.RoleCandidate {
			r.ChangedAxes = []string{"lever:" + lever}
		}
		return r
	}
	return []nativeperf.ExperimentReceipt{mk(nativeperf.RoleBaseline, baseTPS), mk(nativeperf.RoleCandidate, candTPS)}
}

func writeReceiptPair(t *testing.T, dir, name string, pair []nativeperf.ExperimentReceipt) string {
	t.Helper()
	path := filepath.Join(dir, name)
	raw, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestMeasureReceiptAcceptsMatchedPair(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph(), Now: func() time.Time { return pinnedNow }}
	dir := t.TempDir()
	path := writeReceiptPair(t, dir, "pair.json", rtx5090ReceiptPair(t, "rtx5090.q8_1-activation-quant", 100, 105))
	rec, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", "", path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if rec.Schema != ReceiptSchema || rec.LeverID != "rtx5090.q8_1-activation-quant" || rec.EngineID != "fak-native" {
		t.Fatalf("OptimizeReceipt identity drifted: %+v", rec)
	}
	if rec.TracePath != path || rec.TraceSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("workload binding drifted: %+v", rec)
	}
	if rec.BaselineTokS != 100 || rec.CandidateTokS != 105 || rec.DeltaPct != 5 || rec.Reps != 3 {
		t.Fatalf("arm means drifted: %+v", rec)
	}
	if rec.Revision != "rev-candidate" || rec.ArtifactSHA256 != "issue-8635-pinned-artifact" || rec.ForwardPath != "cuda/q4km-p1-decode" {
		t.Fatalf("execution identity drifted: %+v", rec)
	}
	if rec.MeasuredAt != pinnedNow.Format(time.RFC3339) {
		t.Fatalf("measured_at = %q", rec.MeasuredAt)
	}
}

func TestMeasureReceiptRefusesLeverMismatch(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	path := writeReceiptPair(t, dir, "pair.json", rtx5090ReceiptPair(t, "rtx5090.dp4a-q4k-mmvq", 100, 105))
	_, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", "", path)
	if got := refusalClass(t, err); got != ClassReceiptLeverMismatch {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassReceiptLeverMismatch)
	}
}

func TestMeasureReceiptRefusesEnvelopeMismatch(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	pair := rtx5090ReceiptPair(t, "rtx5090.q8_1-activation-quant", 100, 105)
	for i := range pair {
		pair[i].EnvelopeID = "qwen38-27b-q4k-a100-p1-decode"
	}
	path := writeReceiptPair(t, dir, "pair.json", pair)
	_, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", "", path)
	if got := refusalClass(t, err); got != ClassReceiptEnvelopeMismatch {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassReceiptEnvelopeMismatch)
	}
}

func TestMeasureReceiptRefusesUndecodableFile(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	path := filepath.Join(dir, "garbage.json")
	if err := os.WriteFile(path, []byte(`{"not":"a receipt array"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", "", path)
	if got := refusalClass(t, err); got != ClassInvalidInput {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
	}
}

func TestMeasureReceiptRefusesMissingArm(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph()}
	dir := t.TempDir()
	pair := rtx5090ReceiptPair(t, "rtx5090.q8_1-activation-quant", 100, 105)
	path := writeReceiptPair(t, dir, "one-arm.json", pair[:1])
	_, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", "", path)
	if got := refusalClass(t, err); got != ClassInvalidInput {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
	}
}

func TestMeasureRefusesBothWorkloadSources(t *testing.T) {
	cfg := Config{Graph: syntheticGraph()}
	_, err := Measure(context.Background(), cfg, "synth.a", smokeTrace, "pair.json")
	if got := refusalClass(t, err); got != ClassInvalidInput {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
	}
}

func TestMeasureRefusesEnvelopeFlagMismatch(t *testing.T) {
	cfg := Config{Graph: nativeperf.ActiveGraph(), EnvelopeID: "qwen38-27b-q4k-a100-p1-decode"}
	_, err := Measure(context.Background(), cfg, "rtx5090.q8_1-activation-quant", smokeTrace, "")
	if got := refusalClass(t, err); got != ClassInvalidInput {
		t.Fatalf("OptimizeRefusal class = %q, want %q", got, ClassInvalidInput)
	}
}
