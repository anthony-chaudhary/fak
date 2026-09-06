package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var (
	benchResultSink          Result
	benchStringSink          string
	benchClassificationSink  Classification
	benchReplayVerdictSink   ReplayVerdict
	benchFailureBundleSink   FailureBundle
	benchAuditResultSink     ThresholdAuditResult
	benchSuitePlanSink       SuitePlan
	benchReleaseDecisionSink ReleaseDecision
	benchCaseSink            QualityCase
	benchVerdictSink         Verdict
)

func benchValidCase(id, tier, family string, cost CostSpec) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  "benchmark case prompt",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 2},
		Reference: Trace{
			Runner: "reference",
			Tokens: []string{"ok"},
			Text:   "ok",
		},
		Oracles: []string{"greedy-token-diff"},
		Metadata: CaseMetadata{
			Model:     Revision{Name: "model-bench", Revision: "sha256:model-v1"},
			Tokenizer: Revision{Name: "tok-bench", Revision: "sha256:tok-v1"},
			Engine:    EngineSpec{Name: "fak", Backend: "cpu", Flags: map[string]string{"dtype": "f32"}},
			Code:      Revision{Name: "github.com/anthony-chaudhary/fak", Revision: "git:bench"},
			Oracle:    OracleEvidence{Kind: "exact-greedy-trace", Revision: "sha256:oracle-v1"},
			Tolerance: ToleranceSpec{Metric: "exact-token", Revision: "policy:v1"},
			Baseline:  BaselineSpec{ID: "baseline-bench", Revision: "sha256:base-v1"},
			Tier:      TierSpec{Name: tier},
			Cost:      cost,
			Owner:     "quality-team",
			Family:    family,
		},
	}
}

func benchReleaseProvenance(rev string) EvidenceProvenance {
	return EvidenceProvenance{
		Model:     "model-r1",
		Tokenizer: "tok-r1",
		Engine:    "cpu/deterministic",
		Seed:      7,
		Oracle:    "exact-r1",
		Revision:  rev,
		Baseline:  "base-r1/tol-exact",
	}
}

func BenchmarkRunCase_Clean(b *testing.B) {
	c := DemoCase()
	ref := ReferenceRunner{}
	eng := DemoEngine("")
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := RunCase(c, ref, eng, oracles)
		if err != nil {
			b.Fatalf("RunCase: %v", err)
		}
		benchResultSink = res
	}
}

func BenchmarkRunCase_DefectDecode(b *testing.B) {
	c := DemoCase()
	ref := ReferenceRunner{}
	eng := DemoEngine("decode")
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := RunCase(c, ref, eng, oracles)
		if err != nil {
			b.Fatalf("RunCase: %v", err)
		}
		benchResultSink = res
	}
}

func BenchmarkRunCase_DefectReport(b *testing.B) {
	c := DemoCase()
	ref := ReferenceRunner{}
	eng := DemoEngine("report")
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := RunCase(c, ref, eng, oracles)
		if err != nil {
			b.Fatalf("RunCase: %v", err)
		}
		benchResultSink = res
	}
}

func BenchmarkExplain_Pass(b *testing.B) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Explain(res)
	}
}

func BenchmarkExplain_Fail(b *testing.B) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Explain(res)
	}
}

func BenchmarkClassify(b *testing.B) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchClassificationSink = Classify(res)
	}
}

func BenchmarkReplay(b *testing.B) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}
	if res.FailureBundle == nil {
		b.Fatal("expected failure bundle")
	}
	bundle := *res.FailureBundle

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReplayVerdictSink = Replay(bundle)
	}
}

func BenchmarkLoadBundle(b *testing.B) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}
	if res.FailureBundle == nil {
		b.Fatal("expected failure bundle")
	}
	raw, err := json.Marshal(res.FailureBundle)
	if err != nil {
		b.Fatalf("marshal bundle: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loaded, err := LoadBundle(raw)
		if err != nil {
			b.Fatalf("LoadBundle: %v", err)
		}
		benchFailureBundleSink = loaded
	}
}

func BenchmarkLoadCase(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "case_v1.json"))
	if err != nil {
		b.Fatalf("read testdata: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := LoadCase(raw)
		if err != nil {
			b.Fatalf("LoadCase: %v", err)
		}
		benchCaseSink = c
	}
}

func BenchmarkValidateCanonicalCase(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "case_v1.json"))
	if err != nil {
		b.Fatalf("read testdata: %v", err)
	}
	c, err := LoadCase(raw)
	if err != nil {
		b.Fatalf("LoadCase: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.ValidateCanonical(); err != nil {
			b.Fatalf("ValidateCanonical: %v", err)
		}
	}
}

func BenchmarkAuditThreshold(b *testing.B) {
	for _, n := range []int{10, 100, 1000} {
		obs := make([]float64, n)
		for i := 0; i < n; i++ {
			obs[i] = 0.5 + float64(i%20)*0.01
		}
		width := 0.05
		decimals := 4
		perturb := 0.001
		req := ThresholdAuditRequest{
			Threshold:              0.55,
			Comparison:             "at_least",
			Observations:           obs,
			BoundaryWidth:          &width,
			RoundTripDecimalPlaces: &decimals,
			Perturbation:           &perturb,
		}

		b.Run(fmt.Sprintf("obs_%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchAuditResultSink = AuditThreshold(req)
			}
		})
	}
}

func BenchmarkSplitSuite(b *testing.B) {
	cases := make([]QualityCase, 30)
	for i := 0; i < len(cases); i++ {
		tier := "pr"
		family := "deterministic"
		cpu := 1
		mem := int64(32)
		runtime := int64(1 + i%5)
		timeout := runtime * 10
		accel := 0
		switch i % 3 {
		case 1:
			tier = "nightly"
			family = "statistics"
			runtime = 120
			timeout = 600
			mem = 512
		case 2:
			tier = "release"
			family = "gpu_parity"
			runtime = 600
			timeout = 1800
			mem = 2048
			accel = 1
		}
		cases[i] = benchValidCase(fmt.Sprintf("bench-case-%02d", i), tier, family, CostSpec{
			RuntimeSeconds: runtime,
			TimeoutSeconds: timeout,
			CPU:            cpu,
			MemoryMiB:      mem,
			Accelerators:   accel,
		})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSuitePlanSink = SplitSuites(cases, nil)
	}
}

func BenchmarkQualifyRelease(b *testing.B) {
	rev := "artifact-v1"
	oracles, err := Lookup(DemoCase().Oracles)
	if err != nil {
		b.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(DemoCase(), ReferenceRunner{}, DemoEngine(""), oracles)
	if err != nil {
		b.Fatalf("RunCase: %v", err)
	}
	prov := benchReleaseProvenance(rev)
	ev := []Evidence{
		EvidenceFromResult(prov, res),
	}
	gates := []RequiredGate{
		{CaseID: DemoCase().ID, Kind: KindDeterministic, Tier: TierRelease, CostSeconds: 1},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReleaseDecisionSink = QualifyRelease(rev, gates, ev)
	}
}

func BenchmarkGreedyTokenDiff_Judge(b *testing.B) {
	for _, size := range []int{8, 64, 256} {
		tokens := make([]string, size)
		for i := 0; i < size; i++ {
			tokens[i] = fmt.Sprintf("token_%d", i)
		}
		ref := Trace{Tokens: tokens, Text: strings.Join(tokens, " ")}
		eng := Trace{Tokens: append([]string(nil), tokens...), Text: ref.Text}
		oracle := GreedyTokenDiff{}
		c := QualityCase{}

		b.Run(fmt.Sprintf("tokens_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchVerdictSink = oracle.Judge(ref, eng, c)
			}
		})
	}
}

func BenchmarkGroundingRubric_Judge(b *testing.B) {
	c := QualityCase{
		Rubric: RubricSpec{
			Required:  []string{"revenue", "growth", "q3", "margin", "guidance"},
			Forbidden: []string{"bankruptcy", "restatement"},
			MinScore:  0.8,
		},
	}
	eng := Trace{
		Text: "In Q3 the company achieved strong revenue growth and expanded operating margin in line with guidance.",
	}
	oracle := GroundingRubric{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(Trace{}, eng, c)
	}
}

func BenchmarkToolCallFidelity_Judge(b *testing.B) {
	specJSON := `{"tool":"get_weather","args":{` +
		`"city":{"type":"string","required":true},` +
		`"days":{"type":"integer","required":true,"min":1,"max":10},` +
		`"units":{"type":"string","enum":["celsius","fahrenheit"]}}}`
	callJSON := `{"tool":"get_weather","args":{"city":"Oslo","days":3,"units":"celsius"}}`
	c := QualityCase{
		Schema:    CaseSchema,
		ID:        "bench-tool-call",
		Version:   1,
		Prompt:    "Check weather",
		Reference: Trace{Text: specJSON},
		Rubric:    RubricSpec{MinScore: 1.0},
	}
	oracle := toolCallFidelity{}
	ref := Trace{}
	eng := Trace{Text: callJSON}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}

func BenchmarkIncrementalUnicode_Judge(b *testing.B) {
	c := QualityCase{
		Reference: Trace{
			Tokens: []string{"He", string([]byte{0xE4, 0xB8}), string([]byte{0x96}) + "llo", "!"},
			Text:   "He\u4E16llo!",
		},
	}
	ref := c.Reference
	eng := Trace{
		Tokens: []string{"He", string([]byte{0xE4, 0xB8}), string([]byte{0x96}) + "llo", "!"},
		Text:   "He\u4E16llo!",
	}
	oracle := IncrementalUnicode{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}

func BenchmarkTopKTopPBoundary_Judge(b *testing.B) {
	c := TopKTopPCase("bench-topkp",
		[]string{"alpha", "beta", "gamma", "delta"},
		[]float64{math.Log(4), math.Log(3), math.Log(2), math.Log(1)},
		2, 0.7)
	ref := c.Reference
	eng := Trace{Tokens: []string{"alpha", "beta"}, Text: "alpha beta"}
	oracle := TopKTopPBoundary{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}

func BenchmarkBatchInvariance_Judge(b *testing.B) {
	c := BatchInvarianceCase()
	refRunner := BatchInvarianceRunner{}
	engRunner := BatchInvarianceEngine("")
	ref, err := refRunner.Run(c)
	if err != nil {
		b.Fatalf("refRunner: %v", err)
	}
	eng, err := engRunner.Run(c)
	if err != nil {
		b.Fatalf("engRunner: %v", err)
	}
	oracle := BatchInvariance{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}

func BenchmarkConfidenceCalibration_Judge(b *testing.B) {
	groundTruth := `[
  {"claim":"Throughput increased 12% week over week","support":"strong"},
  {"claim":"Latency may improve next quarter","support":"weak"},
  {"claim":"Churn is trending down","support":"none"}
]`
	faithfulClaims := `[
  {"claim":"Throughput increased 12% week over week","confidence":"high"},
  {"claim":"Latency may improve next quarter","confidence":"low"},
  {"claim":"Churn is trending down","confidence":"low"}
]`
	c := QualityCase{
		Schema:    CaseSchema,
		ID:        "bench-calibration",
		Version:   1,
		Reference: Trace{Text: groundTruth},
		Rubric:    RubricSpec{MinScore: 1.0},
	}
	ref := Trace{}
	eng := Trace{Text: faithfulClaims}
	oracle := ConfidenceCalibration{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}

func BenchmarkDecisionRelevance_Judge(b *testing.B) {
	reportText := "# Priority decisions\n" +
		"Decision: adopt the usage-based pricing model for Q4 renewals.\n" +
		"Decision: freeze hiring for the platform team until revenue recovers.\n" +
		"\n" +
		"# Operational notes\n" +
		"Office plants were rotated to the south-facing windows.\n"
	c := QualityCase{
		Schema:  CaseSchema,
		ID:      "bench-decision-relevance",
		Version: 1,
		Rubric: RubricSpec{
			Required: []string{
				"decision: adopt the usage-based pricing model",
				"decision: freeze hiring",
			},
			MinScore: 1.0,
		},
	}
	oracle := drDecisionRelevance{}
	ref := Trace{}
	eng := Trace{Text: reportText}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = oracle.Judge(ref, eng, c)
	}
}
