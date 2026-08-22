package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDeterminismAcrossFixturesAndRefusals(t *testing.T) {
	manifest := testManifest(t)
	for _, fixture := range []string{"known-cliff.jsonl", "missing-cache-evidence.jsonl"} {
		t.Run("fixture "+fixture, func(t *testing.T) {
			observations, err := readObservations(filepath.Join("testdata", fixture))
			if err != nil {
				t.Fatal(err)
			}
			first, err := Analyze(manifest, observations)
			if err != nil {
				t.Fatal(err)
			}
			observations, err = readObservations(filepath.Join("testdata", fixture))
			if err != nil {
				t.Fatal(err)
			}
			second, err := Analyze(manifest, observations)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first, second) {
				t.Fatalf("repeated analysis differs:\nfirst:  %+v\nsecond: %+v", first, second)
			}
		})
	}

	t.Run("byte-stable captured witness", func(t *testing.T) {
		first, err := BuildSelfcheck(manifest)
		if err != nil {
			t.Fatal(err)
		}
		second, err := BuildSelfcheck(manifest)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Fatalf("repeated selfcheck differs:\nfirst:  %+v\nsecond: %+v", first, second)
		}
		firstBytes, err := json.MarshalIndent(first, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		secondBytes, err := json.MarshalIndent(second, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		firstBytes, secondBytes = append(firstBytes, '\n'), append(secondBytes, '\n')
		if !bytes.Equal(firstBytes, secondBytes) {
			t.Fatal("deep-equal selfchecks did not encode to byte-identical JSON")
		}
		captured, err := os.ReadFile(filepath.Join("testdata", "known-cliff-witness.json"))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(firstBytes, captured) {
			t.Fatal("deterministic selfcheck drifted from the checked-in witness")
		}
	})

	for _, refusal := range []struct {
		name   string
		want   string
		mutate func([]Observation) []Observation
	}{
		{
			name: "pin drift",
			want: "pins differ",
			mutate: func(observations []Observation) []Observation {
				observations[0].Pins.RuntimeRevision = "different-runtime"
				return observations
			},
		},
		{
			name: "incomplete envelope",
			want: "observed envelope incomplete",
			mutate: func(observations []Observation) []Observation {
				filtered := make([]Observation, 0, len(observations))
				for _, observation := range observations {
					if observation.PrefixDepthTokens != 16384 {
						filtered = append(filtered, observation)
					}
				}
				return filtered
			},
		},
	} {
		t.Run("refusal "+refusal.name, func(t *testing.T) {
			var failures [2]string
			for run := range failures {
				observations, err := SyntheticObservations(manifest, true)
				if err != nil {
					t.Fatal(err)
				}
				_, err = Analyze(manifest, refusal.mutate(observations))
				requireRecovery(t, err, refusal.want)
				failures[run] = err.Error()
			}
			if failures[0] != failures[1] {
				t.Fatalf("repeated refusal differs:\nfirst:  %s\nsecond: %s", failures[0], failures[1])
			}
		})
	}

	t.Log("witnessed 2 fixture variants, 2 guarded refusals, deep-equal reports, and byte-stable captured JSON")
}

func TestSelfcheckFindsKnownCliffAndRefusesMissingEvidence(t *testing.T) {
	manifest := testManifest(t)
	witness, err := BuildSelfcheck(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if got := witness.KnownCliff.Boundary.DeepestReliablePrefixTokens; got == nil || *got != 8192 {
		t.Fatalf("deepest reusable prefix = %v, want 8192", got)
	}
	cliff := witness.KnownCliff.Boundary.Cliff
	if cliff == nil || cliff.ReliableThroughTokens != 8192 || cliff.UnreliableAtTokens != 12288 {
		t.Fatalf("cliff = %+v, want 8192..12288", cliff)
	}
	if witness.KnownCliff.PressureRecovery.Status != "recovered" {
		t.Fatalf("pressure recovery = %+v", witness.KnownCliff.PressureRecovery)
	}
	if witness.MissingKVData.Boundary.Status != "unknown" || witness.MissingKVData.Boundary.DeepestReliablePrefixTokens != nil {
		t.Fatalf("missing evidence invented a boundary: %+v", witness.MissingKVData.Boundary)
	}
	for _, point := range witness.MissingKVData.DepthCurve {
		if point.PrefillSavedTokens != nil || point.ReuseRatio != nil || point.KVEvidenceSamples != 0 {
			t.Fatalf("missing evidence became zero-valued evidence: %+v", point)
		}
	}
}

func TestManifestAndObservedEnvelopeMeetAcceptance(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	report, err := Analyze(manifest, observations)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Axes.PrefixDepthTokens) < 6 || len(manifest.Axes.SuffixPatterns) < 2 || manifest.Axes.Repetitions < 3 {
		t.Fatalf("manifest target envelope is too small: %+v", manifest.Axes)
	}
	if !report.Envelope.Complete || report.Envelope.PrefixDepths != 6 || report.Envelope.SuffixPatterns != 2 || report.Envelope.TurnCounts != 2 || report.Envelope.ConcurrencyValues != 2 || report.Envelope.PressurePhases != 3 || report.Envelope.MinimumRepetitionsPerArm != 3 || !report.Envelope.Counterbalanced {
		t.Fatalf("observed envelope = %+v", report.Envelope)
	}
	if report.Evidence.SemanticPromptEqual == report.Evidence.WarmRequests {
		t.Fatalf("suffix churn did not separate semantic equality: %+v", report.Evidence)
	}
	if report.Evidence.TokenPrefixEqual != report.Evidence.WarmRequests || report.Evidence.BackendStatusPresent == 0 || report.Evidence.ReuseMeasurementPresent == 0 {
		t.Fatalf("evidence dimensions were conflated or omitted: %+v", report.Evidence)
	}
}

func TestAnalyzeRejectsPinDrift(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	observations[0].Pins.RuntimeRevision = "different-runtime"
	if _, err := Analyze(manifest, observations); err == nil || !strings.Contains(err.Error(), "pins differ") {
		t.Fatalf("pin drift error = %v", err)
	}
}

func TestAnalyzeRejectsOrderingThatLetsWarmupChooseTheResult(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := range observations {
		if observations[i].Repetition == 2 && observations[i].ThermalState == "cold" {
			observations[i].OrderIndex = 1
		}
	}
	if _, err := Analyze(manifest, observations); err == nil || !strings.Contains(err.Error(), "observed envelope incomplete") {
		t.Fatalf("counterbalance error = %v", err)
	}
}

func TestRequestLevelFixturesRemainReplayable(t *testing.T) {
	manifest := testManifest(t)
	for _, fixture := range []struct {
		name       string
		wantStatus string
	}{
		{name: "known-cliff.jsonl", wantStatus: "known"},
		{name: "missing-cache-evidence.jsonl", wantStatus: "unknown"},
	} {
		observations, err := readObservations(filepath.Join("testdata", fixture.name))
		if err != nil {
			t.Fatal(err)
		}
		if len(observations) == 0 || observations[0].RequestID == "" || observations[0].PromptTokens == 0 || observations[0].TTFTMillis == 0 {
			t.Fatalf("%s did not preserve request-level evidence", fixture.name)
		}
		report, err := Analyze(manifest, observations)
		if err != nil {
			t.Fatal(err)
		}
		if report.Boundary.Status != fixture.wantStatus {
			t.Fatalf("%s boundary = %+v", fixture.name, report.Boundary)
		}
	}
}

func TestCapturedWitnessMatchesAnalyzer(t *testing.T) {
	manifest := testManifest(t)
	witness, err := BuildSelfcheck(manifest)
	if err != nil {
		t.Fatal(err)
	}
	want, err := json.MarshalIndent(witness, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	want = append(want, '\n')
	got, err := os.ReadFile(filepath.Join("testdata", "known-cliff-witness.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("captured witness drifted; regenerate with -emit-fixtures after reviewing the analyzer change")
	}
}

func TestRunSelfcheckUsesCheckedInManifest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-selfcheck"})
	if exit != 0 || stderr.Len() != 0 || !strings.Contains(stdout.String(), `"known_deepest_reusable_prefix_tokens": 8192`) || !strings.Contains(stdout.String(), `"status": "unknown"`) {
		t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
	}
}

func TestRunRequiresOneAnalysisMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if exit := run(&stdout, &stderr, []string{"-manifest", filepath.Join("testdata", "campaign.json")}); exit != 2 || !strings.Contains(stderr.String(), "usage:") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func TestManifestRefusalsNameRecovery(t *testing.T) {
	tests := []struct {
		name   string
		want   string
		mutate func(*Manifest)
	}{
		{name: "schema", want: "schema", mutate: func(m *Manifest) { m.Schema = "wrong" }},
		{name: "campaign id", want: "campaign_id", mutate: func(m *Manifest) { m.CampaignID = "" }},
		{name: "pins", want: "pins.backend", mutate: func(m *Manifest) { m.Pins.Backend = "" }},
		{name: "tokenization pins", want: "tokenization", mutate: func(m *Manifest) { m.Tokenization.TokenizerID = "" }},
		{name: "tokenization unit", want: "tokenization.unit", mutate: func(m *Manifest) { m.Tokenization.Unit = "characters" }},
		{name: "prefix depths", want: "prefix_depth_tokens", mutate: func(m *Manifest) { m.Axes.PrefixDepthTokens = []int{1, 2} }},
		{name: "suffix count", want: "suffix_patterns", mutate: func(m *Manifest) { m.Axes.SuffixPatterns = m.Axes.SuffixPatterns[:1] }},
		{name: "suffix values", want: "suffix_patterns", mutate: func(m *Manifest) { m.Axes.SuffixPatterns[0].ID = "" }},
		{name: "turn counts", want: "turn_counts", mutate: func(m *Manifest) { m.Axes.TurnCounts = []int{1} }},
		{name: "concurrency", want: "concurrency", mutate: func(m *Manifest) { m.Axes.Concurrency = []int{1} }},
		{name: "repetitions", want: "repetitions", mutate: func(m *Manifest) { m.Axes.Repetitions = 2 }},
		{name: "reference arm", want: "reference_arm", mutate: func(m *Manifest) { m.ReferenceArm.TurnCount = 999 }},
		{name: "pressure arm values", want: "pressure_arms", mutate: func(m *Manifest) { m.Axes.PressureArms[0].ID = "" }},
		{name: "pressure envelope", want: "pressure arms", mutate: func(m *Manifest) { m.Axes.PressureArms = m.Axes.PressureArms[:1] }},
		{name: "ordering strategy", want: "ordering strategy", mutate: func(m *Manifest) { m.Ordering.Strategy = "fixed" }},
		{name: "ordering values", want: "ordering must", mutate: func(m *Manifest) { m.Ordering.ThermalOrder = []string{"warm", "cold"} }},
		{name: "reset", want: "reset procedures", mutate: func(m *Manifest) { m.Reset.AfterPressure = "" }},
		{name: "confidence", want: "confidence", mutate: func(m *Manifest) { m.Confidence.Level = .90 }},
		{name: "useful work", want: "useful_work_rule", mutate: func(m *Manifest) { m.UsefulWorkRule = "" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest := testManifest(t)
			test.mutate(&manifest)
			requireRecovery(t, manifest.Validate(), test.want)
		})
	}
}

func TestObservationRefusalsNameRecovery(t *testing.T) {
	tests := []struct {
		name   string
		want   string
		mutate func([]Observation) []Observation
	}{
		{name: "empty", want: "at least one observation", mutate: func([]Observation) []Observation { return nil }},
		{name: "identity", want: "must match the campaign", mutate: func(o []Observation) []Observation { o[0].Schema = "wrong"; return o }},
		{name: "pins", want: "pins differ", mutate: func(o []Observation) []Observation { o[0].Pins.RuntimeRevision = "wrong"; return o }},
		{name: "coordinates", want: "outside the declared", mutate: func(o []Observation) []Observation { o[0].PrefixDepthTokens = 999; return o }},
		{name: "pair coordinates", want: "invalid repetition", mutate: func(o []Observation) []Observation { o[0].Repetition = 0; return o }},
		{name: "request measurements", want: "prompt_tokens", mutate: func(o []Observation) []Observation { o[0].PromptTokens = 1; return o }},
		{name: "reset", want: "reset_procedure", mutate: func(o []Observation) []Observation { o[0].ResetProcedure = "wrong"; return o }},
		{name: "negative cache signal", want: "non-negative", mutate: func(o []Observation) []Observation { value := int64(-1); o[0].KV.CachedInputTokens = &value; return o }},
		{name: "cache exceeds prompt", want: "cannot exceed", mutate: func(o []Observation) []Observation {
			value := o[0].PromptTokens + 1
			o[0].KV.CachedInputTokens = &value
			return o
		}},
		{name: "occupancy", want: "occupancy_ratio", mutate: func(o []Observation) []Observation { value := 1.1; o[0].KV.OccupancyRatio = &value; return o }},
		{name: "duplicate thermal request", want: "duplicate", mutate: func(o []Observation) []Observation { return append(o, o[0]) }},
		{name: "missing thermal request", want: "requires one cold and one warm", mutate: func(o []Observation) []Observation { return o[1:] }},
		{name: "incomplete envelope", want: "observed envelope incomplete", mutate: func(o []Observation) []Observation {
			filtered := o[:0]
			for _, observation := range o {
				if observation.PrefixDepthTokens != 16384 {
					filtered = append(filtered, observation)
				}
			}
			return filtered
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest := testManifest(t)
			observations, err := SyntheticObservations(manifest, true)
			if err != nil {
				t.Fatal(err)
			}
			_, err = Analyze(manifest, test.mutate(observations))
			requireRecovery(t, err, test.want)
		})
	}
}

func TestCommandErrorsNameRecovery(t *testing.T) {
	t.Run("flag parse", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if exit := run(&stdout, &stderr, []string{"-not-a-flag"}); exit != 2 {
			t.Fatalf("exit = %d, want 2", exit)
		}
		requireRecoveryText(t, stderr.String(), "flag provided but not defined")
	})
	t.Run("missing manifest", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if exit := run(&stdout, &stderr, []string{"-manifest", filepath.Join(t.TempDir(), "missing.json"), "-selfcheck"}); exit != 1 {
			t.Fatalf("exit = %d, want 1", exit)
		}
		requireRecoveryText(t, stderr.String(), "read manifest")
	})
	t.Run("invalid manifest json", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "manifest.json")
		if err := os.WriteFile(path, []byte("{"), 0o644); err != nil {
			t.Fatal(err)
		}
		requireRunRecovery(t, []string{"-manifest", path, "-selfcheck"}, "decode manifest")
	})
	t.Run("missing observations", func(t *testing.T) {
		requireRunRecovery(t, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-observations", filepath.Join(t.TempDir(), "missing.jsonl")}, "open observations")
	})
	t.Run("invalid observations json", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "observations.jsonl")
		if err := os.WriteFile(path, []byte("{\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		requireRunRecovery(t, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-observations", path}, "decode observations")
	})
	t.Run("write report", func(t *testing.T) {
		requireRunRecovery(t, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-selfcheck", "-output", t.TempDir()}, "write selfcheck")
	})
	t.Run("emit fixtures", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "not-a-directory")
		if err := os.WriteFile(path, []byte("occupied"), 0o644); err != nil {
			t.Fatal(err)
		}
		requireRunRecovery(t, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-emit-fixtures", path}, "emit fixtures")
	})
	t.Run("stdout", func(t *testing.T) {
		var stderr bytes.Buffer
		if exit := run(errorWriter{}, &stderr, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-selfcheck"}); exit != 1 {
			t.Fatalf("exit = %d, want 1", exit)
		}
		requireRecoveryText(t, stderr.String(), "write selfcheck")
	})
}

func TestSelfcheckErrorsNameRecovery(t *testing.T) {
	t.Run("required depths", func(t *testing.T) {
		manifest := testManifest(t)
		manifest.Axes.PrefixDepthTokens = []int{1024, 2048, 4096, 6144, 10240, 16384}
		_, err := SyntheticObservations(manifest, true)
		requireRecovery(t, err, "8192 and 12288")
	})
	tests := []struct {
		name   string
		want   string
		mutate func(*DepthReport, *DepthReport)
	}{
		{name: "known boundary", want: "8k/12k cliff", mutate: func(known, _ *DepthReport) { known.Boundary.Status = "unknown" }},
		{name: "pressure recovery", want: "recover after pressure", mutate: func(known, _ *DepthReport) { known.PressureRecovery.Status = "unknown" }},
		{name: "missing evidence", want: "invented a boundary", mutate: func(_, missing *DepthReport) {
			value := 8192
			missing.Boundary.Status = "known"
			missing.Boundary.DeepestReliablePrefixTokens = &value
		}},
		{name: "envelope", want: "envelope incomplete", mutate: func(known, _ *DepthReport) { known.Envelope.Complete = false }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			depth := 8192
			known := DepthReport{
				Boundary:         BoundaryFinding{Status: "known", DeepestReliablePrefixTokens: &depth, Cliff: &CliffInterval{ReliableThroughTokens: 8192, UnreliableAtTokens: 12288}},
				PressureRecovery: RecoveryFinding{Status: "recovered"},
				Envelope:         EnvelopeFinding{Complete: true},
			}
			missing := DepthReport{
				Boundary:         BoundaryFinding{Status: "unknown"},
				PressureRecovery: RecoveryFinding{Status: "unknown"},
				Envelope:         EnvelopeFinding{Complete: true},
			}
			test.mutate(&known, &missing)
			requireRecovery(t, validateSelfcheckReports(known, missing), test.want)
		})
	}
}

func TestIOErrorsNameRecovery(t *testing.T) {
	t.Run("strict decode", func(t *testing.T) {
		var value map[string]any
		requireRecovery(t, decodeStrict([]byte("{"), &value), "decode JSON")
	})
	t.Run("multiple values", func(t *testing.T) {
		var value map[string]any
		requireRecovery(t, decodeStrict([]byte("{} {}"), &value), "multiple JSON values")
	})
	t.Run("scanner limit", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "oversized.jsonl")
		if err := os.WriteFile(path, bytes.Repeat([]byte("x"), 4*1024*1024+1), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := readObservations(path)
		requireRecovery(t, err, "scan observations")
	})
	t.Run("json encode", func(t *testing.T) {
		requireRecovery(t, writeJSON(&bytes.Buffer{}, "", make(chan int)), "encode JSON report")
	})
	t.Run("json file", func(t *testing.T) {
		requireRecovery(t, writeJSON(&bytes.Buffer{}, t.TempDir(), map[string]bool{"ok": true}), "write output file")
	})
	t.Run("json stdout", func(t *testing.T) {
		requireRecovery(t, writeJSON(errorWriter{}, "", map[string]bool{"ok": true}), "write stdout")
	})
	t.Run("observation file", func(t *testing.T) {
		requireRecovery(t, writeObservationJSONL(t.TempDir(), []Observation{{Schema: observationSchema}}), "write observation JSONL")
	})
	for _, name := range []string{"known-cliff.jsonl", "missing-cache-evidence.jsonl", "known-cliff-witness.json"} {
		t.Run("fixture "+name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.Mkdir(filepath.Join(dir, name), 0o755); err != nil {
				t.Fatal(err)
			}
			requireRecovery(t, emitFixtures(dir, testManifest(t)), name)
		})
	}
}

func TestExplicitErrorsRequireRecovery(t *testing.T) {
	for _, path := range []string{"main.go", "model.go", "analyze.go", "fixture.go"} {
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || (selector.Sel.Name != "New" && selector.Sel.Name != "Errorf") {
				return true
			}
			pkg, ok := selector.X.(*ast.Ident)
			if !ok || (pkg.Name != "errors" && pkg.Name != "fmt") {
				return true
			}
			literal, ok := call.Args[0].(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				return true
			}
			message := strings.Trim(literal.Value, "`\"")
			if !strings.Contains(message, "recovery:") && !strings.Contains(message, "%w") {
				t.Errorf("%s explicit error %q neither names recovery nor wraps a recovery-bearing error", path, message)
			}
			return true
		})
	}
}

func requireRunRecovery(t *testing.T, args []string, want string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if exit := run(&stdout, &stderr, args); exit != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%q", exit, stderr.String())
	}
	requireRecoveryText(t, stderr.String(), want)
}

func requireRecovery(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected refusal containing %q", want)
	}
	requireRecoveryText(t, err.Error(), want)
}

func requireRecoveryText(t *testing.T, text, want string) {
	t.Helper()
	if !strings.Contains(text, want) || !strings.Contains(text, "recovery:") {
		t.Fatalf("failure = %q, want %q and an actionable recovery", text, want)
	}
}

type errorWriter struct{}

func (errorWriter) Write([]byte) (int, error) { return 0, errors.New("sink unavailable") }

func testManifest(t *testing.T) Manifest {
	t.Helper()
	manifest, err := readManifest(filepath.Join("testdata", "campaign.json"))
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}
