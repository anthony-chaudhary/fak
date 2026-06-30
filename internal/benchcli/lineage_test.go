package benchcli

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// A spliced lineage block must be the FIRST member, valid JSON, carry all four
// axes, and leave the caller's original members byte-for-byte intact.
func TestLineageIntoSplicesFirstMember(t *testing.T) {
	lin := Lineage{
		Schema: LineageSchema, AppVersion: "0.0.0", UTC: "2026-06-29T00:00:00Z",
		GitCommit: "deadbee", GoVersion: "go1.23.1", Node: "node-test",
	}
	report := map[string]any{"peak_tok_per_sec": 12.5, "model": "smollm2"}
	b, _ := json.MarshalIndent(report, "", "  ")

	got := lin.Into(b)

	// Still valid JSON, with both the report keys and the lineage block.
	var round map[string]json.RawMessage
	if err := json.Unmarshal(got, &round); err != nil {
		t.Fatalf("stamped report is not valid JSON: %v\n%s", err, got)
	}
	rawLin, ok := round["lineage"]
	if !ok {
		t.Fatalf("no lineage member after Into:\n%s", got)
	}
	if _, ok := round["model"]; !ok {
		t.Fatalf("original member 'model' lost after Into:\n%s", got)
	}
	var back Lineage
	if err := json.Unmarshal(rawLin, &back); err != nil {
		t.Fatalf("lineage block does not decode: %v", err)
	}
	if back != lin {
		t.Fatalf("lineage round-trip = %+v, want %+v", back, lin)
	}
	// lineage must be the first member (the run before its key has no other key).
	li := strings.Index(string(got), `"lineage"`)
	mi := strings.Index(string(got), `"model"`)
	pi := strings.Index(string(got), `"peak_tok_per_sec"`)
	if li > mi || li > pi {
		t.Fatalf("lineage is not the first member (lineage@%d model@%d peak@%d)", li, mi, pi)
	}
}

// Into is a no-op on a non-object, on a compact object with no newline to align
// to, and on a report that already carries lineage (idempotent — no double stamp).
func TestLineageIntoNoOpCases(t *testing.T) {
	lin := Stamp()
	cases := map[string][]byte{
		"array":   []byte("[\n  1,\n  2\n]"),
		"scalar":  []byte("42"),
		"compact": []byte(`{"a":1}`),
		"empty":   []byte(""),
	}
	for name, in := range cases {
		if got := lin.Into(in); string(got) != string(in) {
			t.Errorf("%s: Into mutated a non-spliceable input:\n%s", name, got)
		}
	}
	// Idempotent: a second stamp must not add a second lineage block.
	b, _ := json.MarshalIndent(map[string]any{"a": 1}, "", "  ")
	once := lin.Into(b)
	twice := lin.Into(once)
	if string(once) != string(twice) {
		t.Fatalf("Into is not idempotent:\nonce:\n%s\ntwice:\n%s", once, twice)
	}
	if n := strings.Count(string(twice), `"lineage"`); n != 1 {
		t.Fatalf("double-stamp produced %d lineage members, want 1", n)
	}
}

// An empty object must not gain a trailing comma (which would be invalid JSON).
func TestLineageIntoEmptyObject(t *testing.T) {
	b, _ := json.MarshalIndent(map[string]any{}, "", "  ")
	got := Lineage{Schema: LineageSchema}.Into(b)
	if !json.Valid(got) {
		t.Fatalf("stamped empty object is invalid JSON:\n%s", got)
	}
}

// Stamp fills every axis, honours the deterministic env overrides, and never
// leaves a field blank (fail-soft to "unknown" or a live value).
func TestStampFieldsAndOverrides(t *testing.T) {
	t.Setenv("FAK_BENCH_UTC", "2099-01-02T03:04:05Z")
	t.Setenv("FAK_BENCH_COMMIT", "cafef00d")
	t.Setenv("FAK_BENCH_NODE", "pinned-node")

	lin := Stamp()
	if lin.Schema != LineageSchema {
		t.Errorf("schema = %q, want %q", lin.Schema, LineageSchema)
	}
	if lin.UTC != "2099-01-02T03:04:05Z" {
		t.Errorf("utc override not honoured: %q", lin.UTC)
	}
	if lin.GitCommit != "cafef00d" {
		t.Errorf("commit override not honoured: %q", lin.GitCommit)
	}
	if lin.Node != "pinned-node" {
		t.Errorf("node override not honoured: %q", lin.Node)
	}
	if lin.GoVersion != runtime.Version() {
		t.Errorf("go_version = %q, want %q", lin.GoVersion, runtime.Version())
	}
	if strings.TrimSpace(lin.AppVersion) == "" {
		t.Errorf("app_version is blank; appversion.Current() must always resolve")
	}
}

// MarshalReport accepts both a value and pre-marshalled bytes, stamping either.
func TestMarshalReportValueAndBytes(t *testing.T) {
	t.Setenv("FAK_BENCH_COMMIT", "abc1234")
	fromValue, err := MarshalReport(map[string]any{"k": "v"})
	if err != nil {
		t.Fatalf("MarshalReport(value): %v", err)
	}
	if !strings.Contains(string(fromValue), `"lineage"`) || !strings.Contains(string(fromValue), `"abc1234"`) {
		t.Fatalf("value form missing lineage:\n%s", fromValue)
	}
	pre, _ := json.MarshalIndent(map[string]any{"k": "v"}, "", "  ")
	fromBytes, err := MarshalReport(pre)
	if err != nil {
		t.Fatalf("MarshalReport(bytes): %v", err)
	}
	if !strings.Contains(string(fromBytes), `"lineage"`) {
		t.Fatalf("bytes form missing lineage:\n%s", fromBytes)
	}
}

func TestMarshalReportStampsBenchmarkArtifact(t *testing.T) {
	t.Setenv("FAK_BENCH_UTC", "2026-06-29T01:02:03Z")
	t.Setenv("FAK_BENCH_COMMIT", "0123456789abcdef0123456789abcdef01234567")
	t.Setenv("FAK_APP_VERSION", "9.8.7")
	t.Setenv("FAK_BENCH_NODE", "node-a")
	t.Setenv("FAK_BENCH_HARNESS_NAME", "modelbench")
	t.Setenv("FAK_BENCH_HARNESS_VERSION", "2.1.0")
	t.Setenv("FAK_BENCH_MODEL_COMMIT", "hf-commit-1")
	t.Setenv("FAK_BENCH_MODEL_HASH", "gguf-sha")
	t.Setenv("FAK_BENCH_BUILD_FLAGS", "fakcuda,fakmetal")
	t.Setenv("FAK_BENCH_WITNESS_TEST", "go test ./internal/model")
	t.Setenv("FAK_BENCH_DOS_VERIFY_RESULT", "OK")
	t.Setenv("FAK_BENCH_DOS_COMMIT_AUDIT", "OK")
	t.Setenv("FAK_BENCH_REPRO_COMMAND", "go run ./cmd/modelbench -gguf model.gguf")
	t.Setenv("FAK_BENCH_ARTIFACT", "experiments/benchmark/runs/modelbench/result.json")
	t.Setenv("FAK_BENCH_BASELINE", "llama.cpp CUDA")

	report := map[string]any{
		"model":     "qwen2.5-3b-instruct-q8_0.gguf",
		"precision": "Q8_0",
		"config": map[string]any{
			"batch_size": 4,
			"workers":    8,
		},
		"kpis": map[string]any{
			"decode_tok_per_sec": 70.5,
			"p50_ms":             12.0,
		},
		"baseline": map[string]any{"engine": "llama"},
	}
	got, err := MarshalReport(report)
	if err != nil {
		t.Fatalf("MarshalReport: %v", err)
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(got, &root); err != nil {
		t.Fatalf("stamped JSON invalid: %v\n%s", err, got)
	}
	if _, ok := root["lineage"]; !ok {
		t.Fatalf("lineage block missing:\n%s", got)
	}
	var art BenchmarkArtifact
	if err := json.Unmarshal(root["benchmark_artifact"], &art); err != nil {
		t.Fatalf("benchmark_artifact missing or invalid: %v\n%s", err, got)
	}
	if art.Schema != BenchmarkArtifactSchema {
		t.Fatalf("schema = %q, want %q", art.Schema, BenchmarkArtifactSchema)
	}
	if art.FAKCommit != "0123456789abcdef0123456789abcdef01234567" {
		t.Fatalf("fak_commit = %q", art.FAKCommit)
	}
	if art.HarnessVersion != "2.1.0" || art.Harness.Name != "modelbench" {
		t.Fatalf("harness = %+v version %q", art.Harness, art.HarnessVersion)
	}
	if art.Model.Name != "qwen2.5-3b-instruct-q8_0.gguf" || art.Model.Precision != "Q8_0" ||
		art.Model.SourceCommit != "hf-commit-1" || art.Model.Hash != "gguf-sha" {
		t.Fatalf("model snapshot = %+v", art.Model)
	}
	if art.Config.Hash == "" || art.Config.Hash == lineageUnknown {
		t.Fatalf("config hash was not derived: %+v", art.Config)
	}
	if art.Invalidated.IsInvalid || art.Invalidated.Reason != nil || art.Invalidated.ReplacementRunID != nil {
		t.Fatalf("default invalidation must be clean/null, got %+v", art.Invalidated)
	}
	if art.Witness.DOSVerifyResult != "OK" || art.Witness.DOSCommitAudit != "OK" ||
		art.Witness.TestPath != "go test ./internal/model" {
		t.Fatalf("witness = %+v", art.Witness)
	}
	if art.Lineage.SourceArtifact != "experiments/benchmark/runs/modelbench/result.json" ||
		art.Lineage.Baseline != "llama.cpp CUDA" {
		t.Fatalf("lineage refs = %+v", art.Lineage)
	}
	if art.Results.Units["kpis.decode_tok_per_sec"] != "tokens/s" || art.Results.Baseline["engine"] != "llama" {
		t.Fatalf("result summary = %+v", art.Results)
	}
}

func TestDetectInvalidationRules(t *testing.T) {
	prev := BenchmarkArtifact{
		RunID:  "run-old",
		Model:  ModelSnapshot{Name: "qwen", Precision: "q8", SourceCommit: "a", Hash: "h1"},
		Config: ConfigSnapshot{Hash: "config-a"},
		Harness: HarnessInfo{
			Name: "modelbench",
		},
	}
	next := prev
	next.RunID = "run-new"

	code := DetectInvalidation(prev, next, []string{`internal\model\quant.go`})
	if !code.IsInvalid || code.Reason == nil || !strings.Contains(*code.Reason, "internal/model/") ||
		code.ReplacementRunID == nil || *code.ReplacementRunID != "run-new" {
		t.Fatalf("code invalidation = %+v", code)
	}

	harness := DetectInvalidation(prev, next, []string{"cmd/modelbench/main.go"})
	if !harness.IsInvalid || harness.Reason == nil || !strings.Contains(*harness.Reason, "harness change") {
		t.Fatalf("harness invalidation = %+v", harness)
	}

	nextModel := prev
	nextModel.RunID = "run-model"
	nextModel.Model.Hash = "h2"
	model := DetectInvalidation(prev, nextModel, nil)
	if !model.IsInvalid || model.Reason == nil || !strings.Contains(*model.Reason, "model weights") {
		t.Fatalf("model invalidation = %+v", model)
	}

	nextConfig := prev
	nextConfig.RunID = "run-config"
	nextConfig.Config.Hash = "config-b"
	config := DetectInvalidation(prev, nextConfig, nil)
	if !config.IsInvalid || config.Reason == nil || !strings.Contains(*config.Reason, "config") {
		t.Fatalf("config invalidation = %+v", config)
	}

	clean := DetectInvalidation(prev, next, []string{"docs/README.md"})
	if clean.IsInvalid || clean.Reason != nil {
		t.Fatalf("clean invalidation = %+v", clean)
	}
}

func TestBuildLineageIndexReadsStampedAndLegacyArtifacts(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FAK_BENCH_UTC", "2026-06-29T01:02:03Z")
	t.Setenv("FAK_BENCH_COMMIT", "feedfacefeedfacefeedfacefeedfacefeedface")
	t.Setenv("FAK_BENCH_NODE", "node-a")
	t.Setenv("FAK_BENCH_RUN_ID", "run-stamped")
	t.Setenv("FAK_BENCH_HARNESS_NAME", "radixbench")
	t.Setenv("FAK_BENCH_MODEL_NAME", "smollm2")
	t.Setenv("FAK_BENCH_MODEL_PRECISION", "q8_0")

	if err := WriteReport(filepath.Join(root, "runs", "stamped.json"), map[string]any{"config": map[string]any{"workers": 2}}); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	legacy := []byte(`{
  "$schema": "benchmark/run-manifest.v1",
  "run_id": "legacy-run",
  "timestamp": "20260624T142454Z",
  "git": {"rev": "abc123"},
  "harness": {"name": "fak-gcp-bench", "version": "2"},
  "model": {"name": "qwen", "precision": "Q8_0"},
  "config": {"workers": 8}
}`)
	if err := os.WriteFile(filepath.Join(root, "manifest.json"), legacy, 0o644); err != nil {
		t.Fatalf("write legacy manifest: %v", err)
	}

	idx, err := BuildLineageIndex(root)
	if err != nil {
		t.Fatalf("BuildLineageIndex: %v", err)
	}
	if len(idx.Artifacts) != 2 {
		t.Fatalf("index artifacts = %d, want 2: %+v", len(idx.Artifacts), idx.Artifacts)
	}
	if idx.Artifacts[0].Path != "manifest.json" || idx.Artifacts[0].RunID != "legacy-run" {
		t.Fatalf("legacy entry = %+v", idx.Artifacts[0])
	}
	if idx.Artifacts[1].Path != "runs/stamped.json" || idx.Artifacts[1].RunID != "run-stamped" ||
		idx.Artifacts[1].Model.Name != "smollm2" {
		t.Fatalf("stamped entry = %+v", idx.Artifacts[1])
	}
}
