package overnightwitness_test

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

const artifactSHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

var pinnedFiles = map[string]string{
	"fold-report.json":                  "60b887fb11b3d9383a31ae1b8dfc409565e200db2b0e4ab0798fd350ccb411ee",
	"fak-native-q4km-a100-rows.json":    "d1f82b27f28ce8476420805db1b455221b42c69f3409fa5caefdf13bdd4876e3",
	"fak-native-q4km-a100-identity.txt": "b49bcb79e056dc3d844fd5940718af2e054c2b0950356a6a1b0b3b03675cfde4",
	"llamacpp-q4km-a100-bench.json":     "a5f7eee7efb3b17146c535cece66a20917e5091ceb0e578d33b583c55f6cff31",
}

var sourceFiles = map[string]string{
	"gcp-a100-bf16-reference.jsonl": "15c4d8e678510b36e5d3377a92ba17e495658a58f037d224891bfa0f1c5a71b8",
	"gcp-a100-fp8-reference.jsonl":  "fcb4a4bbc37549443fe9d548f5d1b6eda4effd0fe1f7e57608e794f1602c6328",
}

type experiment struct {
	ExperimentID   string  `json:"experiment_id"`
	NodeClass      string  `json:"node_class"`
	Model          string  `json:"model"`
	Engine         string  `json:"engine"`
	Runtime        string  `json:"runtime"`
	EnableThinking bool    `json:"enable_thinking"`
	QualityPass    bool    `json:"quality_pass"`
	LatencyMS      float64 `json:"latency_ms"`
	Error          *string `json:"error"`
}

type foldReport struct {
	ExperimentRows      int               `json:"experiment_rows"`
	UniqueExperimentIDs int               `json:"unique_experiment_ids"`
	AllQualityPass      bool              `json:"all_quality_pass"`
	SourceSHA256        map[string]string `json:"source_sha256"`
	Arms                []struct {
		Model           string  `json:"model"`
		Engine          string  `json:"engine"`
		EnableThinking  bool    `json:"enable_thinking"`
		Experiments     int     `json:"experiments"`
		QualityPasses   int     `json:"quality_passes"`
		Errors          int     `json:"errors"`
		MedianLatencyMS float64 `json:"median_latency_ms"`
	} `json:"arms"`
	HillClimb []struct {
		QualityEqual      bool    `json:"quality_equal"`
		MedianLatencyGain float64 `json:"median_latency_gain"`
		Verdict           string  `json:"verdict"`
	} `json:"hill_climb"`
	Native struct {
		Engine          string  `json:"engine"`
		Backend         string  `json:"backend"`
		ForwardPath     string  `json:"forward_path"`
		ArtifactSHA256  string  `json:"artifact_sha256"`
		QualityPasses   string  `json:"quality_passes"`
		MedianDecodeTPS float64 `json:"median_decode_tps"`
	} `json:"native_fak_current"`
	Llama struct {
		Commit          string  `json:"commit"`
		MedianDecodeTPS float64 `json:"median_decode_tps"`
	} `json:"llamacpp_reference"`
	Parity struct {
		Verdict            string  `json:"verdict"`
		MatchedDecodeRatio float64 `json:"matched_decode_ratio"`
		Threshold          float64 `json:"threshold"`
	} `json:"parity"`
	NextExperiment string `json:"next_experiment"`
}

func TestOvernightFoldWitness(t *testing.T) {
	var rows []experiment
	for name, wantHash := range sourceFiles {
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(body)
		if got := hex.EncodeToString(sum[:]); got != wantHash {
			t.Fatalf("%s hash=%s want=%s", name, got, wantHash)
		}
		s := bufio.NewScanner(strings.NewReader(string(body)))
		for s.Scan() {
			var row experiment
			if err := json.Unmarshal(s.Bytes(), &row); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			rows = append(rows, row)
		}
		if err := s.Err(); err != nil {
			t.Fatal(err)
		}
	}
	if len(rows) != 120 {
		t.Fatalf("rows=%d want=120", len(rows))
	}
	ids := make(map[string]bool, len(rows))
	type armKey struct {
		model    string
		thinking bool
	}
	arms := make(map[armKey][]float64)
	for _, row := range rows {
		if row.ExperimentID == "" || ids[row.ExperimentID] {
			t.Fatalf("empty or duplicate experiment_id %q", row.ExperimentID)
		}
		ids[row.ExperimentID] = true
		if row.Engine != "vllm-reference" || row.Runtime != "vllm" {
			t.Fatalf("%s engine/runtime=%s/%s", row.ExperimentID, row.Engine, row.Runtime)
		}
		if !row.QualityPass || row.Error != nil {
			t.Fatalf("%s quality=%v error=%v", row.ExperimentID, row.QualityPass, row.Error)
		}
		arms[armKey{row.Model, row.EnableThinking}] = append(arms[armKey{row.Model, row.EnableThinking}], row.LatencyMS)
	}
	if len(ids) != 120 || len(arms) != 4 {
		t.Fatalf("unique=%d arms=%d", len(ids), len(arms))
	}
	for key, values := range arms {
		if len(values) != 30 {
			t.Fatalf("arm %+v rows=%d", key, len(values))
		}
	}

	var report foldReport
	readJSON(t, "fold-report.json", &report)
	if report.ExperimentRows != 120 || report.UniqueExperimentIDs != 120 || !report.AllQualityPass {
		t.Fatalf("bad fold totals: %+v", report)
	}
	if report.SourceSHA256["issue8848-bf16.jsonl"] != sourceFiles["gcp-a100-bf16-reference.jsonl"] || report.SourceSHA256["issue8848-fp8.jsonl"] != sourceFiles["gcp-a100-fp8-reference.jsonl"] {
		t.Fatal("fold source hashes do not pin the ledgers")
	}
	if len(report.Arms) != 4 || len(report.HillClimb) != 2 {
		t.Fatalf("report arms=%d folds=%d", len(report.Arms), len(report.HillClimb))
	}
	for _, arm := range report.Arms {
		values := arms[armKey{arm.Model, arm.EnableThinking}]
		if arm.Engine != "vllm-reference" || arm.Experiments != 30 || arm.QualityPasses != 30 || arm.Errors != 0 || !near(arm.MedianLatencyMS, median(values), 1e-9) {
			t.Fatalf("invalid arm %+v computed_median=%v", arm, median(values))
		}
	}
	for _, fold := range report.HillClimb {
		if fold.Verdict != "KEEP" || !fold.QualityEqual || fold.MedianLatencyGain <= 0 {
			t.Fatalf("invalid hill climb %+v", fold)
		}
	}
	if report.Native.Engine != "fak-native" || report.Native.Backend != "cuda" || report.Native.ForwardPath != "cuda/qwen35-gdn-ssm-decode-v1" || report.Native.ArtifactSHA256 != artifactSHA256 || report.Native.QualityPasses != "5/5 exact Q38" || report.Native.MedianDecodeTPS != 0.3 {
		t.Fatalf("invalid native fold %+v", report.Native)
	}
	if report.Llama.Commit != "171974745" || !near(report.Llama.MedianDecodeTPS, 36.5491, 1e-9) {
		t.Fatalf("invalid llama reference %+v", report.Llama)
	}
	if report.Parity.Verdict != "HOLD_BELOW_PARITY" || report.Parity.MatchedDecodeRatio >= report.Parity.Threshold {
		t.Fatalf("invalid parity %+v", report.Parity)
	}
	if !strings.Contains(report.NextExperiment, "cached fak-native CUDA decode collapse") || !strings.Contains(report.NextExperiment, "session-state/cache restore") {
		t.Fatalf("next experiment is not the selected cache-state lever: %q", report.NextExperiment)
	}

	identity, err := os.ReadFile("fak-native-q4km-a100-identity.txt")
	if err != nil {
		t.Fatal(err)
	}
	text := string(identity)
	for _, marker := range []string{"backend=cuda", "forward_path=cuda/qwen35-gdn-ssm-decode-v1", "q4k=true", "reused=22tok", "decode=3tok"} {
		if !strings.Contains(text, marker) {
			t.Fatalf("native identity missing %q", marker)
		}
	}
	if strings.Contains(strings.ToLower(text), "llama.cpp") {
		t.Fatal("native identity unexpectedly contains llama.cpp delegation")
	}
}

func assertFileHash(t *testing.T, name, want string) {
	t.Helper()
	body, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Fatalf("%s hash=%s want=%s", name, got, want)
	}
}

func readJSON(t *testing.T, name string, dst any) {
	t.Helper()
	body, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, dst); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func median(values []float64) float64 {
	copyValues := append([]float64(nil), values...)
	sort.Float64s(copyValues)
	n := len(copyValues)
	if n%2 == 1 {
		return copyValues[n/2]
	}
	return (copyValues[n/2-1] + copyValues[n/2]) / 2
}

func near(a, b, tolerance float64) bool { return math.Abs(a-b) <= tolerance }
