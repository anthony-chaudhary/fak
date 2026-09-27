// Package main tests for q8bench's pure, deterministic helpers.
//
// These cover the resource-free numeric helpers in main.go — the fixed-seed
// LCG id generator, argmax, and the min/median duration reducers. They need no
// model file, GPU, or network: every expected value is derived directly from
// the documented recurrence / arithmetic and verified against the real code.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	q8benchCLIEnv     = "Q8BENCH_TEST_CLI"
	q8benchCLIArgsEnv = "Q8BENCH_TEST_ARGS"
)

func TestMain(m *testing.M) {
	if os.Getenv(q8benchCLIEnv) == "1" {
		var args []string
		if err := json.Unmarshal([]byte(os.Getenv(q8benchCLIArgsEnv)), &args); err != nil {
			_, _ = os.Stderr.WriteString("test CLI args: " + err.Error())
			os.Exit(2)
		}
		flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
		os.Args = append([]string{"q8bench"}, args...)
		main()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestQ8Bench_OracleGateFailsClosed(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, dir string)
	}{
		{name: "missing oracle"},
		{name: "unreadable oracle", setup: func(t *testing.T, dir string) {
			t.Helper()
			if err := os.Mkdir(filepath.Join(dir, "oracle.json"), 0o755); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "empty file", setup: writeOracleFixture("")},
		{name: "malformed JSON", setup: writeOracleFixture(`{"prompts":[`)},
		{name: "zero prompts", setup: writeOracleFixture(`{"prompts":[]}`)},
		{name: "zero checked positions", setup: writeOracleFixture(`{"prompts":[{"index":0,"ids":[],"argmax_per_pos":[]}]}`)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.setup != nil {
				tt.setup(t, dir)
			}
			stdout, stderr, err := runQ8BenchCLI(t, "--dir", dir)
			if err == nil {
				t.Fatalf("q8bench succeeded with invalid oracle; stdout=%q stderr=%q", stdout.String(), stderr.String())
			}
			errText := strings.ToLower(stderr.String())
			if !strings.Contains(errText, "oracle:") || strings.Contains(errText, "load:") {
				t.Fatalf("q8bench did not reject the oracle before model load; stderr=%q", stderr.String())
			}
		})
	}
}

func writeOracleFixture(contents string) func(*testing.T, string) {
	return func(t *testing.T, dir string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, "oracle.json"), []byte(contents), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestQ8Bench_DecodeVerdictUsesLikeForLikeStatistic(t *testing.T) {
	if got := medianMS([]time.Duration{4 * time.Millisecond, time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond}); got != 2.5 {
		t.Fatalf("even-sample median = %.2f ms, want averaged middle pair 2.5 ms to match Python statistics.median", got)
	}

	dir, oracleArgmax := writeTinyQ8BenchModel(t)
	expDir := t.TempDir()
	writeHFRegimeFixtures(t, expDir, true)
	reportPath := filepath.Join(t.TempDir(), "report.json")
	args := []string{"--dir", dir, "--exp", expDir, "--out", reportPath, "--prefill-reps", "1", "--decode-reps", "4", "--decode-steps", "256", "--decode-prompt", "1"}
	_, stderr, err := runQ8BenchCLI(t, args...)
	if err != nil {
		t.Fatalf("q8bench valid fixture failed: %v\nstderr=%s", err, stderr.String())
	}
	report := readQ8BenchReport(t, reportPath)
	assertSuccessfulQ8BenchReport(t, report, stderr.String())

	writeHFRegimeFixtures(t, expDir, false)
	_, stderr, err = runQ8BenchCLI(t, args...)
	if err != nil {
		t.Fatalf("q8bench unavailable-HF fixture failed: %v\nstderr=%s", err, stderr.String())
	}
	verdict := reportMap(t, readQ8BenchReport(t, reportPath), "verdict")
	if reportBool(t, verdict, "hf_int8_available") || reportBool(t, verdict, "hf_f32_available") ||
		reportBool(t, verdict, "beats_hf_int8") || reportBool(t, verdict, "beats_hf_f32") {
		t.Fatalf("unknown HF regime became available or a win: %#v", verdict)
	}
	if !strings.Contains(stderr.String(), "HF int8 median=unavailable (beat=false)") ||
		!strings.Contains(stderr.String(), "HF f32 median=unavailable (beat=false)") {
		t.Fatalf("human verdict did not identify unavailable HF medians: %q", stderr.String())
	}

	writeJSONFile(t, filepath.Join(dir, "oracle.json"), map[string]any{
		"prompts": []any{map[string]any{"index": 0, "ids": []int{1}, "argmax_per_pos": []int{(oracleArgmax + 1) % 16}}},
	})
	_, stderr, err = runQ8BenchCLI(t, args...)
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
		t.Fatalf("argmax drift exit = %v, want code 2; stderr=%s", err, stderr.String())
	}
	correctness := reportMap(t, readQ8BenchReport(t, reportPath), "correctness")
	if reportBool(t, correctness, "gate_argmax_exact_vs_hf_oracle") || !strings.Contains(stderr.String(), "argmax-exact=false") {
		t.Fatalf("argmax drift was not reflected in JSON and human verdict: report=%#v stderr=%q", correctness, stderr.String())
	}
}

func runQ8BenchCLI(t *testing.T, args ...string) (bytes.Buffer, bytes.Buffer, error) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve q8bench test executable: %v", err)
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal q8bench args: %v", err)
	}
	var stdout, stderr bytes.Buffer
	run := exec.Command(executable)
	run.Env = append(os.Environ(), q8benchCLIEnv+"=1", q8benchCLIArgsEnv+"="+string(encoded))
	run.Stdout = &stdout
	run.Stderr = &stderr
	err = run.Run()
	return stdout, stderr, err
}

func writeTinyQ8BenchModel(t *testing.T) (string, int) {
	t.Helper()
	dir := t.TempDir()
	config := map[string]any{
		"model_type": "llama", "hidden_size": 32, "intermediate_size": 32,
		"num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
		"head_dim": 32, "vocab_size": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"tie_word_embeddings": true, "eos_token_id": -1,
	}
	type tensorSpec struct {
		name  string
		shape []int
		norm  bool
	}
	tensors := []tensorSpec{
		{"model.embed_tokens.weight", []int{16, 32}, false},
		{"model.layers.0.input_layernorm.weight", []int{32}, true},
		{"model.layers.0.self_attn.q_proj.weight", []int{32, 32}, false},
		{"model.layers.0.self_attn.k_proj.weight", []int{32, 32}, false},
		{"model.layers.0.self_attn.v_proj.weight", []int{32, 32}, false},
		{"model.layers.0.self_attn.o_proj.weight", []int{32, 32}, false},
		{"model.layers.0.post_attention_layernorm.weight", []int{32}, true},
		{"model.layers.0.mlp.gate_proj.weight", []int{32, 32}, false},
		{"model.layers.0.mlp.up_proj.weight", []int{32, 32}, false},
		{"model.layers.0.mlp.down_proj.weight", []int{32, 32}, false},
		{"model.norm.weight", []int{32}, true},
	}
	manifest := map[string]any{}
	var raw []byte
	for tensorIndex, tensor := range tensors {
		elements := 1
		for _, dim := range tensor.shape {
			elements *= dim
		}
		offset := len(raw)
		for i := 0; i < elements; i++ {
			value := float32(1)
			if !tensor.norm {
				value = float32(((tensorIndex+1)*(i+3))%11-5) / 50
			}
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
			raw = append(raw, bits[:]...)
		}
		manifest[tensor.name] = map[string]any{"dtype": "f32", "shape": tensor.shape, "offset": offset, "nbytes": elements * 4}
	}
	writeJSONFile(t, filepath.Join(dir, "config.json"), config)
	writeJSONFile(t, filepath.Join(dir, "manifest.json"), manifest)
	if err := os.WriteFile(filepath.Join(dir, "weights.f32"), raw, 0o644); err != nil {
		t.Fatalf("write tiny weights: %v", err)
	}
	m, err := model.Load(dir)
	if err != nil {
		t.Fatalf("load tiny model: %v", err)
	}
	m.Quantize()
	s := m.NewSession()
	s.Quant = true
	argmax := mathx.ArgmaxF32(s.Prefill([]int{1}))
	writeJSONFile(t, filepath.Join(dir, "oracle.json"), map[string]any{
		"prompts": []any{map[string]any{"index": 0, "ids": []int{1}, "argmax_per_pos": []int{argmax}}},
	})
	return dir, argmax
}

func writeHFRegimeFixtures(t *testing.T, dir string, matching bool) {
	t.Helper()
	configs := []any{map[string]any{"decode": map[string]any{
		"prompt_tokens": 8, "decode_steps": 256, "reps": 4, "per_token_median_ms": 0.000001,
	}}}
	if matching {
		configs = append(configs, map[string]any{"decode": map[string]any{
			"prompt_tokens": 1, "decode_steps": 256, "reps": 4, "per_token_median_ms": 1000000,
		}})
	} else {
		configs = append(configs, map[string]any{"decode": map[string]any{"per_token_median_ms": 1000000}})
	}
	for _, name := range []string{"hf.json", "hf-int8.json"} {
		writeJSONFile(t, filepath.Join(dir, name), map[string]any{"configs": configs})
	}
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readQ8BenchReport(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var report map[string]any
	if err := json.Unmarshal(b, &report); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	return report
}

func assertSuccessfulQ8BenchReport(t *testing.T, report map[string]any, human string) {
	t.Helper()
	correctness := reportMap(t, report, "correctness")
	if !reportBool(t, correctness, "gate_argmax_exact_vs_hf_oracle") || reportNumber(t, correctness, "total_positions") != 1 {
		t.Fatalf("correctness report did not record an exact observation: %#v", correctness)
	}
	decode := reportMap(t, report, "decode")
	if reportNumber(t, decode, "per_token_median_ms") <= 0 {
		t.Fatalf("decode median missing: %#v", decode)
	}
	verdict := reportMap(t, report, "verdict")
	for _, key := range []string{"fak_int8_decode_median_ms", "hf_int8_best_decode_median_ms", "hf_f32_best_decode_median_ms"} {
		if reportNumber(t, verdict, key) <= 0 {
			t.Errorf("verdict[%q] missing or nonpositive: %#v", key, verdict)
		}
	}
	if _, old := verdict["fak_int8_decode_ms_min"]; old {
		t.Errorf("verdict still exposes mislabeled min field: %#v", verdict)
	}
	if !reportBool(t, verdict, "beats_hf_int8") || !reportBool(t, verdict, "beats_hf_f32") {
		t.Errorf("faster unlike HF config contaminated matched verdict: %#v", verdict)
	}
	if !strings.Contains(human, "decode(median)=") || !strings.Contains(human, "HF int8 median=") ||
		!strings.Contains(human, "HF f32 median=") || !strings.Contains(human, "argmax-exact=true") || strings.Contains(human, "decode(min)") {
		t.Errorf("human verdict does not name the like-for-like statistic: %q", human)
	}
}

func reportMap(t *testing.T, parent map[string]any, key string) map[string]any {
	t.Helper()
	value, ok := parent[key].(map[string]any)
	if !ok {
		t.Fatalf("report[%q] = %#v, want object", key, parent[key])
	}
	return value
}

func reportNumber(t *testing.T, parent map[string]any, key string) float64 {
	t.Helper()
	value, ok := parent[key].(float64)
	if !ok {
		t.Fatalf("report[%q] = %#v, want number", key, parent[key])
	}
	return value
}

func reportBool(t *testing.T, parent map[string]any, key string) bool {
	t.Helper()
	value, ok := parent[key].(bool)
	if !ok {
		t.Fatalf("report[%q] = %#v, want bool", key, parent[key])
	}
	return value
}

func TestLcgIDs(t *testing.T) {
	// The recurrence is fixed-seed (state0 = 2463534242) and deterministic, so
	// its output is reproducible bit-for-bit. The first five raw masked states
	// are 1266642227, 1626945776, 857116265, 1848955118, 171551119; taken mod
	// 100 they yield the sequence below.
	tests := []struct {
		name  string
		n     int
		vocab int
		want  []int
	}{
		{"n0 empty", 0, 100, []int{}},
		{"first five mod100", 5, 100, []int{27, 76, 65, 18, 19}},
		{"vocab1 all zero", 3, 1, []int{0, 0, 0}},
		{"vocab10", 8, 10, []int{7, 6, 5, 8, 9, 0, 1, 6}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lcgIDs(tt.n, tt.vocab)
			if len(got) != tt.n {
				t.Fatalf("len = %d, want %d", len(got), tt.n)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("len mismatch: got %v want %v", got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("ids[%d] = %d, want %d (full got %v)", i, got[i], tt.want[i], got)
				}
			}
		})
	}

	// Determinism: two independent calls with the same args agree exactly.
	a := lcgIDs(16, 257)
	b := lcgIDs(16, 257)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d: %d != %d", i, a[i], b[i])
		}
	}

	// Every id must be a valid index into [0,vocab).
	const vocab = 50
	for i, id := range lcgIDs(64, vocab) {
		if id < 0 || id >= vocab {
			t.Errorf("ids[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want int
	}{
		{"single", []float32{7}, 0},
		{"max at start", []float32{9, 1, 2, 3}, 0},
		{"max at end", []float32{1, 2, 3, 9}, 3},
		{"max in middle", []float32{1, 8, 2}, 1},
		{"tie returns first", []float32{1, 3, 3, 2}, 1},
		{"all negative", []float32{-5, -2, -9}, 1},
		{"all equal returns first", []float32{4, 4, 4}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mathx.ArgmaxF32(tt.v); got != tt.want {
				t.Errorf("ArgmaxF32(%v) = %d, want %d", tt.v, got, tt.want)
			}
		})
	}
}

func TestMinMS(t *testing.T) {
	tests := []struct {
		name string
		ds   []time.Duration
		want float64
	}{
		{"single", []time.Duration{5 * time.Millisecond}, 5},
		{"min first", []time.Duration{1 * time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond}, 1},
		{"min last", []time.Duration{3 * time.Millisecond, 2 * time.Millisecond, 1 * time.Millisecond}, 1},
		{"sub-millisecond", []time.Duration{1500 * time.Microsecond, 2 * time.Millisecond}, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := minMS(tt.ds); got != tt.want {
				t.Errorf("minMS(%v) = %v, want %v", tt.ds, got, tt.want)
			}
		})
	}
}

func TestMedianMS(t *testing.T) {
	// medianMS follows Python statistics.median (including averaging the middle
	// pair for even inputs) and must not mutate its input.
	tests := []struct {
		name string
		ds   []time.Duration
		want float64
	}{
		{"single", []time.Duration{5 * time.Millisecond}, 5},
		{"odd unsorted", []time.Duration{3 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond}, 2},
		{"even averages middle pair", []time.Duration{4 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond}, 2.5},
		{"fractional", []time.Duration{1500 * time.Microsecond}, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := medianMS(tt.ds); got != tt.want {
				t.Errorf("medianMS(%v) = %v, want %v", tt.ds, got, tt.want)
			}
		})
	}

	// medianMS must leave its argument unsorted (it sorts a copy).
	in := []time.Duration{3 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond}
	_ = medianMS(in)
	if in[0] != 3*time.Millisecond || in[1] != 1*time.Millisecond || in[2] != 2*time.Millisecond {
		t.Errorf("medianMS mutated its input: %v", in)
	}
}
