package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func kernelOracleFixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "kernel_oracle", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func kernelOracleShapesFile(t *testing.T) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), "shapes.json")
	var stdout, stderr bytes.Buffer
	meta := filepath.Join("testdata", "kernel_oracle", "qwen-metadata.json")
	if code := run(&stdout, &stderr, []string{"kernel-oracle", "shapes", "--metadata-json", meta, "--n", "1,512", "--out", out}); code != 0 {
		t.Fatalf("shapes exit %d: %s", code, stderr.String())
	}
	return out
}

func kernelOracleRankFixture(t *testing.T) []byte {
	t.Helper()
	dir := filepath.Join("testdata", "kernel_oracle")
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"kernel-oracle", "rank", "--shapes", kernelOracleShapesFile(t),
		"--ggml", filepath.Join(dir, "ggml-perf.txt"), "--fak-bench", filepath.Join(dir, "fak-bench.txt"),
		"--fak-profile", filepath.Join(dir, "fak-profile.txt"), "--evidence-class", "FIXTURE"})
	if code != 0 {
		t.Fatalf("rank exit %d: %s", code, stderr.String())
	}
	return stdout.Bytes()
}

func findOracleRow(t *testing.T, rows []oracleRow, op, dtype, shape string, n int64) oracleRow {
	t.Helper()
	for _, r := range rows {
		if r.Op == op && r.Dtype == dtype && r.Shape == shape && r.N == n {
			return r
		}
	}
	t.Fatalf("row %s %s %s n=%d not found", op, dtype, shape, n)
	return oracleRow{}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleParsesRecordedTestBackendOpsPerf(t *testing.T) {
	got := parseGGMLPerf(kernelOracleFixture(t, "ggml-perf.txt"), "Vulkan0")
	if got.Backend != "Vulkan0" || got.Device != "AMD Radeon 8060S Graphics (RADV STRIX_HALO)" {
		t.Fatalf("backend=%q device=%q", got.Backend, got.Device)
	}
	if got.Unsupported != 1 || len(got.Rows) != 5 {
		t.Fatalf("unsupported=%d rows=%d, want 1 and 5", got.Unsupported, len(got.Rows))
	}
	byKey := map[string]oracleMeasurement{}
	for _, r := range got.Rows {
		byKey[r.key()] = r
	}
	qkv := byKey[oracleMatKey("iq2_xxs", 10240, 5120, 1)]
	if qkv.US != 120 || qkv.Samples != 2 {
		t.Fatalf("generic iq2_xxs n=1 = %+v, want min-of-2 120us from the Vulkan0 section only", qkv)
	}
	if pp := byKey[oracleMatKey("iq2_xxs", 10240, 5120, 512)]; pp.US != 2500 {
		t.Fatalf("generic n=512 us = %v", pp.US)
	}
	builtin := byKey[oracleMatKey("q6_k", 5120, 17408, 1)]
	if builtin.US != 300 || math.Abs(builtin.TFLOPS-0.5942) > 1e-9 {
		t.Fatalf("builtin q6_k = %+v", builtin)
	}
	var gdn oracleMeasurement
	for _, r := range got.Rows {
		if r.Op == "GATED_DELTA_NET" {
			gdn = r
		}
	}
	if gdn.US != 11.5 || gdn.GBps != 8.15 || gdn.Dtype != "f32" || gdn.Params == "" {
		t.Fatalf("generic non-matmul row = %+v", gdn)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleShapesFromModelTensorDirectory(t *testing.T) {
	var list oracleShapeList
	b, err := os.ReadFile(kernelOracleShapesFile(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &list); err != nil {
		t.Fatal(err)
	}
	if list.Schema != kernelOracleShapesSchema || list.Arch != "qwen35" {
		t.Fatalf("schema=%q arch=%q", list.Schema, list.Arch)
	}
	// 8 matmul weights (token_embd, norm and conv excluded), 7 distinct (dtype,m,k), x2 n values.
	if len(list.Shapes) != 14 {
		t.Fatalf("shapes=%d, want 14", len(list.Shapes))
	}
	for _, s := range list.Shapes {
		if s.Dtype == "q2_k" || s.Dtype == "f32" {
			t.Fatalf("excluded tensor leaked into shapes: %+v", s)
		}
		if s.Dtype == "iq2_xxs" && s.M == 10240 && s.Count != 2 {
			t.Fatalf("qkv count = %d, want 2 layers", s.Count)
		}
	}
	if !containsString(list.Ops, "GATED_DELTA_NET") || !containsString(list.Ops, "FLASH_ATTN_EXT") {
		t.Fatalf("ops = %v", list.Ops)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleShapesExcludeEngramTables(t *testing.T) {
	// Directory metadata only: the published Q2_K Engram table has this shape,
	// but its rows are gathered on demand, never multiplied as a dense matrix.
	const metadata = `{"metadata":[{"key":"general.architecture","value":"deepseek41"}],"tensors":[
		{"name":"blk.1.engram_embd.weight","type":"Q2_K","dims":[256,384006168]},
		{"name":"blk.2.engram_table.weight","type":"Q2_K","dims":[256,384006168]},
		{"name":"blk.1.engram_wkv.weight","type":"Q2_K","dims":[256,512]},
		{"name":"blk.1.engram_q.weight","type":"Q2_K","dims":[256,512]},
		{"name":"blk.1.engram_k.weight","type":"Q2_K","dims":[256,512]},
		{"name":"blk.1.engram_kv.weight","type":"Q2_K","dims":[256,512]}
	]}`
	path := filepath.Join(t.TempDir(), "engram-metadata.json")
	if err := os.WriteFile(path, []byte(metadata), 0600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runKernelOracle(&stdout, &stderr, []string{"shapes", "--metadata-json", path, "--n", "1,512"}); code != 0 {
		t.Fatalf("shapes exit %d: %s", code, stderr.String())
	}
	var list oracleShapeList
	if err := json.Unmarshal(stdout.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Shapes) != 2 {
		t.Fatalf("shapes = %+v, want only projection shapes for n=1,512", list.Shapes)
	}
	for i, n := range []int64{1, 512} {
		s := list.Shapes[i]
		if s.Op != "MUL_MAT" || s.Dtype != "q2_k" || s.M != 512 || s.K != 256 || s.N != n || s.Count != 4 {
			t.Fatalf("projection shape for n=%d = %+v", n, s)
		}
		if len(s.Tensors) != 4 {
			t.Fatalf("projection roles = %v, want four", s.Tensors)
		}
		for _, role := range []string{"engram_wkv", "engram_q", "engram_k", "engram_kv"} {
			if !containsString(s.Tensors, role) {
				t.Errorf("projection %q missing from %v", role, s.Tensors)
			}
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestKernelOracleRankJoinsRanksAndNeverZeroFills(t *testing.T) {
	var rec oracleReceipt
	if err := json.Unmarshal(kernelOracleRankFixture(t), &rec); err != nil {
		t.Fatal(err)
	}
	if rec.Schema != kernelOracleReceiptSchema || rec.EvidenceClass != "FIXTURE" || rec.GGMLBackend != "Vulkan0" {
		t.Fatalf("receipt header = %q %q %q", rec.Schema, rec.EvidenceClass, rec.GGMLBackend)
	}
	if len(rec.Rows) < 2 || rec.Rows[0].Shape != "10240x5120" || rec.Rows[1].Shape != "6144x5120" {
		t.Fatalf("top rows not ranked by score: %+v", rec.Rows[:2])
	}
	qkv := findOracleRow(t, rec.Rows, "MUL_MAT", "iq2_xxs", "10240x5120", 1)
	if qkv.Status != oracleStatusBoth || *qkv.FakUS != 140 || qkv.FakSource != "microbench" || *qkv.FakInsituUS != 150 ||
		*qkv.GGMLUS != 120 || *qkv.Ratio != 1.167 || qkv.PortOrAdapt || *qkv.SharePct != 72.917 {
		t.Fatalf("qkv row = %+v", qkv)
	}
	z := findOracleRow(t, rec.Rows, "MUL_MAT", "iq2_xxs", "6144x5120", 1)
	if z.Status != oracleStatusBoth || z.FakSource != "insitu_profile" || *z.FakUS != 104 || *z.Ratio != 1.3 || !z.PortOrAdapt {
		t.Fatalf("in-situ joined row = %+v", z)
	}
	if z.FakGBps == nil || z.GGMLGBps == nil || z.FakTFLOPS == nil || z.GGMLTFLOPS == nil {
		t.Fatalf("rates missing on a comparable row: %+v", z)
	}
	q6 := findOracleRow(t, rec.Rows, "MUL_MAT", "q6_k", "5120x17408", 1)
	if q6.Status != oracleStatusFakMissing || q6.FakUS != nil || q6.Ratio != nil || q6.GGMLUS == nil {
		t.Fatalf("fak-missing row = %+v", q6)
	}
	iq4 := findOracleRow(t, rec.Rows, "MUL_MAT", "iq4_xs", "17408x5120", 1)
	if iq4.Status != oracleStatusGGMLMissing || iq4.GGMLUS != nil || iq4.Ratio != nil || iq4.FakUS == nil {
		t.Fatalf("ggml-missing row = %+v", iq4)
	}
	lm := findOracleRow(t, rec.Rows, "MUL_MAT", "q4_k", "248320x5120", 512)
	if lm.Status != oracleStatusBothMissing || lm.FakUS != nil || lm.GGMLUS != nil {
		t.Fatalf("unmeasured row = %+v", lm)
	}
	stages := map[string]oracleStage{}
	for _, st := range rec.FakStages {
		stages[st.Kernel+" "+st.Groups] = st
	}
	if st := stages["iq2xxs_matvec 384x1"]; st.USPerCall != 104 || st.JoinedShape == "" {
		t.Fatalf("last profile report must win and join: %+v", st)
	}
	if st := stages["iq2xxs_matvec 320x1"]; st.Unjoined != "ambiguous_2_shapes" {
		t.Fatalf("ambiguous stage = %+v", st)
	}
	if st := stages["qwen35_gdn_q8_in_proj 640x1"]; st.Unjoined != "kernel_geometry_unknown" {
		t.Fatalf("unknown-geometry stage = %+v", st)
	}
	// 9 both_missing rows count toward both missing tallies.
	want := oracleSummary{Rows: 15, Both: 2, FakMissing: 12, GGMLMissing: 10, PortOrAdapt: 1, GGMLUnsupported: 1}
	if rec.Summary != want {
		t.Fatalf("summary = %+v, want %+v", rec.Summary, want)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestKernelOracleReceiptRoundTripsAndIsDeterministic(t *testing.T) {
	first, second := kernelOracleRankFixture(t), kernelOracleRankFixture(t)
	var a, b oracleReceipt
	if err := json.Unmarshal(first, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(second, &b); err != nil {
		t.Fatal(err)
	}
	delete(a.Inputs, "shapes")
	delete(b.Inputs, "shapes")
	ra, _ := json.Marshal(a)
	rb, _ := json.Marshal(b)
	if !bytes.Equal(ra, rb) {
		t.Fatal("two ranks of the same inputs differ")
	}
	var again oracleReceipt
	if err := json.Unmarshal(ra, &again); err != nil {
		t.Fatal(err)
	}
	if rr, _ := json.Marshal(again); !bytes.Equal(ra, rr) {
		t.Fatal("receipt does not round-trip through its schema")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestKernelOracleUsageExitCodes(t *testing.T) {
	for _, argv := range [][]string{{"kernel-oracle"}, {"kernel-oracle", "bogus"}, {"kernel-oracle", "shapes"}, {"kernel-oracle", "rank", "--shapes", "x.json"}} {
		var out, errb bytes.Buffer
		if code := run(&out, &errb, argv); code != 2 {
			t.Fatalf("%v exit %d, want 2", argv, code)
		}
	}
}

func kernelOracleTempHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	return home
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleShapesExpandsGGUFTilde(t *testing.T) {
	home := kernelOracleTempHome(t)
	var model bytes.Buffer
	write := func(v any) {
		t.Helper()
		if err := binary.Write(&model, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	// A complete GGUF v3 file with one 32x1 F16 tensor, read through ggufload.Open.
	model.WriteString("GGUF")
	write(uint32(3))
	write(uint64(1)) // tensor count
	write(uint64(0)) // metadata count
	const name = "output.weight"
	write(uint64(len(name)))
	model.WriteString(name)
	write(uint32(2)) // dimensions
	write(uint64(32))
	write(uint64(1))
	write(uint32(1)) // F16
	write(uint64(0)) // tensor data offset
	for model.Len()%32 != 0 {
		model.WriteByte(0)
	}
	model.Write(make([]byte, 64))
	path := filepath.Join(home, "model.gguf")
	if err := os.WriteFile(path, model.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := run(&stdout, &stderr, []string{"kernel-oracle", "shapes", "--gguf", "~/model.gguf", "--n", "1"}); code != 0 {
		t.Fatalf("shapes exit %d: %s", code, stderr.String())
	}
	var list oracleShapeList
	if err := json.Unmarshal(stdout.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if list.Model != path || len(list.Shapes) != 1 {
		t.Fatalf("model=%q shapes=%+v", list.Model, list.Shapes)
	}
	if s := list.Shapes[0]; s.Dtype != "f16" || s.M != 1 || s.K != 32 || s.N != 1 || s.Count != 1 {
		t.Fatalf("shape=%+v", s)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleShapesExpandsMetadataAndOutputTilde(t *testing.T) {
	home := kernelOracleTempHome(t)
	path := filepath.Join(home, "metadata.json")
	if err := os.WriteFile(path, []byte(kernelOracleFixture(t, "qwen-metadata.json")), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := run(&stdout, &stderr, []string{"kernel-oracle", "shapes", "--metadata-json", "~/metadata.json", "--out", "~/shapes.json"}); code != 0 {
		t.Fatalf("shapes exit %d: %s", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("--out wrote stdout: %s", stdout.String())
	}
	b, err := os.ReadFile(filepath.Join(home, "shapes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var list oracleShapeList
	if err := json.Unmarshal(b, &list); err != nil {
		t.Fatal(err)
	}
	if list.Model != path || len(list.Shapes) == 0 {
		t.Fatalf("model=%q shapes=%d", list.Model, len(list.Shapes))
	}
	if code := run(&stdout, &stderr, []string{"kernel-oracle", "shapes", "--metadata-json", path}); code != 0 {
		t.Fatalf("absolute shapes exit %d: %s", code, stderr.String())
	}
	if !bytes.Equal(b, stdout.Bytes()) {
		t.Fatal("tilde file output differs from absolute-path stdout")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestKernelOracleRankExpandsInputAndOutputTilde(t *testing.T) {
	home := kernelOracleTempHome(t)
	shapes, err := os.ReadFile(kernelOracleShapesFile(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "shapes.json"), shapes, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"ggml-perf.txt", "fak-bench.txt", "fak-profile.txt"} {
		if err := os.WriteFile(filepath.Join(home, name), []byte(kernelOracleFixture(t, name)), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, flags := range [][]string{{"ggml", "fak-bench", "fak-profile"}, {"ggml"}, {"fak-bench"}, {"fak-profile"}} {
		t.Run(strings.Join(flags, "_"), func(t *testing.T) {
			tildeArgs := []string{"kernel-oracle", "rank", "--shapes", "~/shapes.json", "--out", "~/rank.json"}
			absArgs := []string{"kernel-oracle", "rank", "--shapes", filepath.Join(home, "shapes.json")}
			for _, flag := range flags {
				name := flag + ".txt"
				if flag == "ggml" {
					name = "ggml-perf.txt"
				}
				tildeArgs = append(tildeArgs, "--"+flag, "~/"+name)
				absArgs = append(absArgs, "--"+flag, filepath.Join(home, name))
			}
			var stdout, stderr bytes.Buffer
			if code := run(&stdout, &stderr, tildeArgs); code != 0 {
				t.Fatalf("rank exit %d: %s", code, stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("--out wrote stdout: %s", stdout.String())
			}
			b, err := os.ReadFile(filepath.Join(home, "rank.json"))
			if err != nil {
				t.Fatal(err)
			}
			if code := run(&stdout, &stderr, absArgs); code != 0 {
				t.Fatalf("absolute rank exit %d: %s", code, stderr.String())
			}
			if !bytes.Equal(b, stdout.Bytes()) {
				t.Fatal("tilde file output differs from absolute-path stdout")
			}
		})
	}
}

func oracleLedgerFixture(t *testing.T) perfledger.Record {
	t.Helper()
	return perfledger.Record{Schema: perfledger.Schema, RequestID: "measured-request", Model: "Qwen-fixture", Identity: perfledger.Identity{Planner: "inkernel", Backend: "vulkan", Version: "test-source"}, NativeTiming: &perfledger.NativeTiming{Kernels: &perfledger.NativeKernels{Source: perfledger.KernelSourceStructured, Backend: "vulkan", DecodeSteps: 31, GPUMSPerToken: 2, WallMSPerToken: 3, Complete: true, Pipelining: perfledger.KernelPipeliningDisabledByProfile, SelectedKernels: []perfledger.SelectedKernel{{Kernel: "iq2xxs_matvec", Groups: "640x1", CallsPerToken: 7, USPerCall: 80, Share: 0.5}, {Kernel: "unknown_kernel", Groups: "1x1", CallsPerToken: 1, USPerCall: 1, Share: 0.01}}}}}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleStructuredLedgerSelectsExactWindowAndPreservesLineage(t *testing.T) {
	r := oracleLedgerFixture(t)
	warm := r
	warm.RequestID = "warm-request"
	a, _ := json.Marshal(warm)
	b, _ := json.Marshal(r)
	stages, err := parseOracleLedger(string(a)+"\n"+string(b)+"\n", r.RequestID, "/models/Qwen-fixture.gguf")
	if err != nil {
		t.Fatal(err)
	}
	shapes := oracleShapeList{Shapes: []oracleShape{{Op: "MUL_MAT", Dtype: "iq2_xxs", M: 10240, K: 5120, N: 1}}}
	rec := rankKernelOracle(oracleRankInput{shapes: shapes, stages: stages})
	row := findOracleRow(t, rec.Rows, "MUL_MAT", "iq2_xxs", "10240x5120", 1)
	if row.FakUS == nil || *row.FakUS != 80 || row.Status != oracleStatusGGMLMissing || row.FakSource != "structured_perf_ledger" {
		t.Fatal("device observation missing or absent ggml zero-filled")
	}
	st := rec.FakStages[0]
	if st.Ledger.Line != 2 || st.Ledger.RequestID != r.RequestID || st.Ledger.CallsPerToken != 7 || st.CallsBatch != 0 || st.Ledger.Method != "structured_device_timestamp_intervals" || st.JoinedShape == "" {
		t.Fatalf("lineage or geometry lost: %+v", st)
	}
	if rec.FakStages[1].Unjoined != "kernel_geometry_unknown" {
		t.Fatal("unknown kernel inferred")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleStructuredLedgerUnverifiedObservationsRemainUnjoined(t *testing.T) {
	for _, name := range []string{"model", "source", "pipeline", "incomplete", "untimed", "elided", "zero_time", "invalid_groups", "zero_calls", "native_identity"} {
		t.Run(name, func(t *testing.T) {
			r := oracleLedgerFixture(t)
			k := r.NativeTiming.Kernels
			switch name {
			case "model":
				r.Model = "different-model"
			case "source":
				k.Source = "unknown"
			case "pipeline":
				k.Pipelining = "unknown"
			case "incomplete":
				k.Complete = false
			case "untimed":
				k.UntimedCompute = 1
			case "elided":
				k.BarriersElided = 1
			case "zero_time":
				k.GPUMSPerToken = 0
			case "invalid_groups":
				k.SelectedKernels[0].Groups = "unknown"
			case "zero_calls":
				k.SelectedKernels[0].CallsPerToken = 0
			case "native_identity":
				r.Backend = "cpu-ref"
			}
			raw, _ := json.Marshal(r)
			stages, err := parseOracleLedger(string(raw), r.RequestID, "/models/Qwen-fixture.gguf")
			if err != nil || len(stages) != 2 || stages[0].Unjoined == "" {
				t.Fatalf("unverified row accepted: stages=%+v err=%v", stages, err)
			}
			rec := rankKernelOracle(oracleRankInput{shapes: oracleShapeList{Shapes: []oracleShape{{Op: "MUL_MAT", Dtype: "iq2_xxs", M: 10240, K: 5120, N: 1}}}, stages: stages})
			if rec.Rows[0].FakUS != nil || rec.Rows[0].Status != oracleStatusBothMissing {
				t.Fatal("unverified observation gained rank timing")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleStructuredLedgerRejectsAmbiguousOrAbsentWindow(t *testing.T) {
	r := oracleLedgerFixture(t)
	raw, _ := json.Marshal(r)
	for _, text := range []string{string(raw) + "\n" + string(raw), "{", `{"schema":"unknown","request_id":"measured-request"}`, ""} {
		if _, err := parseOracleLedger(text, r.RequestID, "Qwen-fixture"); !errors.Is(err, errKernelOracleUsage) {
			t.Fatal("missing or ambiguous window accepted")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelOracleStructuredLedgerCLIHashesRawInput(t *testing.T) {
	r := oracleLedgerFixture(t)
	line, _ := json.Marshal(r)
	path := filepath.Join(t.TempDir(), "on.perf.jsonl")
	if err := os.WriteFile(path, append(line, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	args := []string{"kernel-oracle", "rank", "--shapes", kernelOracleShapesFile(t), "--fak-ledger", path, "--fak-ledger-request", r.RequestID}
	if code := run(&out, &stderr, args); code != 0 {
		t.Fatal(stderr.String())
	}
	var rec oracleReceipt
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatal(err)
	}
	if rec.Inputs["fak_ledger"] == "" || len(rec.FakStages) != 2 || rec.FakStages[0].Unjoined != "ledger_model_mismatch" {
		t.Fatal("raw input hash or model mismatch missing")
	}
	out.Reset()
	mixed := append(append([]string{}, args...), "--ggml", filepath.Join("testdata", "kernel_oracle", "ggml-perf.txt"))
	if code := run(&out, &stderr, mixed); code != 2 || out.Len() != 0 {
		t.Fatal("incompatible host and device timers gained comparison output")
	}
	out.Reset()
	if code := run(&out, &stderr, args[:len(args)-2]); code == 0 {
		t.Fatal("implicit window accepted")
	}
}
