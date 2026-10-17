package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
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
