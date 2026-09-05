package roofline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFold_CurrentPickedPendingAndTarget is the core fold contract: current is
// picked from a matching real artifact, the best (highest) value wins, a lane
// with no matching artifact is PENDING, and a synthetic/failed artifact never
// fills a real-model lane.
func TestFold_CurrentPickedPendingAndTarget(t *testing.T) {
	lanes := []Lane{
		{ID: "A", Kind: KindDecodeSingle, RequireReal: true, Ceiling: Ceiling{Lo: 150, Hi: 200}},
		{ID: "B", Kind: KindAggregate, RequireReal: true, Ceiling: Ceiling{Lo: 11000, Hi: 14000}},
		{ID: "D", Kind: KindPrefill, RequireReal: true, Ceiling: Ceiling{Lo: 11000, Hi: 14000}},
	}
	arts := []Artifact{
		{RunID: "real-decode-a", Real753B: true, Meas: []Measurement{{Kind: KindDecodeSingle, TokS: 23.38, Witness: "layer"}}},
		{RunID: "real-decode-b", Real753B: true, Meas: []Measurement{{Kind: KindDecodeSingle, TokS: 20.0, Witness: "worse"}}},
		{RunID: "synthetic-prefill", Synthetic: true, Meas: []Measurement{{Kind: KindPrefill, TokS: 999.0, Witness: "synthetic"}}},
		{RunID: "wedged-agg", Real753B: true, Failed: true, Meas: []Measurement{{Kind: KindAggregate, TokS: 5.0}}},
	}

	rows := Fold(lanes, arts)
	if len(rows) != 3 {
		t.Fatalf("want 3 rows, got %d", len(rows))
	}
	byID := map[string]Row{}
	for _, r := range rows {
		byID[r.Lane.ID] = r
	}

	// Lane A: current is the BEST real decode measurement (23.38), witnessed by run-id.
	a := byID["A"]
	if a.Status != StatusMeasured {
		t.Fatalf("lane A: want MEASURED, got %s", a.Status)
	}
	if a.Current != 23.38 {
		t.Errorf("lane A current: want 23.38 (the best real artifact), got %v", a.Current)
	}
	if a.WitnessRun != "real-decode-a" {
		t.Errorf("lane A witness run: want real-decode-a, got %q", a.WitnessRun)
	}

	// Lane B: only a WEDGED (failed) real artifact exists -> PENDING, not 5.0.
	if b := byID["B"]; b.Status != StatusPending || b.Current != 0 {
		t.Errorf("lane B: want PENDING/0 (wedged artifact must not count), got %s/%v", b.Status, b.Current)
	}

	// Lane D: only a SYNTHETIC prefill exists -> PENDING (real required), not 999.
	if d := byID["D"]; d.Status != StatusPending || d.Current != 0 {
		t.Errorf("lane D: want PENDING/0 (synthetic must not fill a real lane), got %s/%v", d.Status, d.Current)
	}
}

// TestTargetIsEightyPercentOfCeiling asserts the 80%-target invariant on the
// real lane spec: target = 0.8 × ceiling for every numeric-ceiling lane, and the
// host-bound lane exposes no numeric target.
func TestTargetIsEightyPercentOfCeiling(t *testing.T) {
	for _, ln := range Lanes() {
		c := ln.Ceiling
		if c.HostBound {
			// host-bound lane: no numeric roofline, target math is 0×0.
			if c.Lo != 0 || c.Hi != 0 {
				t.Errorf("lane %s: host-bound lane should carry no numeric ceiling, got %v–%v", ln.ID, c.Lo, c.Hi)
			}
			continue
		}
		if got, want := c.TargetLo(), 0.8*c.Lo; math.Abs(got-want) > 1e-9 {
			t.Errorf("lane %s TargetLo: want 0.8*%v=%v, got %v", ln.ID, c.Lo, want, got)
		}
		if got, want := c.TargetHi(), 0.8*c.Hi; math.Abs(got-want) > 1e-9 {
			t.Errorf("lane %s TargetHi: want 0.8*%v=%v, got %v", ln.ID, c.Hi, want, got)
		}
	}
	// Spot-check the documented numbers: Lane A 150–200 -> 120–160.
	a := Lanes()[0]
	if a.ID != "A" || a.Ceiling.TargetLo() != 120 || a.Ceiling.TargetHi() != 160 {
		t.Errorf("lane A 80%% target: want 120–160 from 150–200, got %v–%v", a.Ceiling.TargetLo(), a.Ceiling.TargetHi())
	}
}

// TestLoadArtifacts_NormalizesSchemas writes a small synthetic set of run
// artifacts mirroring the three real schemas (verdict A/B, context curve, engine
// wedge) plus a non-GLM run, and asserts the loader normalizes each correctly.
func TestLoadArtifacts_NormalizesSchemas(t *testing.T) {
	root := t.TempDir()
	// The ceiling doc must exist for Generate; LoadArtifacts only needs the runs tree.
	mustWrite(t, filepath.Join(root, filepath.FromSlash(CeilingDocRel)),
		"---\ntitle: \"Test ceiling doc\"\n---\nbody\n")

	// 1) real 753B single-stream decode A/B (verdict schema).
	writeRun(t, root, "a100", "l1ab", `{
      "run_id":"a100-glm52-l1-rowsplit-ab-TEST","machine_id":"a100","timestamp":"2026-07-09T23:42:40Z",
      "summary":"Real 753B GLM-5.2 decode","scope":"REAL-753B;UD-Q4_K_M",
      "verdict":{"winner":"layer","decode_toks_layer":23.38,"decode_toks_row":7.129}}`)

	// 2) synthetic glm_moe_dsa context curve.
	writeRun(t, root, "a100", "dsa", `{
      "run_id":"a100-glm52-dsa-decode-TEST","machine_id":"a100","timestamp":"2026-07-09T23:36:30Z",
      "summary":"Synthetic GLM-5.2-shaped (glm_moe_dsa), NOT-the-753B","scope":"synthetic-weights;NOT-the-753B",
      "curve":[
        {"prompt_len":128,"status":"collected","decode_tok_s":26.61,"prefill_tok_s":35.87},
        {"prompt_len":512,"status":"collected","decode_tok_s":20.9,"prefill_tok_s":24.74},
        {"prompt_len":2048,"status":"aborted"}]}`)

	// 3) real GLM-5.2 cpu serve that wedged (engine schema, ok=false).
	writeRun(t, root, "cpu-server-a", "wedge", `{
      "schema":"fak.cpu-serve-bench.v1","machine_id":"cpu-server-a","model":"GLM-5.2-UD-Q4_K_M",
      "ok":false,"headline":"NO usable throughput tok/s obtained. fak-native CPU serve WEDGED on host RAM.",
      "engines":{"fak-native":{"ok":false,"decode_tok_per_sec":null}}}`)

	// 4) a non-GLM run that must be ignored entirely.
	writeRun(t, root, "gcp-g2-l4", "qwen", `{
      "run_id":"gcp-qwen-TEST","machine_id":"gcp-g2-l4","summary":"qwen2.5 decode",
      "curve":[{"status":"collected","decode_tok_s":88.0}]}`)

	arts, err := LoadArtifacts(root)
	if err != nil {
		t.Fatalf("LoadArtifacts: %v", err)
	}
	if len(arts) != 3 {
		t.Fatalf("want 3 GLM artifacts (qwen skipped), got %d: %+v", len(arts), arts)
	}
	byRun := map[string]Artifact{}
	for _, a := range arts {
		byRun[a.RunID] = a
	}

	// verdict schema -> real, single decode = best of layer/row.
	l1 := byRun["a100-glm52-l1-rowsplit-ab-TEST"]
	if !l1.Real753B || l1.Synthetic {
		t.Errorf("l1ab: want real/non-synthetic, got real=%v synthetic=%v", l1.Real753B, l1.Synthetic)
	}
	if v, ok := measOf(l1, KindDecodeSingle); !ok || v != 23.38 {
		t.Errorf("l1ab decode: want 23.38, got %v (ok=%v)", v, ok)
	}

	// curve schema -> synthetic, decode+prefill best points.
	dsa := byRun["a100-glm52-dsa-decode-TEST"]
	if !dsa.Synthetic || dsa.Real753B {
		t.Errorf("dsa: want synthetic/non-real, got synthetic=%v real=%v", dsa.Synthetic, dsa.Real753B)
	}
	if v, ok := measOf(dsa, KindDecodeSingle); !ok || v != 26.61 {
		t.Errorf("dsa decode: want best 26.61, got %v (ok=%v)", v, ok)
	}
	if v, ok := measOf(dsa, KindPrefill); !ok || v != 35.87 {
		t.Errorf("dsa prefill: want best 35.87, got %v (ok=%v)", v, ok)
	}

	// engine wedge -> real, failed, no measurement.
	w := byRun["cpu-server-a-wedge-result"]
	// run_id absent in the wedge JSON, so the loader falls back to the path.
	if w.RunID == "" {
		// find the wedge by its Failed flag instead.
		for _, a := range arts {
			if a.Failed && a.MachineID == "cpu-server-a" {
				w = a
			}
		}
	}
	if !w.Failed || !w.Real753B || len(w.Meas) != 0 {
		t.Errorf("wedge: want real/failed/no-meas, got failed=%v real=%v meas=%d", w.Failed, w.Real753B, len(w.Meas))
	}
}

// TestGenerate_TempEndToEnd folds the synthetic set end to end and checks the
// dashboard: Lane A measured, others pending, synthetic + failed surfaced.
func TestGenerate_TempEndToEnd(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, filepath.FromSlash(CeilingDocRel)),
		"---\ntitle: \"Test ceiling doc\"\n---\nbody\n")
	writeRun(t, root, "a100", "l1ab", `{
      "run_id":"real-l1ab","machine_id":"a100","summary":"Real 753B GLM-5.2","scope":"REAL-753B",
      "verdict":{"winner":"layer","decode_toks_layer":23.38,"decode_toks_row":7.129}}`)
	writeRun(t, root, "a100", "dsa", `{
      "run_id":"synthetic-dsa","machine_id":"a100","summary":"glm_moe_dsa synthetic NOT-the-753B",
      "scope":"synthetic-weights","curve":[{"status":"collected","decode_tok_s":26.61,"prefill_tok_s":35.87}]}`)

	d, err := Generate(root)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if d.CeilingDocTitle != "Test ceiling doc" {
		t.Errorf("ceiling title: want %q, got %q", "Test ceiling doc", d.CeilingDocTitle)
	}
	if d.Measured() != 1 || d.Pending() != 3 {
		t.Errorf("want 1 measured / 3 pending, got %d / %d", d.Measured(), d.Pending())
	}
	if len(d.Synthetic) != 1 {
		t.Errorf("want 1 synthetic witness surfaced, got %d", len(d.Synthetic))
	}
	md := d.Markdown()
	for _, want := range []string{"23.4", "PENDING", "120–160", "150–200", "8800–11200", "host-bound", "synthetic-dsa"} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown missing %q", want)
		}
	}
}

// TestGenerateRealDashboardDoc runs the generator against the real repo tree. By
// default it verifies the fold (Lane A measured from the real l1-rowsplit run);
// with ROOFLINE_WRITE_DOC set it (re)writes the committed dashboard doc. It skips
// cleanly when the real artifacts are not present, so it never fails in isolation.
func TestGenerateRealDashboardDoc(t *testing.T) {
	root, ok := repoRoot()
	if !ok {
		t.Skip("repo root (go.mod) not found from test dir")
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(RunsRel))); err != nil {
		t.Skipf("real runs tree not present: %v", err)
	}
	d, err := Generate(root)
	if err != nil {
		t.Skipf("real ceiling doc not present: %v", err)
	}
	if len(d.Rows) != 4 {
		t.Fatalf("want 4 lanes, got %d", len(d.Rows))
	}
	// Lane A must be measured from the real l1-rowsplit artifact (~23.4 tok/s).
	a := d.Rows[0]
	if a.Lane.ID != "A" || a.Status != StatusMeasured {
		t.Fatalf("lane A: want MEASURED from real artifact, got %s/%s", a.Lane.ID, a.Status)
	}
	if a.Current < 20 || a.Current > 30 {
		t.Errorf("lane A current out of expected range for real GLM-5.2 decode: %v", a.Current)
	}

	if os.Getenv("ROOFLINE_WRITE_DOC") == "" {
		return
	}
	out := filepath.Join(root, filepath.FromSlash(DocOutRel))
	if err := os.WriteFile(out, []byte(d.Markdown()), 0o644); err != nil {
		t.Fatalf("write dashboard doc: %v", err)
	}
	t.Logf("wrote %s (%d measured / %d pending)", DocOutRel, d.Measured(), d.Pending())
}

// --- test helpers ----------------------------------------------------------

func measOf(a Artifact, k Kind) (float64, bool) {
	for _, m := range a.Meas {
		if m.Kind == k {
			return m.TokS, true
		}
	}
	return 0, false
}

func writeRun(t *testing.T, root, machine, name, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(RunsRel), machine, name, "result.json")
	mustWrite(t, p, body)
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func repoRoot() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// --- benchmarks -------------------------------------------------------------

var (
	benchSinkRows     []Row
	benchSinkString   string
	benchSinkArtifact Artifact
	benchSinkLanes    []Lane
)

func sampleBenchmarkArtifacts(n int) []Artifact {
	arts := make([]Artifact, 0, n)
	for i := 0; i < n; i++ {
		switch i % 4 {
		case 0:
			arts = append(arts, Artifact{
				RunID:     fmt.Sprintf("run-decode-%03d", i),
				MachineID: "server-3",
				Real753B:  true,
				Meas: []Measurement{
					{Kind: KindDecodeSingle, TokS: 20.0 + float64(i%10), Witness: "-sm layer, batch=1"},
				},
			})
		case 1:
			arts = append(arts, Artifact{
				RunID:     fmt.Sprintf("run-agg-%03d", i),
				MachineID: "server-3",
				Real753B:  true,
				Meas: []Measurement{
					{Kind: KindAggregate, TokS: 10000.0 + float64(i*50), Witness: "concurrency 64"},
				},
			})
		case 2:
			arts = append(arts, Artifact{
				RunID:     fmt.Sprintf("run-synth-%03d", i),
				MachineID: "server-3",
				Synthetic: true,
				Meas: []Measurement{
					{Kind: KindPrefill, TokS: 500.0 + float64(i), Witness: "synthetic"},
				},
			})
		case 3:
			arts = append(arts, Artifact{
				RunID:     fmt.Sprintf("run-wedge-%03d", i),
				MachineID: "server-2",
				Real753B:  true,
				Failed:    true,
				FailNote:  "host RAM allocation failed",
			})
		}
	}
	return arts
}

// BenchmarkFoldRoofline measures the throughput of folding run artifacts against
// canonical drive lanes, scaling across small, medium, and large artifact pools.
func BenchmarkFoldRoofline(b *testing.B) {
	lanes := Lanes()
	for _, size := range []int{4, 20, 100} {
		b.Run(fmt.Sprintf("artifacts_%d", size), func(b *testing.B) {
			arts := sampleBenchmarkArtifacts(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkRows = Fold(lanes, arts)
			}
		})
	}
}

// BenchmarkRenderDashboard measures rendering the folded roofline dashboard
// to markdown, including lanes table, synthetic witnesses, and failure notes.
func BenchmarkRenderDashboard(b *testing.B) {
	lanes := Lanes()
	arts := sampleBenchmarkArtifacts(16)
	d := Dashboard{
		CeilingDoc:      CeilingDocRel,
		CeilingDocTitle: "GLM-5.2 GPU-server roofline dashboard: current vs 80%-target vs ceiling, one row per lane",
		Rows:            Fold(lanes, arts),
		GLMArtifacts:    len(arts),
	}
	for _, a := range arts {
		switch {
		case a.Synthetic:
			d.Synthetic = append(d.Synthetic, a)
		case a.Failed:
			d.FailedReal = append(d.FailedReal, a)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = d.Markdown()
	}
}

// BenchmarkNormalizeArtifact measures raw JSON result normalization across
// the three production schemas: verdict A/B, context curve, and engine wedge.
func BenchmarkNormalizeArtifact(b *testing.B) {
	verdictRaw := []byte(`{
		"run_id":"a100-glm52-l1-rowsplit-ab-TEST","machine_id":"a100","timestamp":"2026-07-09T23:42:40Z",
		"summary":"Real 753B GLM-5.2 decode","scope":"REAL-753B;UD-Q4_K_M",
		"verdict":{"winner":"layer","decode_toks_layer":23.38,"decode_toks_row":7.129}
	}`)
	var verdictMap map[string]any
	if err := json.Unmarshal(verdictRaw, &verdictMap); err != nil {
		b.Fatalf("unmarshal verdict JSON: %v", err)
	}

	curveRaw := []byte(`{
		"run_id":"a100-glm52-dsa-decode-TEST","machine_id":"a100","timestamp":"2026-07-09T23:36:30Z",
		"summary":"Synthetic GLM-5.2-shaped (glm_moe_dsa), NOT-the-753B","scope":"synthetic-weights;NOT-the-753B",
		"curve":[
			{"prompt_len":128,"status":"collected","decode_tok_s":26.61,"prefill_tok_s":35.87},
			{"prompt_len":512,"status":"collected","decode_tok_s":20.9,"prefill_tok_s":24.74},
			{"prompt_len":2048,"status":"aborted"}
		]
	}`)
	var curveMap map[string]any
	if err := json.Unmarshal(curveRaw, &curveMap); err != nil {
		b.Fatalf("unmarshal curve JSON: %v", err)
	}

	wedgeRaw := []byte(`{
		"schema":"fak.cpu-serve-bench.v1","machine_id":"cpu-server-a","model":"GLM-5.2-UD-Q4_K_M",
		"ok":false,"headline":"NO usable throughput tok/s obtained. fak-native CPU serve WEDGED on host RAM.",
		"engines":{"fak-native":{"ok":false,"decode_tok_per_sec":null}}
	}`)
	var wedgeMap map[string]any
	if err := json.Unmarshal(wedgeRaw, &wedgeMap); err != nil {
		b.Fatalf("unmarshal wedge JSON: %v", err)
	}

	cases := []struct {
		name string
		raw  []byte
		m    map[string]any
		rel  string
	}{
		{"verdict_ab", verdictRaw, verdictMap, "experiments/benchmark/runs/by-machine/a100/l1ab/result.json"},
		{"context_curve", curveRaw, curveMap, "experiments/benchmark/runs/by-machine/a100/dsa/result.json"},
		{"engine_wedge", wedgeRaw, wedgeMap, "experiments/benchmark/runs/by-machine/cpu-server-a/wedge/result.json"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			raw := tc.raw
			m := tc.m
			rel := tc.rel
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkArtifact = normalize(m, raw, rel)
			}
		})
	}
}

// BenchmarkFrontMatterTitle measures front-matter title extraction speed on doc headers.
func BenchmarkFrontMatterTitle(b *testing.B) {
	doc := "---\ntitle: \"GLM-5.2 GPU-server roofline dashboard: current vs 80%-target vs ceiling, one row per lane\"\ndescription: \"A deterministic fold of run artifacts\"\n---\n# Roofline Dashboard\n"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = frontMatterTitle(doc)
	}
}

// BenchmarkLanes measures canonical drive lane construction and roofline ceiling allocation.
func BenchmarkLanes(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkLanes = Lanes()
	}
}
