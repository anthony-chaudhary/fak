package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureShape pins the shape-math expectations independently of any preset.
var fixtureShape = ModelShape{Name: "fixture", Layers: 10, IndexStepLayers: 5, KVAppendLayers: 10, AttendLayers: 10, RouterLayers: 8}

func TestGlueCensusShapeMath(t *testing.T) {
	inv := []GlueCallSite{
		{ID: "a", Class: GluePerIndexLayer, Factor: 1},
		{ID: "b", Class: GluePerIndexLayer, Factor: 2},
		{ID: "c", Class: GluePerKVLayer, Factor: 1},
		{ID: "d", Class: GluePerAttendLayer, Factor: 3},
		{ID: "e", Class: GluePerRouterLayer, Factor: 1},
		{ID: "f", Class: GluePerToken, Factor: 2},
		{ID: "g", Class: GluePrimitive, Factor: 9}, // must not be summed
	}
	want := 1*5 + 2*5 + 1*10 + 3*10 + 1*8 + 2 + 0
	if got := computeRoundTrips(inv, fixtureShape); got != want {
		t.Fatalf("computeRoundTrips = %d, want %d", got, want)
	}
}

func TestModelShapeNormalizeDefaultsAndBounds(t *testing.T) {
	s, err := (ModelShape{Name: "x", Layers: 7}).normalize()
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if s.IndexStepLayers != 7 || s.KVAppendLayers != 7 || s.AttendLayers != 7 || s.RouterLayers != 7 {
		t.Fatalf("defaults not filled to layers: %+v", s)
	}
	if _, err := (ModelShape{Layers: 0}).normalize(); err == nil {
		t.Fatal("layers=0 must fail")
	}
	if _, err := (ModelShape{Layers: 4, RouterLayers: 5}).normalize(); err == nil {
		t.Fatal("router_layers > layers must fail")
	}
}

// TestGlueCensusGLMDsaPresetCrossCheck runs the full census against the LIVE
// tree: zero drift (the inventory still matches trunk), and the computed
// round-trips/token lands within ±20% of the documented ~1100 hand-count. The
// reference is the documented constant; the computed figure comes only from
// inventory x shape.
func TestGlueCensusGLMDsaPresetCrossCheck(t *testing.T) {
	root := repoRoot()
	drift, occ := verifyInventory(root, glueCensusInventory)
	if len(drift) > 0 {
		for _, d := range drift {
			t.Errorf("drift %s/%s [%s]: %s", d.Site.File, d.Site.Func, d.Site.Pattern, d.Reason)
		}
		return
	}
	shape, err := glueCensusPresets["glm-dsa"].normalize()
	if err != nil {
		t.Fatalf("preset normalize: %v", err)
	}
	total := computeRoundTrips(glueCensusInventory, shape)
	xc := crossCheck(total, documentedGLMDsaRoundTrips)
	if !xc.Pass {
		t.Fatalf("cross-check failed: computed=%d ref=%d delta=%+.2f%%", total, xc.Reference, xc.DeltaPct)
	}
	if total <= 0 {
		t.Fatal("computed total must be positive")
	}
	// Every counted row must have been observed at least once in the live source.
	for _, s := range glueCensusInventory {
		if s.Class == GluePrimitive {
			continue
		}
		if occ[s.ID] < 1 {
			t.Errorf("counted row %s has no live occurrence", s.ID)
		}
	}
}

func writeFixtureGo(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const fixtureGLMSession = `package model

func glmDsaIndexStep(xn []float32) int {
	qResid := proj(ap+"q_a_proj.weight", xn)
	k := proj(ap+"indexer.wk.weight", xn)
	if ik.indexSelect(qResid, k) {
		return 1
	}
	return 0
}

func glmDsaAppendAttentionKV(xn []float32) int {
	_ = proj(ap+"q_b_proj.weight", xn)
	return 1
}
`

func TestGlueCensusDriftDetection(t *testing.T) {
	dir := t.TempDir()
	writeFixtureGo(t, dir, "session.go", fixtureGLMSession)
	inv := []GlueCallSite{
		{ID: "ok-wk", File: "session.go", Func: "glmDsaIndexStep", Pattern: "indexer.wk.weight", Class: GluePerIndexLayer, Factor: 1},
		{ID: "gone-pattern", File: "session.go", Func: "glmDsaIndexStep", Pattern: "indexer.weights_proj.weight", Class: GluePerIndexLayer, Factor: 1},
		{ID: "gone-func", File: "session.go", Func: "glmDsaOldStep", Pattern: "whatever", Class: GluePerIndexLayer, Factor: 1},
		{ID: "gone-file", File: "nope.go", Func: "", Pattern: "x", Class: GluePerToken, Factor: 1},
	}
	drift, _ := verifyInventory(dir, inv)
	reasons := map[string]string{}
	for _, d := range drift {
		reasons[d.Site.ID] = d.Reason
	}
	if reasons["gone-pattern"] != "pattern-absent" {
		t.Errorf("gone-pattern reason = %q, want pattern-absent (reasons=%v)", reasons["gone-pattern"], reasons)
	}
	if reasons["gone-func"] != "func-missing" {
		t.Errorf("gone-func reason = %q, want func-missing", reasons["gone-func"])
	}
	if reasons["gone-file"] != "file-missing" {
		t.Errorf("gone-file reason = %q, want file-missing", reasons["gone-file"])
	}
	if _, ok := reasons["ok-wk"]; ok {
		t.Error("ok-wk must not drift")
	}
}

const fixtureKernelCovered = `package model

func mul(name string, xt []float32) []float32 {
	y := be.Read(be.MatMul(wt, xt))
	return y
}

func rogueRead(name string) []float32 {
	return be.Read(rogueTensor)
}
`

func TestGlueCensusDiscoveryUncovered(t *testing.T) {
	dir := t.TempDir()
	writeFixtureGo(t, dir, "kernel.go", fixtureKernelCovered)
	// Cover only mul; rogueRead's be.Read( must surface as uncovered.
	inv := []GlueCallSite{{ID: "seam-mul", File: "kernel.go", Func: "mul", Pattern: "be.Read(be.MatMul(wt, xt))", Class: GluePrimitive}}
	// discoverUncovered reads the package-level file list; scope it to the fixture.
	prev := glueCensusDiscoverFiles
	glueCensusDiscoverFiles = []string{"kernel.go"}
	defer func() { glueCensusDiscoverFiles = prev }()

	uncovered := discoverUncovered(dir, inv)
	if len(uncovered) != 1 {
		t.Fatalf("uncovered = %d sites (%+v), want exactly 1 (rogueRead)", len(uncovered), uncovered)
	}
	u := uncovered[0]
	if u.File != "kernel.go" || u.Marker != "be.Read(" || !strings.Contains(u.Snippet, "rogueTensor") {
		t.Fatalf("wrong uncovered site: %+v", u)
	}
	if u.Line != 9 {
		t.Errorf("uncovered line = %d, want 9 (inside rogueRead)", u.Line)
	}
}

// TestGlueCensusModelShapeJSONFlag pins the --model-shape JSON flag end to end
// (computed = 5x10 index + 4x40 kv + 2x40 attend + 1x40 router = 330).
func TestGlueCensusModelShapeJSONFlag(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunGlueCensus(&out, &errb, []string{
		"--root", repoRoot(),
		`--model-shape={"layers":40,"index_step_layers":10}`, "--check", "330", "--no-discover",
	})
	if code != 0 {
		t.Fatalf("RunGlueCensus exit = %d, stderr=%q output:\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{
		"TOTAL estimated synchronous host<->device round-trips / decoded token: 330",
		"cross-check vs documented ref=330: PASS delta=+0.00%",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

// TestGlueCensusPresetDefaultCounters pins the full CLI path for a preset that
// leaves the per-class layer counters zero: they must default to layers BEFORE
// the math runs (regression: total was once computed on the pre-normalized
// shape and reported 0).
func TestGlueCensusPresetDefaultCounters(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunGlueCensus(&out, &errb, []string{"--root", repoRoot(), "--preset", "tiny-4", "--no-discover"})
	if code != 0 {
		t.Fatalf("RunGlueCensus exit = %d, stderr=%q output:\n%s", code, errb.String(), out.String())
	}
	want := "TOTAL estimated synchronous host<->device round-trips / decoded token: 48"
	if !strings.Contains(out.String(), want) {
		t.Fatalf("output missing %q:\n%s", want, out.String())
	}
}

func TestGlueCensusCrossCheckBounds(t *testing.T) {
	cases := []struct {
		computed, ref int
		pass          bool
	}{
		{1104, 1100, true},  // the GLM-DSA census result
		{880, 1100, true},   // exactly -20%
		{1320, 1100, true},  // exactly +20%
		{879, 1100, false},  // just outside
		{1321, 1100, false}, // just outside
		{500, 1100, false},
	}
	for _, tc := range cases {
		res := crossCheck(tc.computed, tc.ref)
		if res.Pass != tc.pass {
			t.Errorf("crossCheck(%d,%d).Pass = %v, want %v (delta=%+.2f%%)", tc.computed, tc.ref, res.Pass, tc.pass, res.DeltaPct)
		}
	}
}
