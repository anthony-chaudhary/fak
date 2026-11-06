package devcmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureLaunchShape pins the shape-math expectations independently of any preset.
var fixtureLaunchShape = LaunchShape{Name: "fixture", Layers: 10, IndexStepLayers: 5, KVAppendLayers: 10, AttendLayers: 10, MoELayers: 8, ActiveExperts: 4}

func TestLaunchCensusShapeMath(t *testing.T) {
	inv := []LaunchCallSite{
		{ID: "a", Class: LaunchPerIndexLayer, Factor: 1},
		{ID: "b", Class: LaunchPerKVLayer, Factor: 2},
		{ID: "c", Class: LaunchPerAttendLayer, Factor: 1},
		{ID: "d", Class: LaunchPerMoELayer, Factor: 1},
		{ID: "e", Variant: variantPerExpert, Class: LaunchPerActiveExpert, Factor: 3},
		{ID: "f", Variant: variantBatched, Class: LaunchPerMoELayer, Factor: 3},
		{ID: "g", Class: LaunchPerToken, Factor: 2},
		{ID: "h", Class: LaunchPerRequest, Factor: 1},
		{ID: "i", Class: LaunchPrimitive, Factor: 9}, // must not be summed
	}
	// looped: 1*5 + 2*10 + 1*10 + 1*8 + 3*4*8 + 2 + 1 = 142 (batched row excluded)
	if got := computeLaunches(inv, fixtureLaunchShape, variantPerExpert); got != 142 {
		t.Fatalf("looped total = %d, want 142", got)
	}
	// batched: 1*5 + 2*10 + 1*10 + 1*8 + 3*8 + 2 + 1 = 70 (looped row excluded)
	if got := computeLaunches(inv, fixtureLaunchShape, variantBatched); got != 70 {
		t.Fatalf("batched total = %d, want 70", got)
	}
	// Per-MoE-layer expert dispatch: looped = 3 x 4 experts; batched = 3 flat.
	if got := expertDispatchPerMoELayer(inv, fixtureLaunchShape, variantPerExpert); got != 12 {
		t.Fatalf("looped expert/MoE layer = %d, want 12", got)
	}
	if got := expertDispatchPerMoELayer(inv, fixtureLaunchShape, variantBatched); got != 3 {
		t.Fatalf("batched expert/MoE layer = %d, want 3", got)
	}
}

// fak-test:runtime fast est=1ms
func TestExpertDispatchPerMoELayerMatchingVariantClasses(t *testing.T) {
	t.Parallel()
	shape := LaunchShape{ActiveExperts: 4}
	cases := []struct {
		class LaunchClass
		want  int
	}{
		{LaunchPerIndexLayer, 0},
		{LaunchPerKVLayer, 0},
		{LaunchPerAttendLayer, 0},
		{LaunchPerMoELayer, 3},
		{LaunchPerActiveExpert, 12},
		{LaunchPerToken, 0},
		{LaunchPerRequest, 0},
		{LaunchPrimitive, 0},
		{LaunchClass("future-class"), 0},
	}
	for _, variants := range []struct{ selected, other string }{
		{variantPerExpert, variantBatched},
		{variantBatched, variantPerExpert},
	} {
		for _, tc := range cases {
			t.Run(variants.selected+"/"+string(tc.class), func(t *testing.T) {
				inv := []LaunchCallSite{
					{Class: tc.class, Variant: variants.selected, Factor: 3},
					{Class: LaunchPerActiveExpert, Variant: variants.other, Factor: 100},
					{Class: LaunchPerActiveExpert, Variant: launchVariantShared, Factor: 100},
				}
				if got := expertDispatchPerMoELayer(inv, shape, variants.selected); got != tc.want {
					t.Fatalf("expert subtotal = %d, want %d for matching class %q", got, tc.want, tc.class)
				}
			})
		}
	}
}

func TestLaunchShapeNormalizeDefaultsAndBounds(t *testing.T) {
	s, err := (LaunchShape{Name: "x", Layers: 7, MoELayers: 3, ActiveExperts: 2}).normalize()
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if s.IndexStepLayers != 7 || s.KVAppendLayers != 7 || s.AttendLayers != 7 {
		t.Fatalf("defaults not filled to layers: %+v", s)
	}
	if _, err := (LaunchShape{Layers: 0}).normalize(); err == nil {
		t.Fatal("layers=0 must fail")
	}
	if _, err := (LaunchShape{Layers: 4, MoELayers: 5, ActiveExperts: 1}).normalize(); err == nil {
		t.Fatal("moe_layers > layers must fail")
	}
	if _, err := (LaunchShape{Layers: 4, MoELayers: 4}).normalize(); err == nil {
		t.Fatal("moe_layers>0 without active_experts must fail (top-k is a census input)")
	}
	// Dense-only shapes (moe_layers=0 AND no top-k) have no MoE work; the
	// all-zero MoE pair is the dense reading (a JSON shape cannot distinguish
	// an omitted moe_layers from an explicit 0).
	dense, err := (LaunchShape{Layers: 4, MoELayers: 0}).normalize()
	if err != nil || dense.MoELayers != 0 || dense.ActiveExperts != 0 {
		t.Fatalf("dense shape must normalize to no MoE work: %+v (%v)", dense, err)
	}
	// A stated top-k with zero moe_layers means every layer is MoE.
	s, err = (LaunchShape{Layers: 4, ActiveExperts: 2}).normalize()
	if err != nil || s.MoELayers != 4 {
		t.Fatalf("zero moe_layers + top-k must default to all layers: %+v (%v)", s, err)
	}
}

// TestLaunchCensusGLMDsaPresetCrossCheck runs the full census against the LIVE
// tree: zero drift (the inventory still matches trunk), every counted row
// observed, and the computed looped expert-dispatch count per MoE layer lands
// within ±20% of the documented ~24 hand-count. The reference is the
// documented constant; the computed figure comes only from inventory x shape.
func TestLaunchCensusGLMDsaPresetCrossCheck(t *testing.T) {
	root := repoRoot()
	drift, occ := verifyLaunchInventory(root, launchCensusInventory)
	if len(drift) > 0 {
		for _, d := range drift {
			t.Errorf("drift %s/%s [%s]: %s", d.Site.File, d.Site.Func, d.Site.Pattern, d.Reason)
		}
		return
	}
	shape, err := launchCensusPresets["glm-dsa"].normalize()
	if err != nil {
		t.Fatalf("preset normalize: %v", err)
	}
	computed := expertDispatchPerMoELayer(launchCensusInventory, shape, variantPerExpert)
	xc := crossCheck(computed, documentedGLMDsaExpertParFors)
	if !xc.Pass {
		t.Fatalf("cross-check failed: computed=%d ref=%d delta=%+.2f%%", computed, xc.Reference, xc.DeltaPct)
	}
	looped := computeLaunches(launchCensusInventory, shape, variantPerExpert)
	batched := computeLaunches(launchCensusInventory, shape, variantBatched)
	if looped <= 0 || batched >= looped {
		t.Fatalf("totals implausible: looped=%d batched=%d", looped, batched)
	}
	// Every counted row must have been observed at least once in the live source.
	for _, s := range launchCensusInventory {
		if s.Class == LaunchPrimitive {
			continue
		}
		if occ[s.ID] < 1 {
			t.Errorf("counted row %s has no live occurrence", s.ID)
		}
	}
}

func writeFixtureGoFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const fixtureLaunchSession = `package model

func expertSwiGLU(xn []float32) []float32 {
	gu := mulGroup(mat, []string{gn, un}, xn, []int{I, I}, H)
	_ = gu
	return mat.mul(dn, xn, H, I)
}

func batchExpertRows(K, rowsPer, inDim int, fn func(e, rlo, rhi int)) {
	total := K * rowsPer
	body := func(lo, hi int) {}
	parForRange(total, total*inDim, body)
}

func rogueDispatch(xn []float32) {
	parFor(8, 4, func(lo, hi int) {})
}
`

func TestLaunchCensusDriftDetection(t *testing.T) {
	dir := t.TempDir()
	writeFixtureGoFile(t, dir, "session.go", fixtureLaunchSession)
	inv := []LaunchCallSite{
		{ID: "ok-gateup", File: "session.go", Func: "expertSwiGLU", Pattern: "mulGroup(mat, []string{gn, un}", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 2},
		{ID: "gone-pattern", File: "session.go", Func: "expertSwiGLU", Pattern: "mat.mul(downX", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 1},
		{ID: "gone-func", File: "session.go", Func: "oldFunc", Pattern: "whatever", Class: LaunchPerMoELayer, Factor: 1},
		{ID: "gone-file", File: "nope.go", Func: "", Pattern: "x", Class: LaunchPerToken, Factor: 1},
	}
	drift, _ := verifyLaunchInventory(dir, inv)
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
	if _, ok := reasons["ok-gateup"]; ok {
		t.Error("ok-gateup must not drift")
	}
}

// TestLaunchCensusAddedSiteChangesTotal is the new-regression signal: a launch
// site ADDED to the decode path moves the computed launches/token baseline by
// exactly its factor x its class counter — the prose "~24/layer" estimate had
// no such property.
func TestLaunchCensusAddedSiteChangesTotal(t *testing.T) {
	dir := t.TempDir()
	writeFixtureGoFile(t, dir, "session.go", fixtureLaunchSession)
	base := []LaunchCallSite{
		{ID: "ok-gateup", File: "session.go", Func: "expertSwiGLU", Pattern: "mulGroup(mat, []string{gn, un}", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 2},
	}
	shape := LaunchShape{Name: "fixture", Layers: 4, MoELayers: 4, ActiveExperts: 2}
	before := computeLaunches(base, shape, variantPerExpert)
	// A new per-MoE-layer dispatch site lands in the decode path.
	added := LaunchCallSite{ID: "new-site", File: "session.go", Func: "rogueDispatch", Pattern: "parFor(8, 4", Class: LaunchPerMoELayer, Factor: 1}
	afterInv := append(append([]LaunchCallSite{}, base...), added)
	after := computeLaunches(afterInv, shape, variantPerExpert)
	if after-before != shape.MoELayers {
		t.Fatalf("added site moved total by %d, want %d (one per MoE layer)", after-before, shape.MoELayers)
	}
	// The added site verifies cleanly against the fixture tree (it is real code).
	drift, occ := verifyLaunchInventory(dir, afterInv)
	if len(drift) > 0 {
		t.Fatalf("unexpected drift: %+v", drift)
	}
	if occ["new-site"] < 1 {
		t.Fatal("new-site must have a live occurrence")
	}
}

func TestLaunchCensusDiscoveryUncovered(t *testing.T) {
	dir := t.TempDir()
	writeFixtureGoFile(t, dir, "session.go", fixtureLaunchSession)
	// Cover expertSwiGLU and batchExpertRows; rogueDispatch's parFor must surface.
	inv := []LaunchCallSite{
		{ID: "ok-gateup", File: "session.go", Func: "expertSwiGLU", Pattern: "mulGroup(mat, []string{gn, un}", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 2},
		{ID: "ok-batched", File: "session.go", Func: "batchExpertRows", Pattern: "parForRange(total, total*inDim, body)", Class: LaunchPerMoELayer, Variant: variantBatched, Factor: 3},
	}
	prev := launchCensusDiscoverFiles
	launchCensusDiscoverFiles = []string{"session.go"}
	defer func() { launchCensusDiscoverFiles = prev }()

	uncovered := discoverLaunchUncovered(dir, inv)
	if len(uncovered) != 1 {
		t.Fatalf("uncovered = %d sites (%+v), want exactly 1 (rogueDispatch)", len(uncovered), uncovered)
	}
	u := uncovered[0]
	if u.File != "session.go" || u.Marker != "parFor(" || !strings.Contains(u.Snippet, "parFor(8, 4") {
		t.Fatalf("wrong uncovered site: %+v", u)
	}
	if u.Line != 16 {
		t.Errorf("uncovered line = %d, want 16 (inside rogueDispatch)", u.Line)
	}
}

// TestLaunchCensusModelShapeJSONFlag pins the --model-shape JSON flag +
// --check end to end: looped total = 5x10 index + 4x40 kv + 2x40 attend +
// 1x40 router + 3x4x40 looped experts = 810; cross-check computed 3x4=12
// vs ref 12 -> PASS.
func TestLaunchCensusModelShapeJSONFlag(t *testing.T) {
	var out, errb strings.Builder
	code := RunLaunchCensus(&out, &errb, []string{
		"--root", repoRoot(),
		`--model-shape={"layers":40,"index_step_layers":10,"kv_append_layers":40,"attend_layers":40,"moe_layers":40,"active_experts":4}`,
		"--check", "12", "--no-discover", "--json",
	})
	if code != 0 {
		t.Fatalf("RunLaunchCensus exit = %d, stderr=%q output:\n%s", code, errb.String(), out.String())
	}
	var doc LaunchCensusResult
	if err := json.Unmarshal([]byte(out.String()), &doc); err != nil {
		t.Fatalf("decode --json output: %v\n%s", err, out.String())
	}
	if doc.TotalLooped != 810 {
		t.Errorf("TotalLooped = %d, want 810", doc.TotalLooped)
	}
	if doc.TotalBatched != 450 {
		t.Errorf("TotalBatched = %d, want 450", doc.TotalBatched)
	}
	if doc.ExpertPerMoELayerLooped != 12 || doc.ExpertPerMoELayerBatched != 3 {
		t.Errorf("expert/MoE layer = %d/%d, want 12/3", doc.ExpertPerMoELayerLooped, doc.ExpertPerMoELayerBatched)
	}
	if doc.CrossCheck == nil || !doc.CrossCheck.Pass || doc.CrossCheck.DeltaPct != 0 {
		t.Errorf("cross-check = %+v, want PASS delta 0", doc.CrossCheck)
	}
}

// TestLaunchCensusPresetTotals pins the named presets' totals through the full
// CLI path (normalization before math, both variants, text rendering).
func TestLaunchCensusPresetTotals(t *testing.T) {
	cases := []struct {
		preset  string
		looped  int
		batched int
		expertL int
		expertB int
	}{
		{"tiny-4", 72, 60, 6, 3},
		{"glm-dsa", 3237, 1368, 24, 3},
	}
	for _, tc := range cases {
		var out, errb strings.Builder
		code := RunLaunchCensus(&out, &errb, []string{"--root", repoRoot(), "--preset", tc.preset, "--no-discover", "--json"})
		if code != 0 {
			t.Fatalf("%s: exit = %d, stderr=%q", tc.preset, code, errb.String())
		}
		var doc LaunchCensusResult
		if err := json.Unmarshal([]byte(out.String()), &doc); err != nil {
			t.Fatalf("%s: decode: %v", tc.preset, err)
		}
		if doc.TotalLooped != tc.looped || doc.TotalBatched != tc.batched {
			t.Errorf("%s: totals = %d/%d, want %d/%d", tc.preset, doc.TotalLooped, doc.TotalBatched, tc.looped, tc.batched)
		}
		if doc.ExpertPerMoELayerLooped != tc.expertL || doc.ExpertPerMoELayerBatched != tc.expertB {
			t.Errorf("%s: expert/MoE layer = %d/%d, want %d/%d", tc.preset, doc.ExpertPerMoELayerLooped, doc.ExpertPerMoELayerBatched, tc.expertL, tc.expertB)
		}
	}
	// The glm-dsa default cross-check (no explicit REF) asserts the documented
	// ~24 parFors/MoE-layer figure and must PASS on a live tree.
	var out, errb strings.Builder
	code := RunLaunchCensus(&out, &errb, []string{"--root", repoRoot(), "--preset", "glm-dsa", "--no-discover", "--check"})
	if code != 0 {
		t.Fatalf("glm-dsa --check exit = %d, stderr=%q output:\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{
		"cross-check vs documented ref=24 parFors/MoE-layer",
		"PASS delta=+0.00%",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}
