package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureH2DShape pins the shape-math expectations independently of any preset.
var fixtureH2DShape = H2DShape{Name: "fixture", HiddenSize: 10, Vocab: 100, Layers: 5, Batch: 2, SeqLen: 30, BlockTokens: 10}

// TestH2DShapeMath exercises every classification: per-token (with and without
// batch scaling), per-layer, per-request, and the zero-copy witness rows.
func TestH2DShapeMath(t *testing.T) {
	inv := []H2DCallSite{
		{ID: "pt", Class: H2DPerToken, Copies: 1, BytesExpr: H2DBytesHiddenRowF32},
		{ID: "pt-batch", Class: H2DPerToken, Copies: 1, BytesExpr: H2DBytesVocabF32, BatchScales: true},
		{ID: "pl-zero", Class: H2DPerLayer, Copies: 0}, // zero-copy per-layer seam
		{ID: "pt-zero", Class: H2DPerToken, Copies: 0}, // host-resident state
		{ID: "pr", Class: H2DPerRequest, Copies: 1, BytesExpr: H2DBytesBlockTableF32},
	}
	tot := computeH2DTotals(inv, fixtureH2DShape)
	// pt: 1 fire x 1 copy x 40 B; pt-batch: 2 fires x 1 copy x 400 B.
	if tot.CopiesPerToken != 3 {
		t.Errorf("copies/token = %d, want 3", tot.CopiesPerToken)
	}
	if tot.BytesPerToken != 40+800 {
		t.Errorf("bytes/token = %d, want 840", tot.BytesPerToken)
	}
	// pr: 1 copy x 4 x ceil(30/10) = 12 B/request; NOT summed into /token.
	if tot.RequestSyncCopies != 1 || tot.RequestSyncBytes != 12 {
		t.Errorf("request sync = %d copies / %d B, want 1 / 12", tot.RequestSyncCopies, tot.RequestSyncBytes)
	}
}

func TestH2DShapeNormalizeDefaultsAndBounds(t *testing.T) {
	s, err := (H2DShape{HiddenSize: 8, Vocab: 16, Layers: 3}).normalize()
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if s.Batch != 1 || s.SeqLen != 4096 || s.BlockTokens != 64 {
		t.Fatalf("defaults not filled: %+v", s)
	}
	for _, bad := range []H2DShape{
		{HiddenSize: 0, Vocab: 1, Layers: 1},
		{HiddenSize: 1, Vocab: 0, Layers: 1},
		{HiddenSize: 1, Vocab: 1, Layers: 0},
		{HiddenSize: 1, Vocab: 1, Layers: 1, Batch: -1},
		{HiddenSize: 1, Vocab: 1, Layers: 1, BlockTokens: 0, SeqLen: -1},
	} {
		if _, err := bad.normalize(); err == nil {
			t.Errorf("normalize(%+v) must fail", bad)
		}
	}
}

// TestH2DLiveTreeZeroDrift runs the full census against the LIVE tree: zero
// drift (the inventory still matches trunk) and every row observed at least
// once in the live source.
func TestH2DLiveTreeZeroDrift(t *testing.T) {
	root := repoRoot()
	drift, occ := verifyH2DInventory(root, h2dInventory)
	if len(drift) > 0 {
		for _, d := range drift {
			t.Errorf("drift %s/%s [%s]: %s", d.Site.File, d.Site.Func, d.Site.Pattern, d.Reason)
		}
		return
	}
	for _, s := range h2dInventory {
		if occ[s.ID] < 1 {
			t.Errorf("row %s has no live occurrence", s.ID)
		}
	}
}

// TestH2DPresetTiny4 pins the full CLI path for the default preset against the
// live tree: copies/token = 1 (hal embed row) + 1 (batch lane, batch=1) +
// 1 (split logits) = 3; bytes/token = 4x512 + 4x512 + 4x1024 = 8192; the paged
// block table stays request-scoped (4 x ceil(128/32) = 16 B/request).
func TestH2DPresetTiny4(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunH2DInventory(&out, &errb, []string{"--root", repoRoot(), "--preset", "tiny-4", "--no-discover"})
	if code != 0 {
		t.Fatalf("RunH2DInventory exit = %d, stderr=%q\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{
		"TOTAL estimated H2D copies / decoded token: 3",
		"TOTAL estimated H2D bytes / decoded token: 8192",
		"request-scoped (dirty-gated) H2D: 1 copy(ies), <= 16 B/request",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

// TestH2DPresetSmallBatch pins the batch-scaled math: batch=4 makes the
// per-lane embedding upload fire 4x (copies 1+4+1=6; bytes 16384 + 4x16384 +
// 4x151936 = 689664).
func TestH2DPresetSmallBatch(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunH2DInventory(&out, &errb, []string{"--root", repoRoot(), "--preset", "small-batch-4x16k", "--no-discover"})
	if code != 0 {
		t.Fatalf("RunH2DInventory exit = %d, stderr=%q\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{
		"TOTAL estimated H2D copies / decoded token: 6",
		"TOTAL estimated H2D bytes / decoded token: 689664",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

// TestH2DModelShapeJSONFlag pins the --model-shape JSON flag end to end:
// copies = 1 + 3 (batch lanes) + 1 = 5; bytes = 1024 + 3x1024 + 2048 = 6144;
// request sync = 4 x ceil(4096/64) = 256 B.
func TestH2DModelShapeJSONFlag(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunH2DInventory(&out, &errb, []string{
		"--root", repoRoot(),
		`--model-shape={"hidden_size":256,"vocab":512,"layers":2,"batch":3}`, "--no-discover",
	})
	if code != 0 {
		t.Fatalf("RunH2DInventory exit = %d, stderr=%q\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{
		"TOTAL estimated H2D copies / decoded token: 5",
		"TOTAL estimated H2D bytes / decoded token: 6144",
		"request-scoped (dirty-gated) H2D: 1 copy(ies), <= 256 B/request",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

// TestH2DDriftDetection verifies the four drift reasons on a fixture tree.
func TestH2DDriftDetection(t *testing.T) {
	dir := t.TempDir()
	body := `package model

func sampleConstrained(history []int, logits []float32, c *DecodeConstraint) int {
	if c.maskActive() {
		c.Mask.MaskLogits(history, logits)
	}
	return argmaxF32(logits)
}
`
	p := filepath.Join(dir, "constraint.go")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	inv := []H2DCallSite{
		{ID: "ok-mask", File: "constraint.go", Func: "sampleConstrained", Pattern: "c.Mask.MaskLogits(history, logits)", Class: H2DPerToken},
		{ID: "gone-pattern", File: "constraint.go", Func: "sampleConstrained", Pattern: "c.Bias[tok]", Class: H2DPerToken},
		{ID: "gone-func", File: "constraint.go", Func: "oldSampler", Pattern: "whatever", Class: H2DPerToken},
		{ID: "gone-file", File: "nope.go", Func: "", Pattern: "x", Class: H2DPerToken},
	}
	drift, _ := verifyH2DInventory(dir, inv)
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
	if _, ok := reasons["ok-mask"]; ok {
		t.Error("ok-mask must not drift")
	}
}

// TestH2DDriftFlipsVerdictEndToEnd proves the regression-detection property:
// removing an inventoried H2D/mask seam from the live source flips the verb
// from PASS (exit 0) to DRIFT (exit 1). The fixture tree is a copy of the six
// real state-path files with the mask seam deleted from constraint.go.
func TestH2DDriftFlipsVerdictEndToEnd(t *testing.T) {
	root := repoRoot()
	dir := t.TempDir()
	constraintDst := filepath.Join(dir, filepath.FromSlash("internal/model/constraint.go"))
	for _, rel := range h2dDiscoverFiles {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		dst := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Baseline: the untouched copy passes.
	var baseOut bytes.Buffer
	if code := RunH2DInventory(&baseOut, &bytes.Buffer{}, []string{"--root", dir, "--no-discover"}); code != 0 {
		t.Fatalf("baseline exit = %d, want 0:\n%s", code, baseOut.String())
	}
	// Mutate: delete the inventoried mask seam from the live copy.
	data, err := os.ReadFile(constraintDst)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(constraintDst, []byte(strings.ReplaceAll(string(data), "c.Mask.MaskLogits(history, eff)", "// mask seam removed")), 0o644); err != nil {
		t.Fatal(err)
	}
	// The missing mask seam must flip the verdict to DRIFT.
	var out, errb bytes.Buffer
	code := RunH2DInventory(&out, &errb, []string{"--root", dir, "--no-discover"})
	if code != 1 {
		t.Fatalf("mutated exit = %d, want 1 (DRIFT):\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), "DRIFT") || !strings.Contains(out.String(), "mask-host-resident") {
		t.Fatalf("drift report missing the removed seam:\n%s", out.String())
	}
}

const fixtureH2DKernelCovered = `package compute

func growAppend(d *dslice, srcPtr unsafe.Pointer, nFloats int, site string) {
	C.fcuda_kv_write((*C.float)(d.ptr), (*C.float)(srcPtr), C.int(d.len), C.int(nFloats))
}

func rogueUpload(vals []float32) {
	C.fcuda_h2d(dst.ptr, unsafe.Pointer(&vals[0]), C.size_t(len(vals)*4))
}
`

func TestH2DDiscoveryUncovered(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "kernel.go")
	if err := os.WriteFile(p, []byte(fixtureH2DKernelCovered), 0o644); err != nil {
		t.Fatal(err)
	}
	// Cover only growAppend; rogueUpload's C.fcuda_h2d( must surface.
	inv := []H2DCallSite{{ID: "kv-append", File: "kernel.go", Func: "growAppend", Pattern: "C.fcuda_kv_write(", Class: H2DPerLayer}}
	prev := h2dDiscoverFiles
	h2dDiscoverFiles = []string{"kernel.go"}
	defer func() { h2dDiscoverFiles = prev }()

	uncovered := discoverH2DUncovered(dir, inv)
	if len(uncovered) != 1 {
		t.Fatalf("uncovered = %d sites (%+v), want exactly 1 (rogueUpload)", len(uncovered), uncovered)
	}
	u := uncovered[0]
	if u.File != "kernel.go" || u.Marker != "C.fcuda_h2d(" || !strings.Contains(u.Snippet, "vals[0]") {
		t.Fatalf("wrong uncovered site: %+v", u)
	}
	if u.Line != 8 {
		t.Errorf("uncovered line = %d, want 8 (inside rogueUpload)", u.Line)
	}
}

// TestH2DSnapshotWrite pins the committed-snapshot path: --snapshot writes a
// parseable v1 document with the same totals as the text report.
func TestH2DSnapshotWrite(t *testing.T) {
	dir := t.TempDir()
	snap := filepath.Join(dir, "nested", "h2d-inventory.json")
	var out, errb bytes.Buffer
	code := RunH2DInventory(&out, &errb, []string{"--root", repoRoot(), "--preset", "tiny-4", "--no-discover", "--snapshot", snap})
	if code != 0 {
		t.Fatalf("RunH2DInventory exit = %d, stderr=%q", code, errb.String())
	}
	data, err := os.ReadFile(snap)
	if err != nil {
		t.Fatalf("snapshot not written: %v", err)
	}
	var doc H2DInventoryDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("snapshot does not parse: %v", err)
	}
	if doc.Schema != "fak.dev.h2d-inventory.v1" || doc.Totals.CopiesPerToken != 3 || doc.Totals.BytesPerToken != 8192 {
		t.Fatalf("snapshot totals drifted: %+v", doc.Totals)
	}
	if len(doc.Sites) != len(h2dInventory) {
		t.Fatalf("snapshot sites = %d, want %d", len(doc.Sites), len(h2dInventory))
	}
}
