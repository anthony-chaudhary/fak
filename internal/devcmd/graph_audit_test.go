package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// graphFixtureFiles is a minimal tree carrying every pinned pattern of
// graphAuditInventory exactly once, so the audit can be exercised against a
// deterministic root instead of the live repository. The seams are written as
// comments because the audit is string-based over go/ast function spans (the
// glue-census approach): only presence/absence in the right scope matters.
var graphFixtureFiles = map[string]string{
	"internal/model/hal.go": `package model

func (s *Session) tokenHALOutput(id, pos int) int {
	// once.GraphOnceReplay(x, s.halKV, pos) steady-state capture-once replay seam
	// gr.GraphBegin() classic per-token capture fallback seam
	return id + pos
}
`,
	"internal/model/glm_dsa_session.go": `package model

func (s *Session) decodeBandGLMDsa(id int, x []float32, lo, hi, pos int, isFirst, isLast bool) ([]float32, error) {
	mat := s.glmDsaMatKernel()
	_ = mat
	return x, nil
}
`,
	"internal/model/session_token_hidden.go": `package model

func (s *Session) tokenHiddenRow(lineageID, pos int, x []float32) []float32 {
	// s.blockStep(l, pos, x, cos, sin, f32Kernel{m}) per-layer host setup
	return x
}
`,
	"internal/compute/cuda.go": `package compute

// FAK_CUDA_GRAPH consent flag gates the reusable CUDA-graph decode path.

func (c *cudaBackend) GraphBegin() bool {
	// C.fcuda_graph_begin(
	return false
}

func (c *cudaBackend) GraphEndLaunch() {
	// C.fcuda_graph_end_launch(
}
`,
	"internal/compute/cuda_graph_once_qualification.go": `package compute

func (c *cudaBackend) GraphOnceBegin(input Tensor, kv KVStore, pos int) bool {
	// C.fak_graph_once_create(
	return false
}

func (c *cudaBackend) GraphOnceReplay(input Tensor, kv KVStore, pos int) (Tensor, bool) {
	// C.fak_graph_once_replay(
	return nil, false
}
`,
	"internal/compute/vulkan.go": `package compute

func (v *vulkanBackend) MatMul(w, x Tensor) Tensor { return nil }
`,
	"internal/compute/metal.go": `package compute

func (c *metalBackend) MatMul(w, x Tensor) Tensor { return nil }
`,
	"internal/compute/cpuref.go": `package compute

func (c *cpuBackend) MatMul(w, x Tensor) Tensor { return nil }
`,
}

// writeGraphFixture materializes the fixture tree under a fresh temp root,
// applying mutate to the file map first (nil = unmodified).
func writeGraphFixture(t *testing.T, mutate func(files map[string]string)) string {
	t.Helper()
	files := map[string]string{}
	for k, v := range graphFixtureFiles {
		files[k] = v
	}
	if mutate != nil {
		mutate(files)
	}
	root := t.TempDir()
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", full, err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", full, err)
		}
	}
	return root
}

func runGraphAudit(t *testing.T, argv ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := RunGraphAudit(&out, &errb, argv)
	return code, out.String(), errb.String()
}

// TestGraphAuditFixtureAllFourClasses proves the fixture covers all four
// classifications and that the coverage rollup counts only path-role rows.
func TestGraphAuditFixtureAllFourClasses(t *testing.T) {
	root := writeGraphFixture(t, nil)
	code, out, errOut := runGraphAudit(t, "--root", root)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (no drift, no --fail-on); stderr=%s", code, errOut)
	}
	want := map[string]GraphClass{
		"hal-capture-once-replay": GraphCaptured,
		"hal-classic-graph":       GraphCaptured,
		"glm-dsa-band-step":       GraphPerLayerHostSetup,
		"legacy-f32-blockstep":    GraphPerLayerHostSetup,
		"vulkan-no-graph":         GraphNoGraphSupport,
		"metal-no-graph":          GraphNoGraphSupport,
		"cpuref-no-graph":         GraphNoGraphSupport,
	}
	var outB bytes.Buffer
	if c := RunGraphAudit(&outB, ioDiscard{}, []string{"--root", root, "--json"}); c != 0 {
		t.Fatalf("--json exit = %d, want 0", c)
	}
	var res GraphAuditResult
	if err := json.Unmarshal(outB.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal --json output: %v\n%s", err, outB.String())
	}
	got := map[string]GraphClass{}
	for _, s := range res.Sites {
		got[s.ID] = s.Verdict
	}
	if len(res.Drift) != 0 {
		t.Fatalf("fixture drift = %d, want 0: %+v", len(res.Drift), res.Drift)
	}
	c := res.Coverage
	if c.Captured != 2 || c.PerLayerHostSetup != 2 || c.NoGraphSupport != 3 || c.Unknown != 0 {
		t.Fatalf("coverage = %+v, want captured=2 plsh=2 ngs=3 unknown=0", c)
	}
	if c.TotalPaths != 7 || c.UncoveredPaths != 5 {
		t.Fatalf("coverage total/uncovered = %d/%d, want 7/5", c.TotalPaths, c.UncoveredPaths)
	}
	for id, class := range want {
		if got[id] != class {
			t.Errorf("verdict[%s] = %s, want %s", id, got[id], class)
		}
	}
	for _, id := range []string{"cuda-graph-flag", "cuda-graph-begin", "cuda-graph-end-launch", "cuda-graph-once-create", "cuda-graph-once-replay"} {
		if got[id] != GraphCaptured {
			t.Errorf("primitive verdict[%s] = %s, want CAPTURED", id, got[id])
		}
	}
	if !strings.Contains(out, "UNCOVERED decode paths") || !strings.Contains(out, ": 5") {
		t.Errorf("text report missing uncovered count:\n%s", out)
	}
}

// ioDiscard is a tiny io.Writer sink for the stderr stream in tests.
type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }

// TestGraphAuditDriftRemovesCaptureSite proves the invokability gate: deleting
// the capture-once replay call site from the HAL step flips its verdict to
// UNKNOWN, records DRIFT, and fails the run.
func TestGraphAuditDriftRemovesCaptureSite(t *testing.T) {
	root := writeGraphFixture(t, func(files map[string]string) {
		files["internal/model/hal.go"] = strings.Replace(
			files["internal/model/hal.go"],
			"\t// once.GraphOnceReplay(x, s.halKV, pos) steady-state capture-once replay seam\n", "", 1)
	})
	code, _, errOut := runGraphAudit(t, "--root", root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (drift after removing the capture site); stderr=%s", code, errOut)
	}
	var outB bytes.Buffer
	if c := RunGraphAudit(&outB, ioDiscard{}, []string{"--root", root, "--json"}); c != 1 {
		t.Fatalf("--json exit = %d, want 1", c)
	}
	var res GraphAuditResult
	if err := json.Unmarshal(outB.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(res.Drift) != 1 || res.Drift[0].Site.ID != "hal-capture-once-replay" || res.Drift[0].Reason != "pattern-absent" {
		t.Fatalf("drift = %+v, want hal-capture-once-replay/pattern-absent", res.Drift)
	}
	for _, s := range res.Sites {
		if s.ID == "hal-capture-once-replay" && s.Verdict != GraphUnknown {
			t.Fatalf("verdict[hall-capture-once-replay] = %s, want UNKNOWN after the capture site vanished", s.Verdict)
		}
	}
	if res.Coverage.Captured != 1 || res.Coverage.Unknown != 1 {
		t.Fatalf("coverage after drift = %+v, want captured=1 unknown=1", res.Coverage)
	}
}

// TestGraphAuditNegativePatternFlip proves the other direction: a capture
// marker appearing where the inventory says it must not (a new graph path in
// the GLM-DSA band) is reported as DRIFT and fails the run, forcing the table
// to be reclassified instead of silently going stale.
func TestGraphAuditNegativePatternFlip(t *testing.T) {
	root := writeGraphFixture(t, func(files map[string]string) {
		files["internal/model/glm_dsa_session.go"] = strings.Replace(
			files["internal/model/glm_dsa_session.go"],
			"\treturn x, nil\n", "\t// gr.GraphBegin() someone wired capture into the band\n\treturn x, nil\n", 1)
	})
	code, _, _ := runGraphAudit(t, "--root", root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (negative pattern present)", code)
	}
	var outB bytes.Buffer
	if c := RunGraphAudit(&outB, ioDiscard{}, []string{"--root", root, "--json"}); c != 1 {
		t.Fatalf("--json exit = %d, want 1", c)
	}
	var res GraphAuditResult
	if err := json.Unmarshal(outB.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(res.Drift) != 1 || res.Drift[0].Site.ID != "glm-dsa-band-step" || res.Drift[0].Reason != "negative-pattern-present" {
		t.Fatalf("drift = %+v, want glm-dsa-band-step/negative-pattern-present", res.Drift)
	}
}

// TestGraphAuditFailOnUncovered proves the gate: the fixture has five
// uncovered paths, so --fail-on uncovered exits non-zero while the plain run
// passes.
func TestGraphAuditFailOnUncovered(t *testing.T) {
	root := writeGraphFixture(t, nil)
	if code, _, errOut := runGraphAudit(t, "--root", root, "--fail-on", "uncovered"); code != 1 {
		t.Fatalf("--fail-on uncovered exit = %d, want 1; stderr=%s", code, errOut)
	}
	if code, _, errOut := runGraphAudit(t, "--root", root, "--fail-on", "bogus"); code != 2 {
		t.Fatalf("--fail-on bogus exit = %d, want 2; stderr=%s", code, errOut)
	}
}

// TestGraphAuditSnapshotWritesMachineReadableTable proves --snapshot lands the
// trendable JSON (with an empty head_sha outside a git tree) and that the file
// round-trips through the same result type.
func TestGraphAuditSnapshotWritesMachineReadableTable(t *testing.T) {
	root := writeGraphFixture(t, nil)
	snap := filepath.Join(t.TempDir(), "nested", "graph-audit.json")
	code, out, errOut := runGraphAudit(t, "--root", root, "--snapshot", snap)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errOut)
	}
	if !strings.Contains(out, "snapshot written:") {
		t.Fatalf("text report missing snapshot notice:\n%s", out)
	}
	body, err := os.ReadFile(snap)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	if !bytes.HasSuffix(body, []byte("\n")) {
		t.Fatal("snapshot must end with a newline")
	}
	var res GraphAuditResult
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	if res.HeadSHA != gitHeadSHA(root) {
		t.Fatalf("head_sha = %q, want %q (best-effort resolution of the fixture root's git tree)", res.HeadSHA, gitHeadSHA(root))
	}
	if res.Coverage.UncoveredPaths != 5 {
		t.Fatalf("snapshot coverage = %+v, want 5 uncovered", res.Coverage)
	}
	if len(res.Sites) != len(graphAuditInventory) {
		t.Fatalf("snapshot sites = %d, want inventory rows %d", len(res.Sites), len(graphAuditInventory))
	}
	for i, want := range graphAuditInventory {
		got := res.Sites[i]
		if !reflect.DeepEqual(got.GraphSite, want) {
			t.Errorf("snapshot site %d = %+v, want inventory row %+v", i, got.GraphSite, want)
		}
		if got.Verdict != want.Class || got.Occurrences != 1 {
			t.Errorf("snapshot site %q verdict/occurrences = %s/%d, want %s/1", want.ID, got.Verdict, got.Occurrences, want.Class)
		}
	}
}

// TestGraphAuditLiveTreeNoDrift anchors the committed inventory to trunk: zero
// drift on the live repository means every pinned capture/replay site,
// primitive, and backend marker still exists at this commit. This is the row
// the snapshot trends against.
func TestGraphAuditLiveTreeNoDrift(t *testing.T) {
	root := repoRoot()
	drift, occ := verifyGraphInventory(root, graphAuditInventory)
	if len(drift) > 0 {
		for _, d := range drift {
			t.Errorf("drift %s/%s [%s]: %s", d.Site.File, d.Site.Func, d.Site.Pattern, d.Reason)
		}
		return
	}
	for _, s := range graphAuditInventory {
		if occ[s.ID] < 1 {
			t.Errorf("row %s has no live occurrence", s.ID)
		}
	}
}
