package covmatrix

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFamiliesMatchResolver(t *testing.T) {
	if len(Families) == 0 {
		t.Fatal("Families roster is empty")
	}
	for _, f := range Families {
		if f.Topology == "" {
			t.Errorf("Family %s has no topology", f.Name)
		}
	}
}

// TestResolverTokensExistInSource is the cross-check the epic names as the C1 acceptance
// gate: every Family that pins a ResolverToken must actually appear as a resolveSpecFor
// case in internal/model/tensor_resolver.go. If a peer renames or removes a resolver case
// without updating the roster, the matrix would silently describe a kernel that no longer
// exists — this test reds the trunk on exactly that drift. (Families with an empty token
// are the identity Llama default or families detected by a config predicate rather than a
// resolver substring; those are covered by the model-package conformance contract, #1081.)
func TestResolverTokensExistInSource(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot locate test source path")
	}
	// internal/covmatrix/covmatrix_test.go -> repo root is two dirs up from internal/.
	root := filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
	resolver := filepath.Join(root, "internal", "model", "tensor_resolver.go")
	b, err := os.ReadFile(resolver)
	if err != nil {
		t.Skipf("resolver source not readable (%v); cross-check skipped", err)
	}
	src := string(b)
	for _, f := range Families {
		if f.ResolverToken == "" {
			continue
		}
		needle := "\"" + f.ResolverToken + "\""
		if !strings.Contains(src, needle) {
			t.Errorf("family %q pins ResolverToken %q but no %s appears in tensor_resolver.go — roster has drifted from the kernel",
				f.Name, f.ResolverToken, needle)
		}
	}
}

// TestGridIsDeterministic is the other half of C1's acceptance: two builds at one commit
// must be byte-identical (no clock/map-order/randomness), so the committed snapshot
// regenerates with no diff and `--compare` is meaningful.
func TestGridIsDeterministic(t *testing.T) {
	a, err := json.Marshal(Build())
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	b, err := json.Marshal(Build())
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(a) != string(b) {
		t.Error("Build() is not deterministic across two calls")
	}
}

func TestBackends(t *testing.T) {
	if len(Backends) == 0 {
		t.Fatal("Backends roster is empty")
	}

	// CPU must be present
	foundCPU := false
	for _, b := range Backends {
		if b == "cpu" {
			foundCPU = true
			break
		}
	}
	if !foundCPU {
		t.Error("cpu backend missing from roster")
	}
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name    string
		family  Family
		backend string
		want    Support
	}{
		{
			name:    "Llama on CPU with CI oracle is SUPPORTED",
			family:  Family{Name: "Llama", Topology: PreNorm, OracleInCI: true},
			backend: "cpu",
			want:    Supported,
		},
		{
			name:    "Llama on CUDA (PreNorm) is SUPPORTED",
			family:  Family{Name: "Llama", Topology: PreNorm, OracleInCI: true},
			backend: "cuda",
			want:    Supported,
		},
		{
			name:    "Non-PreNorm on accelerated backend is FENCED",
			family:  Family{Name: "Gemma", Topology: PostNorm, OracleInCI: false},
			backend: "cuda",
			want:    Fenced,
		},
		{
			name:    "Non-PreNorm on CPU without CI oracle is PROOF-PATH-ONLY",
			family:  Family{Name: "Gemma", Topology: PostNorm, OracleInCI: false},
			backend: "cpu",
			want:    ProofPathOnly,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classify(tt.family, tt.backend)
			if got != tt.want {
				t.Errorf("classify(%v, %s) = %v, want %v", tt.family, tt.backend, got, tt.want)
			}
		})
	}
}

func TestGrid(t *testing.T) {
	cells := Grid()

	expectedCount := len(Families) * len(Backends)
	if len(cells) != expectedCount {
		t.Errorf("Grid() returned %d cells, want %d (%d families × %d backends)",
			len(cells), expectedCount, len(Families), len(Backends))
	}

	// Check for duplicates
	seen := make(map[string]bool)
	for _, c := range cells {
		key := c.Family + "/" + c.Backend
		if seen[key] {
			t.Errorf("Duplicate cell: %s", key)
		}
		seen[key] = true
	}
}

func TestBuildEmitsControlPanePayload(t *testing.T) {
	payload := Build()

	if payload.Schema != Schema {
		t.Errorf("Payload schema = %s, want %s", payload.Schema, Schema)
	}

	// The control-pane fold writes the debt under corpus[DebtKey] (corpus.growth_debt);
	// the Payload carries no top-level DebtKey field — that key is the consumer's contract.
	if _, ok := payload.Corpus[DebtKey]; !ok {
		t.Errorf("Payload corpus missing %q key", DebtKey)
	}

	// Check that growth_debt is present
	foundDebt := false
	for _, kpi := range payload.KPIs {
		if kpi.Key == "no_undefined_cells" {
			foundDebt = true
			break
		}
	}
	if !foundDebt {
		t.Error("Payload missing no_undefined_cells KPI")
	}

	// Check that payload is JSON-serializable
	_, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		t.Errorf("Payload not JSON-serializable: %v", err)
	}
}

// TestStaleCells is the C5 (#1084) gate: the --stale lens must surface exactly the
// honest-but-incomplete residual — cells that RUN but that no CI oracle executes — and
// must exclude the CI-witnessed cpu cells, the fenced cells, and the debt cells.
//
// The residual is keyed per CELL, not per family. Every oracle in the roster is a CPU
// f32 witness, so it clears its family's cpu cell only; an accelerated cell is a
// separate hot-path copy no cpu oracle runs. Asserting merely "non-empty" was not
// enough to hold that line: while some families still had OracleInCI == false the list
// stayed non-empty for the wrong reason, and completing the oracle roster emptied it
// entirely even though 18 accelerated cells remained unwitnessed. The accelerated
// assertion below is the one that actually binds.
func TestStaleCells(t *testing.T) {
	stale := StaleCells()
	if len(stale) == 0 {
		t.Fatal("StaleCells returned empty; expected the runs-but-unwitnessed residual")
	}

	// The contract the doctrine (#1244) and supportmaturity.FromSupport both state:
	// StaleCells flags "the accelerated-SUPPORTED cells whose witness is in fact
	// absent". No oracle in the roster executes an accelerated backend, so EVERY
	// accelerated SUPPORTED cell must appear — including those of a family that
	// carries a CPU oracle.
	inStale := make(map[Cell]bool, len(stale))
	for _, c := range stale {
		inStale[c.Cell] = true
	}
	acceleratedSupported := 0
	for _, c := range Grid() {
		if !accelerated(c.Backend) || c.Support != Supported {
			continue
		}
		acceleratedSupported++
		if !inStale[c] {
			t.Errorf("accelerated SUPPORTED cell %+v is unwitnessed in CI and must be stale", c)
		}
	}
	if acceleratedSupported == 0 {
		t.Error("no accelerated SUPPORTED cells in the grid; this gate would be vacuous")
	}

	// Determinism: two calls byte-identical (the package's whole-grid contract).
	a, err := json.Marshal(StaleCells())
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	b, err := json.Marshal(StaleCells())
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(a) != string(b) {
		t.Error("StaleCells is not deterministic across two calls")
	}

	oracle := make(map[string]bool, len(Families))
	for _, f := range Families {
		oracle[f.Name] = f.OracleInCI
	}
	for _, c := range stale {
		if oracle[c.Family] && !accelerated(c.Backend) {
			t.Errorf("family %q has a CPU oracle, so its cpu cell must not be stale: %+v", c.Family, c)
		}
		if c.Support == Fenced || c.Support == Undefined {
			t.Errorf("stale list must exclude %s cells: %+v", c.Support, c)
		}
		if c.Support != ProofPathOnly && c.Support != Supported {
			t.Errorf("a stale cell must be a running cell (SUPPORTED/PROOF-PATH-ONLY), got %s: %+v", c.Support, c)
		}
		if c.Reason == "" {
			t.Errorf("stale cell missing a reason: %+v", c)
		}
		// The reason must agree with the support level.
		if c.Support == ProofPathOnly && c.Reason != StaleProofPath {
			t.Errorf("PROOF-PATH-ONLY cell %+v should carry StaleProofPath, got %q", c, c.Reason)
		}
		if c.Support == Supported && c.Reason != StaleUnwitnessed {
			t.Errorf("SUPPORTED-no-oracle cell %+v should carry StaleUnwitnessed, got %q", c, c.Reason)
		}
	}

	// Cross-check the count against an independent recomputation over the grid.
	want := 0
	for _, c := range Grid() {
		if oracle[c.Family] && !accelerated(c.Backend) {
			continue
		}
		if c.Support == ProofPathOnly || c.Support == Supported {
			want++
		}
	}
	if len(stale) != want {
		t.Errorf("StaleCells returned %d cells, recomputation expected %d", len(stale), want)
	}
}

func TestCountBy(t *testing.T) {
	testCells := []Cell{
		{Support: Supported},
		{Support: Supported},
		{Support: Fenced},
		{Support: ProofPathOnly},
		{Support: Undefined},
		{Support: Undefined},
	}

	counts := countBy(testCells)

	if counts[Supported] != 2 {
		t.Errorf("counts[Supported] = %d, want 2", counts[Supported])
	}
	if counts[Fenced] != 1 {
		t.Errorf("counts[Fenced] = %d, want 1", counts[Fenced])
	}
	if counts[ProofPathOnly] != 1 {
		t.Errorf("counts[ProofPathOnly] = %d, want 1", counts[ProofPathOnly])
	}
	if counts[Undefined] != 2 {
		t.Errorf("counts[Undefined] = %d, want 2", counts[Undefined])
	}
}

// modelSource reads one file out of internal/model relative to this test's own path, or
// skips. It is the file-loading half of TestResolverTokensExistInSource, shared by the two
// cross-checks below so each one asserts rather than re-derives the repo layout.
func modelSource(t *testing.T, name string) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot locate test source path")
	}
	// internal/covmatrix/covmatrix_test.go -> repo root is two dirs up from internal/.
	root := filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
	b, err := os.ReadFile(filepath.Join(root, "internal", "model", name))
	if err != nil {
		t.Skipf("internal/model/%s not readable (%v); cross-check skipped", name, err)
	}
	return string(b)
}

// kernelTopologyCases parses internal/model/config.go's family -> BlockTopology default
// switch and returns each assigned topology mapped to the `case` condition that assigns it.
// The switch is a flat "case <family predicates>: c.BlockTopology = X" ladder, so remembering
// the last `case` line and reading the assignment under it is enough — no go/ast needed, and a
// restructure that breaks the shape is caught by the completeness check in the caller.
func kernelTopologyCases(src string) map[Topology]string {
	assigned := map[string]Topology{
		"PostNorm":         PostNorm,
		"SandwichNorm":     SandwichNorm,
		"ParallelResidual": ParallelResidual,
	}
	out := map[Topology]string{}
	pending := ""
	for _, line := range strings.Split(src, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "case ") {
			pending = trimmed
			continue
		}
		rhs, isAssign := strings.CutPrefix(trimmed, "c.BlockTopology = ")
		if !isAssign || pending == "" {
			continue
		}
		if topo, known := assigned[rhs]; known {
			out[topo] = pending
		}
		pending = ""
	}
	return out
}

// TestFamilyTopologyMatchesKernel binds the roster's TOPOLOGY axis to the kernel, the way
// TestResolverTokensExistInSource binds its FAMILY axis. covmatrix.go names this as the
// #1080 follow-on ("read the topology + fence facts straight from go/ast so even the
// per-family topology cannot drift"), and until now it was the *unwitnessed* axis — yet it
// is the one that decides FENCED vs SUPPORTED for every accelerated cell, so a silent drift
// here moves the whole climb-debt headline the milestone scorecard folds.
//
// The binding runs both ways. A family the roster calls PostNorm/SandwichNorm/
// ParallelResidual must be named by the config.go case that assigns exactly that topology; a
// family the roster calls PreNorm (or SparseAttn, which is a covmatrix-local fence axis, not
// a model.BlockTopology — the MLA/DSA/MSA families lower to the PreNorm zero value) must be
// named by NO case, because the kernel's switch is what would move it off PreNorm.
func TestFamilyTopologyMatchesKernel(t *testing.T) {
	cases := kernelTopologyCases(modelSource(t, "config.go"))
	for _, topo := range []Topology{PostNorm, SandwichNorm, ParallelResidual} {
		if cases[topo] == "" {
			t.Fatalf("no `case ...: c.BlockTopology = %s` found in internal/model/config.go; the topology switch moved or changed shape — re-derive this cross-check before trusting the roster", topo)
		}
	}

	checked := 0
	for _, f := range Families {
		if f.ResolverToken == "" {
			continue // identity default / config-predicate families carry no substring to match
		}
		checked++
		needle := "\"" + f.ResolverToken
		for topo, cond := range cases {
			named := strings.Contains(cond, needle)
			if topo == f.Topology && !named {
				t.Errorf("family %q pins Topology %s but config.go's %s case does not name %q — the roster has drifted from the kernel: %s",
					f.Name, f.Topology, topo, f.ResolverToken, cond)
			}
			if topo != f.Topology && named {
				t.Errorf("family %q pins Topology %s but config.go assigns %s for %q — the roster has drifted from the kernel: %s",
					f.Name, f.Topology, topo, f.ResolverToken, cond)
			}
		}
	}
	if checked == 0 {
		t.Error("no family carried a ResolverToken; this cross-check would be vacuous")
	}
}

// funcBody returns the source of the named method, from its signature line to the first
// column-0 closing brace, or "" when the signature is absent.
func funcBody(src, signature string) string {
	i := strings.Index(src, signature)
	if i < 0 {
		return ""
	}
	rest := src[i:]
	if end := strings.Index(rest, "\n}"); end >= 0 {
		return rest[:end]
	}
	return rest
}

// TestSparseAttnFenceCoverageMatchesKernel pins the fence-coverage exception classify()
// records: the two SparseAttn fences do NOT cover the same backends. requireMiniMaxSession
// refuses a compute.Backend, so MiniMax-MSA is fenced on every accelerated backend and its
// FENCED cells are honest. requireGLMDsaSession refuses only Metal and PrecisionPolicy and
// permits a Backend (#86 partial), so the MLA/DSA cells on the HAL backends are called FENCED
// behind a fence that does not reach them.
//
// The asymmetry is the whole point of the test: a comment saying "these four cells are
// mis-classified" rots the instant someone closes the gap, and a stale disclaimer is just
// another wrong claim. Pinning the gap makes closing it RED here, which forces the classify()
// comment and the FENCED/SUPPORTED question to be revisited in the same change.
func TestSparseAttnFenceCoverageMatchesKernel(t *testing.T) {
	glm := funcBody(modelSource(t, "kv.go"), "func (s *Session) requireGLMDsaSession() {")
	if glm == "" {
		t.Fatal("requireGLMDsaSession not found in internal/model/kv.go; the SparseAttn fence named by covmatrix.go no longer exists")
	}
	if !strings.Contains(glm, "s.Metal") {
		t.Error("requireGLMDsaSession no longer guards on s.Metal; the MLA/DSA metal cells may no longer be FENCED — re-check classify()")
	}
	if strings.Contains(glm, "s.Backend") {
		t.Error("requireGLMDsaSession now names s.Backend: the MLA/DSA HAL exception recorded in classify() has changed. Re-read the fence and update classify()'s KNOWN EXCEPTION note (and revisit whether DeepSeek-MLA / GLM-5.2-DSA x cuda,vulkan should still classify as FENCED)")
	}

	msa := funcBody(modelSource(t, "minimax_m3_session.go"), "func (s *Session) requireMiniMaxSession() {")
	if msa == "" {
		t.Fatal("requireMiniMaxSession not found in internal/model/minimax_m3_session.go; the MSA fence named by covmatrix.go no longer exists")
	}
	if !strings.Contains(msa, "s.Backend") || !strings.Contains(msa, "s.Metal") {
		t.Error("requireMiniMaxSession no longer refuses both s.Backend and s.Metal; MiniMax-MSA's accelerated cells may no longer be FENCED — re-check classify()")
	}
}
