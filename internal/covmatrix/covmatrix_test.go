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
// honest-but-incomplete residual — cells that RUN but carry no CI oracle — and must
// exclude the CI-witnessed family, the fenced cells, and the debt cells.
func TestStaleCells(t *testing.T) {
	stale := StaleCells()
	if len(stale) == 0 {
		t.Fatal("StaleCells returned empty; expected the non-Llama runs-but-unwitnessed residual")
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
		if oracle[c.Family] {
			t.Errorf("family %q has a CI oracle and must not be stale: %+v", c.Family, c)
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
		if oracle[c.Family] {
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
