package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"
)

// layoutFixtureSHA is a closed 40-hex digit pin SHA for fixture contracts;
// the pin-content checks never consult git, so the value only has to satisfy
// the registry's shaPattern.
const layoutFixtureSHA = "aaaaaaaaaabbbbbbbbbbccccccccccdddddddddd"

// layoutNvfp4FixtureBody places the forward function name on line 4 so a
// transform pin at line 4 passes the pin-content check.
const layoutNvfp4FixtureBody = "package compute\n\n// SwizzleNVFP4Scales is the fixture pin target.\nfunc SwizzleNVFP4Scales(rows, scaleCols int, linear []byte) ([]byte, error) { return linear, nil }\n\nfunc UnswizzleNVFP4Scales(rows, scaleCols int, tiled []byte) ([]byte, error) { return tiled, nil }\n"

func layoutWriteFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// layoutMatrixJSON renders a minimal valid packing matrix (schema
// fak.packingmatrix.v1) over the given family tags.
func layoutMatrixJSON(families ...string) string {
	cells := make([]string, 0, len(families))
	for _, f := range families {
		cells = append(cells, fmt.Sprintf(`{"family":%q,"device":"cpu","mode":"native","evidence":["fixture"]}`, f))
	}
	return fmt.Sprintf(`{"schema":"fak.packingmatrix.v1","generated_at":"2026-11-05","audit_head":"%s","ticket":"lg-layoutcontract-test","cells":[%s],"excluded_pairs":[]}`,
		layoutFixtureSHA, strings.Join(cells, ","))
}

// layoutFixtureContract builds a closed contract whose pins resolve into the
// fixture tree. With withTransform it registers the real NVFP4 swizzle pair
// pinned to line 4 of internal/compute/nvfp4_scale_layout.go.
func layoutFixtureContract(family, srcPath string, srcLine int, withTransform bool) layoutcontract.Contract {
	c := layoutcontract.Contract{
		Family:          family,
		WeightOrder:     layoutcontract.RowMajor,
		InterleaveWidth: 1,
		ScaleLayout:     layoutcontract.ScaleLinear,
		PaddingRule:     "fixture rule",
		PackingPath:     layoutcontract.PackingNative,
		Source:          layoutcontract.SourceRef{Path: srcPath, Line: srcLine, SHA: layoutFixtureSHA},
	}
	if withTransform {
		c.PackingPath = layoutcontract.PackingRuntime
		c.Transform = layoutcontract.Transform{
			Name:   "compute.SwizzleNVFP4Scales / compute.UnswizzleNVFP4Scales",
			Source: layoutcontract.SourceRef{Path: "internal/compute/nvfp4_scale_layout.go", Line: 4, SHA: layoutFixtureSHA},
			Forward: func(s layoutcontract.Shape, b []byte) ([]byte, error) {
				return compute.SwizzleNVFP4Scales(s.Rows, s.Cols, b)
			},
			Inverse: func(s layoutcontract.Shape, b []byte) ([]byte, error) {
				return compute.UnswizzleNVFP4Scales(s.Rows, s.Cols, b)
			},
		}
	}
	return c
}

func assertLayoutViolation(t *testing.T, rep *LayoutReport, check, family string) {
	t.Helper()
	for _, v := range rep.Violations {
		if v.Check == check && v.Family == family {
			return
		}
	}
	t.Fatalf("report lacks a %s violation for %s; violations=%v", check, family, rep.Violations)
}

// TestLayoutContractCheckHappyPath drives the verb end-to-end against the live
// registry and the landed packing matrix: no violations, the closed outcome
// is gaps (the matrix covers families the registry does not register yet),
// and every delta is reported explicitly rather than folded into "complete".
//
// fak-test:runtime fast est=250ms lane=default
func TestLayoutContractCheckHappyPath(t *testing.T) {
	var out, errOut bytes.Buffer
	code := RunLayoutContract(&out, &errOut, []string{"check", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, errOut.String(), out.String())
	}
	var rep LayoutReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode report: %v\n%s", err, out.String())
	}
	if rep.Schema != layoutCheckSchema {
		t.Fatalf("schema=%q, want %q", rep.Schema, layoutCheckSchema)
	}
	if rep.Outcome != layoutOutcomeGaps {
		t.Fatalf("outcome=%q, want %q (matrix covers more families than the registry)", rep.Outcome, layoutOutcomeGaps)
	}
	if want := []string{"MXFP4", "NVFP4", "Q4_K", "Q8_0"}; !reflect.DeepEqual(rep.Families, want) {
		t.Fatalf("families=%v, want %v", rep.Families, want)
	}
	for _, c := range rep.Checks {
		if c.Status == layoutStatusViolated {
			t.Fatalf("check %s violated on the live registry: %s", c.Name, c.Detail)
		}
	}
	matrixOnly := map[string]bool{}
	for _, g := range rep.Gaps {
		if g.Kind == layoutGapRegistryOnly {
			t.Fatalf("unexpected registry-only gap %q on the live registry", g.Family)
		}
		if g.Kind == layoutGapMatrixOnly {
			matrixOnly[g.Family] = true
		}
	}
	wantGaps := []string{"awq_int4_group", "f16_hgemm", "fp8_block32_e8m0", "gptq_int4_group", "iq_raw", "q2_k", "q3_k", "q5_k", "q6_k"}
	if len(matrixOnly) != len(wantGaps) {
		t.Fatalf("matrix-only gaps=%v, want %v", sortedKeys(matrixOnly), wantGaps)
	}
	for _, f := range wantGaps {
		if !matrixOnly[f] {
			t.Fatalf("missing explicit matrix-only gap %q; gaps=%v", f, rep.Gaps)
		}
	}

	var hOut, hErr bytes.Buffer
	if code := RunLayoutContract(&hOut, &hErr, []string{"check"}); code != 0 {
		t.Fatalf("human mode code=%d stderr=%s", code, hErr.String())
	}
	if !strings.Contains(hOut.String(), "layout-contract check: GAPS") {
		t.Fatalf("human output lacks the closed outcome token:\n%s", hOut.String())
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// TestLayoutContractCheckJSONShapeClosed pins the machine-output schema: the
// exact top-level key set on a pass verdict, the closed outcome/status
// vocabularies, and the closed ordered check-name list.
//
// fak-test:runtime fast est=100ms lane=default
func TestLayoutContractCheckJSONShapeClosed(t *testing.T) {
	root := layoutWriteFixture(t, map[string]string{
		"internal/compute/cpuref.go":      strings.Repeat("weight row\n", 20),
		"docs/data/packing-coverage.json": layoutMatrixJSON("q8_0"),
	})
	rep, err := CheckLayoutContracts(LayoutOptions{
		Root:      root,
		Contracts: []layoutcontract.Contract{layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false)},
	})
	if err != nil {
		t.Fatalf("clean fixture refused: %v", err)
	}
	if rep.Outcome != layoutOutcomePass {
		t.Fatalf("outcome=%q, want %q", rep.Outcome, layoutOutcomePass)
	}
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(b, &keys); err != nil {
		t.Fatal(err)
	}
	wantKeys := []string{"checks", "families", "outcome", "root", "schema", "scope"}
	if got := sortedKeys(mapFromKeys(keys)); !reflect.DeepEqual(got, wantKeys) {
		t.Fatalf("top-level keys=%v, want %v (closed schema)", got, wantKeys)
	}
	var rep2 LayoutReport
	if err := json.Unmarshal(b, &rep2); err != nil {
		t.Fatal(err)
	}
	switch rep2.Outcome {
	case layoutOutcomePass, layoutOutcomeGaps, layoutOutcomeFail:
	default:
		t.Fatalf("outcome %q escapes the closed vocabulary", rep2.Outcome)
	}
	names := make([]string, 0, len(rep2.Checks))
	for _, c := range rep2.Checks {
		names = append(names, c.Name)
		switch c.Status {
		case layoutStatusOK, layoutStatusGaps, layoutStatusViolated:
		default:
			t.Fatalf("status %q escapes the closed vocabulary", c.Status)
		}
	}
	if !reflect.DeepEqual(names, layoutCheckNames) {
		t.Fatalf("check names=%v, want the closed ordered set %v", names, layoutCheckNames)
	}
	if len(rep2.Violations) != 0 || len(rep2.Gaps) != 0 {
		t.Fatalf("pass verdict carries rows: gaps=%v violations=%v", rep2.Gaps, rep2.Violations)
	}
}

func mapFromKeys(m map[string]json.RawMessage) map[string]bool {
	out := make(map[string]bool, len(m))
	for k := range m {
		out[k] = true
	}
	return out
}

// TestLayoutContractCheckStalePinPlantedDefect is the recurrence witness for
// LML-04/LML-22: a layout fact that moved leaves its pin stale, and the guard
// reports it instead of letting the drift read as "contract intact". Each
// sub-case plants a distinct stale-pin shape into an injected contract list.
//
// fak-test:runtime fast est=100ms lane=default
func TestLayoutContractCheckStalePinPlantedDefect(t *testing.T) {
	t.Run("line_past_eof", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"internal/model/quant_q4k.go":     strings.Repeat("weight row\n", 10),
			"docs/data/packing-coverage.json": layoutMatrixJSON("q8_0"),
		})
		rep, err := CheckLayoutContracts(LayoutOptions{
			Root:      root,
			Contracts: []layoutcontract.Contract{layoutFixtureContract("Q8_0", "internal/model/quant_q4k.go", 99, false)},
		})
		if !errors.Is(err, ErrLayoutStalePin) {
			t.Fatalf("err=%v, want ErrLayoutStalePin", err)
		}
		assertLayoutViolation(t, rep, layoutCheckPins, "Q8_0")
		if rep.Outcome != layoutOutcomeFail {
			t.Fatalf("outcome=%q, want fail", rep.Outcome)
		}
	})

	t.Run("file_missing", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"docs/data/packing-coverage.json": layoutMatrixJSON("q8_0"),
		})
		_, err := CheckLayoutContracts(LayoutOptions{
			Root:      root,
			Contracts: []layoutcontract.Contract{layoutFixtureContract("Q8_0", "internal/model/quant_q4k.go", 3, false)},
		})
		if !errors.Is(err, ErrLayoutStalePin) {
			t.Fatalf("err=%v, want ErrLayoutStalePin", err)
		}
	})

	t.Run("transform_pin_moved", func(t *testing.T) {
		// Line 4 exists but no longer cites the forward function: the cited
		// definition moved and the pin must go stale.
		body := "package compute\n\n// RepackQ4KInterleaved moved to another file.\nvar moved = true\n\nfunc other() {}\n"
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/q4krepack.go":          body,
			"internal/compute/nvfp4_scale_layout.go": layoutNvfp4FixtureBody,
			"docs/data/packing-coverage.json":        layoutMatrixJSON("nvfp4"),
		})
		c := layoutFixtureContract("NVFP4", "internal/compute/nvfp4_scale_layout.go", 4, true)
		c.Transform.Source = layoutcontract.SourceRef{Path: "internal/compute/q4krepack.go", Line: 4, SHA: layoutFixtureSHA}
		rep, err := CheckLayoutContracts(LayoutOptions{Root: root, Contracts: []layoutcontract.Contract{c}})
		if !errors.Is(err, ErrLayoutStalePin) {
			t.Fatalf("err=%v, want ErrLayoutStalePin", err)
		}
		assertLayoutViolation(t, rep, layoutCheckPins, "NVFP4")
	})
}

// TestLayoutContractCheckHalfRegisteredPairPlantedDefect plants both
// orientations of a half-registered transform pair; the closed invariant
// refuses each with the integrity sentinel.
//
// fak-test:runtime fast est=50ms lane=default
func TestLayoutContractCheckHalfRegisteredPairPlantedDefect(t *testing.T) {
	root := layoutWriteFixture(t, map[string]string{
		"internal/compute/cpuref.go":      strings.Repeat("weight row\n", 20),
		"docs/data/packing-coverage.json": layoutMatrixJSON("q8_0"),
	})
	id := func(s layoutcontract.Shape, b []byte) ([]byte, error) { return b, nil }
	forwardOnly := layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false)
	forwardOnly.Transform.Forward = id
	backwardOnly := layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false)
	backwardOnly.Transform.Inverse = id

	rep, err := CheckLayoutContracts(LayoutOptions{
		Root:      root,
		Contracts: []layoutcontract.Contract{forwardOnly, backwardOnly},
	})
	if !errors.Is(err, ErrLayoutIntegrity) {
		t.Fatalf("err=%v, want ErrLayoutIntegrity", err)
	}
	halfPairs := 0
	for _, v := range rep.Violations {
		if v.Check == layoutCheckIntegrity {
			halfPairs++
		}
	}
	if halfPairs != 2 {
		t.Fatalf("half-pair violations=%d, want 2; violations=%v", halfPairs, rep.Violations)
	}
}

// TestLayoutContractCheckUnknownFamilyPlantedDefect pins the fail-closed
// refusal for a family the registry does not register: the sentinel is
// errors.Is-assertable at the API level and the CLI exits 2 with the closed
// refusal token.
//
// fak-test:runtime fast est=50ms lane=default
func TestLayoutContractCheckUnknownFamilyPlantedDefect(t *testing.T) {
	_, err := CheckLayoutContracts(LayoutOptions{Family: "BOGUS_FAM"})
	if !errors.Is(err, layoutcontract.ErrUnknownFamily) {
		t.Fatalf("err=%v, want layoutcontract.ErrUnknownFamily", err)
	}

	var out, errOut bytes.Buffer
	code := RunLayoutContract(&out, &errOut, []string{"check", "--family", "BOGUS_FAM"})
	if code != 2 {
		t.Fatalf("code=%d, want 2; stderr=%s", code, errOut.String())
	}
	if out.Len() != 0 {
		t.Fatalf("refusal wrote stdout: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "LAYOUT_CONTRACT_REFUSED") {
		t.Fatalf("stderr lacks the closed refusal token: %s", errOut.String())
	}
}

// TestLayoutContractCheckRoundTripDefectPlantedDefect plants a transform pair
// whose inverse is the identity: forward o inverse cannot restore the payload,
// and the guard must catch the offset-class bug the ledger rows paid for in
// hardware re-runs. A second sub-case registers a pair with no witness shape
// and requires the vacuous-witness refusal instead of a silent skip.
//
// fak-test:runtime fast est=100ms lane=default
func TestLayoutContractCheckRoundTripDefectPlantedDefect(t *testing.T) {
	t.Run("inverse_not_inverting", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/nvfp4_scale_layout.go": layoutNvfp4FixtureBody,
			"docs/data/packing-coverage.json":        layoutMatrixJSON("nvfp4"),
		})
		c := layoutFixtureContract("NVFP4", "internal/compute/nvfp4_scale_layout.go", 4, true)
		c.Transform.Inverse = func(s layoutcontract.Shape, b []byte) ([]byte, error) { return b, nil }
		rep, err := CheckLayoutContracts(LayoutOptions{Root: root, Contracts: []layoutcontract.Contract{c}})
		if !errors.Is(err, ErrLayoutRoundTrip) {
			t.Fatalf("err=%v, want ErrLayoutRoundTrip", err)
		}
		for _, v := range rep.Violations {
			if v.Check != layoutCheckRoundTrip {
				t.Fatalf("unexpected violation outside the round-trip check: %+v", v)
			}
		}
		if rep.Outcome != layoutOutcomeFail {
			t.Fatalf("outcome=%q, want fail", rep.Outcome)
		}
	})

	t.Run("pair_without_witness_shape", func(t *testing.T) {
		reverse := func(s layoutcontract.Shape, b []byte) ([]byte, error) {
			out := make([]byte, len(b))
			for i := range b {
				out[i] = b[len(b)-1-i]
			}
			return out, nil
		}
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/nvfp4_scale_layout.go": layoutNvfp4FixtureBody,
			"docs/data/packing-coverage.json":        layoutMatrixJSON("nvfp4"),
		})
		c := layoutFixtureContract("ZZZ_FAKE", "internal/compute/nvfp4_scale_layout.go", 4, true)
		c.Transform.Forward = reverse
		c.Transform.Inverse = reverse
		rep, err := CheckLayoutContracts(LayoutOptions{Root: root, Contracts: []layoutcontract.Contract{c}})
		if !errors.Is(err, ErrLayoutNoWitness) {
			t.Fatalf("err=%v, want ErrLayoutNoWitness", err)
		}
		assertLayoutViolation(t, rep, layoutCheckRoundTrip, "ZZZ_FAKE")
	})
}

// TestLayoutContractCheckCoverageGapsExplicit pins the coverage semantics:
// deltas are reported as gaps (exit 0, never silently complete), and a matrix
// that is missing or off-schema refuses instead of reading as "no gaps".
//
// fak-test:runtime fast est=50ms lane=default
func TestLayoutContractCheckCoverageGapsExplicit(t *testing.T) {
	t.Run("registry_only_gap_reports_not_fails", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/cpuref.go":      strings.Repeat("weight row\n", 20),
			"internal/model/quant_q4k.go":     strings.Repeat("weight row\n", 20),
			"docs/data/packing-coverage.json": layoutMatrixJSON("q8_0"),
		})
		rep, err := CheckLayoutContracts(LayoutOptions{
			Root: root,
			Contracts: []layoutcontract.Contract{
				layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false),
				layoutFixtureContract("Q4_K", "internal/model/quant_q4k.go", 5, false),
			},
		})
		if err != nil {
			t.Fatalf("gaps must not refuse: %v", err)
		}
		if rep.Outcome != layoutOutcomeGaps {
			t.Fatalf("outcome=%q, want gaps", rep.Outcome)
		}
		found := false
		for _, g := range rep.Gaps {
			if g.Kind == layoutGapRegistryOnly && g.Family == "q4_k" {
				found = true
			}
		}
		if !found {
			t.Fatalf("registry-only gap for q4_k not reported: %v", rep.Gaps)
		}
		var out, errOut bytes.Buffer
		if code := RunLayoutContract(&out, &errOut, []string{"check", "--json", "--root", root}); code != 0 {
			t.Fatalf("gaps must exit 0 through the CLI: code=%d stderr=%s", code, errOut.String())
		}
	})

	t.Run("matrix_missing_refuses", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/cpuref.go": strings.Repeat("weight row\n", 20),
		})
		_, err := CheckLayoutContracts(LayoutOptions{
			Root:      root,
			Contracts: []layoutcontract.Contract{layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false)},
		})
		if !errors.Is(err, ErrLayoutMatrixMissing) {
			t.Fatalf("err=%v, want ErrLayoutMatrixMissing", err)
		}
		var out, errOut bytes.Buffer
		if code := RunLayoutContract(&out, &errOut, []string{"check", "--root", root}); code != 2 {
			t.Fatalf("code=%d, want 2; stderr=%s", code, errOut.String())
		}
	})

	t.Run("matrix_wrong_schema_refuses", func(t *testing.T) {
		root := layoutWriteFixture(t, map[string]string{
			"internal/compute/cpuref.go": strings.Repeat("weight row\n", 20),
			"docs/data/packing-coverage.json": strings.Replace(layoutMatrixJSON("q8_0"),
				`"schema":"fak.packingmatrix.v1"`, `"schema":"fak.packingmatrix.v0"`, 1),
		})
		_, err := CheckLayoutContracts(LayoutOptions{
			Root:      root,
			Contracts: []layoutcontract.Contract{layoutFixtureContract("Q8_0", "internal/compute/cpuref.go", 5, false)},
		})
		if !errors.Is(err, ErrLayoutMatrixSchema) {
			t.Fatalf("err=%v, want ErrLayoutMatrixSchema", err)
		}
	})
}
