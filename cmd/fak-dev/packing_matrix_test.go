package main

// packing-matrix reader tests. The fixtures mirror the closed contract of
// docs/data/packing-coverage.json (schema fak.packingmatrix.v1) in a temp
// tree, so no live repository state is touched.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func packingMatrixFixture(t *testing.T, mutate func(*packingMatrixDoc)) string {
	t.Helper()
	doc := &packingMatrixDoc{
		Schema:      packingMatrixSourceSchema,
		GeneratedAt: "2026-11-03",
		AuditHead:   "9bd0f1516ae06f8e2848f07b8b9a63c78c920efd",
		Cells: []packingMatrixCell{
			{
				Family:   "q4_k",
				Device:   "cpu",
				Mode:     packingModeNative,
				Evidence: []string{"internal/model/quant_q4k.go:42 - resident Q4_K holds GGUF super-block bytes verbatim"},
			},
			{
				Family:   "nvfp4",
				Device:   "cuda",
				Mode:     packingModeRuntimeRepack,
				Evidence: []string{"DEFECT CELL - internal/compute/cuda/weight_formats.cuh:370-373 - per-call scale swizzle"},
			},
			{
				Family:   "mxfp4_e8m0",
				Device:   "vulkan",
				Mode:     packingModeUnknown,
				Evidence: []string{"missing probe: ROCm backend Upload dispatch"},
			},
		},
		Excluded: []packingMatrixExcluded{
			{Family: "awq_int4_group", Device: "cuda", Reason: "host-resident; no device upload path exists"},
		},
		KnownGaps: []string{"Metal and ROCm weight paths are unaudited"},
	}
	if mutate != nil {
		mutate(doc)
	}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	root := t.TempDir()
	dir := filepath.Join(root, "docs", "data")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir fixture tree: %v", err)
	}
	file := filepath.Join(dir, "packing-coverage.json")
	if err := os.WriteFile(file, b, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return root
}

func runPackingMatrixCapture(t *testing.T, argv ...string) (int, string, string) {
	t.Helper()
	var out, errOut strings.Builder
	code := run(&out, &errOut, argv)
	return code, out.String(), errOut.String()
}

func decodePackingMatrixReport(t *testing.T, out string) *packingMatrixReport {
	t.Helper()
	var rep packingMatrixReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("report is not valid JSON: %v\n%s", err, out)
	}
	return &rep
}

func TestPackingMatrixShowReport(t *testing.T) {
	root := packingMatrixFixture(t, nil)
	code, out, errOut := runPackingMatrixCapture(t, "packing-matrix", "show", "--root", root, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errOut)
	}
	rep := decodePackingMatrixReport(t, out)
	if rep.Schema != packingMatrixReportSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, packingMatrixReportSchema)
	}
	want := packingMatrixTotals{Cells: 3, Native: 1, RuntimeRepack: 1, Unknown: 1}
	if rep.Totals != want {
		t.Errorf("totals = %+v, want %+v", rep.Totals, want)
	}
	if rep.NativeRatio != "1/3" {
		t.Errorf("native_ratio = %q, want %q", rep.NativeRatio, "1/3")
	}
	if rep.Verdict != packingVerdictDefect {
		t.Errorf("verdict = %q, want %q", rep.Verdict, packingVerdictDefect)
	}
	if len(rep.Defects) != 1 || rep.Defects[0].Family != "nvfp4" || rep.Defects[0].Device != "cuda" {
		t.Errorf("defect_cells = %+v, want single nvfp4 x cuda row", rep.Defects)
	}
	if !strings.HasPrefix(rep.Defects[0].FirstEvidence, "DEFECT CELL") {
		t.Errorf("first_evidence = %q, want DEFECT CELL prefix", rep.Defects[0].FirstEvidence)
	}
	if len(rep.Unknowns) != 1 || rep.Unknowns[0].Family != "mxfp4_e8m0" {
		t.Errorf("unknown_cells = %+v, want single mxfp4_e8m0 row", rep.Unknowns)
	}
	if rep.Excluded != 1 || len(rep.KnownGaps) != 1 {
		t.Errorf("excluded=%d known_gaps=%d, want 1 and 1", rep.Excluded, len(rep.KnownGaps))
	}
	if rep.Root != root || rep.Source != packingMatrixSourcePath {
		t.Errorf("root/source = %q/%q, want %q/%q", rep.Root, rep.Source, root, packingMatrixSourcePath)
	}
}

func TestPackingMatrixHumanRender(t *testing.T) {
	root := packingMatrixFixture(t, nil)
	code, out, errOut := runPackingMatrixCapture(t, "packing-matrix", "show", "--root", root)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errOut)
	}
	for _, want := range []string{
		"native-ratio=1/3",
		"DEFECT nvfp4 x cuda",
		"UNKNOWN mxfp4_e8m0 x vulkan",
		"EXCLUDED pairs=1",
		"KNOWN GAPS:",
		"VERDICT runtime-repack-present",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("human report missing %q:\n%s", want, out)
		}
	}
}

func TestPackingMatrixFailOn(t *testing.T) {
	defective := packingMatrixFixture(t, nil)
	code, out, _ := runPackingMatrixCapture(t, "packing-matrix", "show", "--root", defective, "--fail-on", packingModeRuntimeRepack)
	if code != 1 {
		t.Fatalf("defective matrix with --fail-on runtime-repack: exit = %d, want 1\n%s", code, out)
	}

	clean := packingMatrixFixture(t, func(d *packingMatrixDoc) {
		d.Cells = []packingMatrixCell{d.Cells[0]}
	})
	code, out, _ = runPackingMatrixCapture(t, "packing-matrix", "show", "--root", clean, "--fail-on", packingModeRuntimeRepack, "--json")
	if code != 0 {
		t.Fatalf("clean matrix with --fail-on runtime-repack: exit = %d, want 0\n%s", code, out)
	}
	rep := decodePackingMatrixReport(t, out)
	if rep.Verdict != packingVerdictClean {
		t.Errorf("verdict = %q, want %q", rep.Verdict, packingVerdictClean)
	}
}

func TestPackingMatrixRefusals(t *testing.T) {
	cases := []struct {
		name   string
		root   string
		mutate func(*packingMatrixDoc)
		token  string
	}{
		{name: "missing file", root: t.TempDir(), token: "PACKING_MATRIX_NOT_FOUND"},
		{
			name:   "foreign schema",
			mutate: func(d *packingMatrixDoc) { d.Schema = "fak.packingmatrix.v0" },
			token:  "PACKING_MATRIX_SCHEMA_MISMATCH",
		},
		{
			name: "open mode value",
			mutate: func(d *packingMatrixDoc) {
				d.Cells[0].Mode = "row-major-swizzled"
			},
			token: "PACKING_MATRIX_MODE_UNKNOWN",
		},
		{
			name: "duplicate cell",
			mutate: func(d *packingMatrixDoc) {
				d.Cells = append(d.Cells, d.Cells[0])
			},
			token: "PACKING_MATRIX_DUPLICATE_CELL",
		},
		{
			name: "empty cells",
			mutate: func(d *packingMatrixDoc) {
				d.Cells = nil
			},
			token: "PACKING_MATRIX_EMPTY_CELLS",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := tc.root
			if tc.mutate != nil {
				root = packingMatrixFixture(t, tc.mutate)
			}
			code, out, errOut := runPackingMatrixCapture(t, "packing-matrix", "show", "--root", root, "--json")
			if code != 2 {
				t.Fatalf("exit = %d, want 2; stdout=%s stderr=%s", code, out, errOut)
			}
			if !strings.Contains(errOut, tc.token) {
				t.Errorf("stderr missing typed token %q:\n%s", tc.token, errOut)
			}
		})
	}
}

func TestPackingMatrixUsage(t *testing.T) {
	cases := [][]string{
		{},
		{"bogus"},
		{"show", "--fail-on", "swizzled"},
		{"show", "--root"},
		{"show", "--bogus-flag"},
	}
	for _, argv := range cases {
		full := append([]string{"packing-matrix"}, argv...)
		code, out, errOut := runPackingMatrixCapture(t, full...)
		if code != 2 {
			t.Errorf("argv=%v: exit = %d, want 2; stdout=%s", full, code, out)
		}
		if !strings.Contains(errOut, "usage:") {
			t.Errorf("argv=%v: stderr missing usage:\n%s", full, errOut)
		}
	}
}

func TestPackingMatrixRootDiscovery(t *testing.T) {
	root := packingMatrixFixture(t, nil)
	nested := filepath.Join(root, "cmd", "fak-dev")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(nested); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	defer func() {
		if err := os.Chdir(wd); err != nil {
			t.Fatalf("chdir back: %v", err)
		}
	}()

	code, out, errOut := runPackingMatrixCapture(t, "packing-matrix", "show", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errOut)
	}
	rep := decodePackingMatrixReport(t, out)
	if rep.Root != root {
		t.Errorf("discovered root = %q, want %q", rep.Root, root)
	}
}

func TestHelpAdvertisesPackingMatrix(t *testing.T) {
	var s strings.Builder
	writeHelp(&s)
	if !strings.Contains(s.String(), "packing-matrix show") {
		t.Errorf("help does not advertise the packing-matrix verb:\n%s", s.String())
	}
}
